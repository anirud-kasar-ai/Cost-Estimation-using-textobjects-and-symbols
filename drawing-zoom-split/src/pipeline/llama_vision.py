"""Vision LLM client for drawing/wing boxes (OpenCV does the crops).

Runtimes:
  - groq   → Groq OpenAI-compatible multimodal chat (default)
  - hosted → Hugging Face Inference API / dedicated Endpoint
  - local  → Transformers MllamaForConditionalGeneration
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
from io import BytesIO
from typing import Any

from PIL import Image

from pipeline import config
from pipeline.extraction.symbol_table_extractor import catalog_id_for_entry

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTANCE: "LlamaVisionClient | None" = None

CLASSIFY_PROMPT = """You are deciding whether an architectural/engineering sheet contains plan drawings.

Return JSON only, no markdown:
{"has_diagram": true, "plan_area": [x1, y1, x2, y2], "reason": "short"}

Rules:
- Coordinates are normalized 0-1 (origin top-left).
- has_diagram is true when the sheet contains floor plans, site plans, RCP, roof
  plans, schematics, rack elevations, or single-line diagrams of a building.
- plan_area must be ONE box that covers ALL plan linework on the sheet together.
  Include every wing/building plan panel and their local labels/callouts.
- NEVER include tables, schedules, legends, general notes, sheet notes, title
  block, key plan, text-only panels, symbol lists, photos, or empty margins.
- has_diagram is false for cover sheets, index pages, legends-only sheets,
  notes-only sheets, specification sheets, photo-detail sheets, and sheets of
  typical details (flashing, curb, drain) that do not also show a building plan.
  If false, omit plan_area.
"""

WINGS_PROMPT = """This image is a cropped architectural drawing that may hold several
plans of different parts of a building on one sheet.

Each plan carries a printed title set in the largest, boldest type on the
drawing — a wing, a building, an area, a level, or a roof. Find EVERY such
titled plan.

Return JSON only, no markdown:
{"wings": [{"name": "<title exactly as printed>", "box": [x1, y1, x2, y2]}]}

Rules:
- Coordinates are normalized 0-1 (origin top-left) on THIS cropped image.
- Copy each name from the drawing itself. Never invent, translate, abbreviate
  or expand a title, and never reuse a name from another sheet.
- Return one entry per titled plan. If the sheet shows 4, return 4.
- Each box must cover the COMPLETE plan under that title — all rooms, walls
  and linework — not just the title text.
- Order plans top-to-bottom, then left-to-right.
- A title may sit above, below or beside the plan it names.
- Boxes may touch but should not heavily overlap.
- Exclude tables, schedules, legends, notes, title blocks, key plans, scale
  bars, and text-only panels.
- If there is one undivided plan with no title of its own, return a single
  region named FULL-PLAN covering the whole drawing.
"""

RETRY_PROMPT = "Your previous reply was not valid JSON. Reply with JSON only, no markdown, matching the schema."

SYMBOL_DETECT_SYSTEM = (
    "You detect device/symbol instances on architectural floor plan crops. "
    "Reply with a single JSON object only. No markdown."
)


def _build_symbol_detect_prompt(entries: list[Any]) -> str:
    lines = [
        "This image is a cropped wing floor plan from an architectural sheet.",
        "Detect every visible instance of symbols from the catalog below.",
        "",
        "Return JSON only:",
        '{"detections": [{"symbol": "<catalog id>", "box": [x1, y1, x2, y2], "confidence": 0.0}]}',
        "",
        "Rules:",
        "- Coordinates are normalized 0-1 on THIS image (origin top-left).",
        "- symbol must match a catalog Symbol id exactly (text tag or description when no tag).",
        "- Count graphical legend symbols (cameras, raceway, mounts, drops) by visual shape,",
        "  not only text labels on the plan.",
        "- One tight box per symbol instance on the plan linework.",
        "- Ignore room labels, dimensions, notes, title text, scale bars, and legend tables.",
        "- If none found, return {\"detections\": []}.",
        "",
        "Symbol catalog:",
    ]
    for entry in entries:
        symbol = getattr(entry, "symbol", None) or entry.get("symbol") if isinstance(entry, dict) else None
        description = getattr(entry, "description", None) or (entry.get("description") if isinstance(entry, dict) else "")
        mfg = getattr(entry, "mfg_model", None) or (entry.get("mfg_model") if isinstance(entry, dict) else None)
        part = getattr(entry, "part_number", None) or (entry.get("part_number") if isinstance(entry, dict) else None)
        if isinstance(entry, dict):
            symbol_label = str(symbol or description or "").strip()
            if not symbol and description and part:
                symbol_label = f"{str(description).strip()} ({str(part).strip()})"
        else:
            symbol_label = catalog_id_for_entry(entry)
        if not symbol_label:
            continue
        row = f"- Symbol={symbol_label!r} | Description={str(description or '').strip()!r}"
        if mfg:
            row += f" | MFG/Model={str(mfg).strip()!r}"
        if part:
            row += f" | Part={str(part).strip()!r}"
        lines.append(row)
    return "\n".join(lines)


def downscale_for_llama(image: Image.Image, max_side: int | None = None) -> Image.Image:
    """Shrink the long side so the vision model sees layout, not a 300 DPI sheet."""
    max_side = int(max_side or config.VISION_LLAMA_MAX_SIDE)
    rgb = image.convert("RGB")
    w, h = rgb.size
    long_side = max(w, h)
    if long_side <= max_side:
        return rgb
    scale = max_side / float(long_side)
    return rgb.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (strips markdown fences)."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(cleaned[start : end + 1])


def _normalize_box(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = vals
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return None
    return [x1, y1, x2, y2]


def _image_to_data_url(image: Image.Image, *, quality: int = 88) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class LlamaVisionClient:
    """Vision LLM client used only for structured boxes; OpenCV does crops."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.hosted_client = None
        self.groq_ready = False
        self.runtime = config.VISION_LLM_RUNTIME
        self.model_id = config.GROQ_MODEL_ID if self.runtime == "groq" else config.HF_MODEL_ID
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if self.runtime == "groq":
            self._load_groq()
            return
        if self.runtime == "hosted":
            self._load_hf_hosted()
            return
        if self.runtime == "local":
            self._load_local()
            return
        self._load_error = (
            f"Unknown VISION_LLM_RUNTIME={self.runtime!r}. "
            "Use groq, hosted, or local."
        )
        logger.error(self._load_error)

    def _load_groq(self) -> None:
        if not config.GROQ_API_KEY:
            self._load_error = (
                "GROQ_API_KEY is empty. Set it in drawing-zoom-split/.env "
                "(create a key at https://console.groq.com/keys)."
            )
            logger.error(self._load_error)
            return
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            self._load_error = f"httpx is required for Groq runtime: {exc}"
            logger.exception(self._load_error)
            return
        self.groq_ready = True
        self.model_id = config.GROQ_MODEL_ID
        logger.info("Groq vision client ready: %s", self.model_id)

    def _load_hf_hosted(self) -> None:
        if not config.HF_TOKEN:
            self._load_error = (
                "HF_TOKEN is empty. Accept the Llama license on Hugging Face "
                "and set HF_TOKEN in .env"
            )
            logger.error(self._load_error)
            return
        try:
            from huggingface_hub import InferenceClient

            endpoint = config.HF_INFERENCE_ENDPOINT or config.HF_MODEL_ID
            self.hosted_client = InferenceClient(
                model=endpoint,
                token=config.HF_TOKEN,
                timeout=300,
            )
            self.model_id = endpoint
            logger.info("Hugging Face hosted client ready: %s", endpoint)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Could not initialize HF hosted inference: {exc}"
            logger.exception(self._load_error)

    def _load_local(self) -> None:
        if not config.HF_TOKEN:
            self._load_error = (
                "HF_TOKEN is empty. Accept the Llama 3.2 Vision license on Hugging Face "
                "and set HF_TOKEN in .env"
            )
            logger.error(self._load_error)
            return

        try:
            import torch
            from huggingface_hub import login
            from transformers import AutoProcessor, MllamaForConditionalGeneration
        except ImportError as exc:
            self._load_error = (
                f"Missing ML dependency: {exc}. Install requirements.txt and a CUDA torch build."
            )
            logger.exception(self._load_error)
            return

        if not torch.cuda.is_available() and not config.VISION_ALLOW_CPU_MODEL:
            self._load_error = (
                "Local Llama 3.2 Vision 11B needs a CUDA GPU for practical inference. "
                "Set VISION_ALLOW_CPU_MODEL=true only if this machine has enough RAM."
            )
            logger.warning(self._load_error)
            return

        try:
            login(token=config.HF_TOKEN, add_to_git_credential=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("huggingface_hub.login failed: %s", exc)

        model_id = config.HF_MODEL_ID
        device = config.VISION_DEVICE
        load_4bit = config.HF_LOAD_4BIT and torch.cuda.is_available() and device != "cpu"
        kwargs: dict[str, Any] = {"device_map": "auto" if device == "auto" else None}
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        elif device == "cpu":
            kwargs["device_map"] = {"": "cpu"}

        try:
            if load_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                kwargs["torch_dtype"] = dtype

            logger.info("Loading %s (4bit=%s) …", model_id, load_4bit)
            self.processor = AutoProcessor.from_pretrained(model_id, token=config.HF_TOKEN)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                token=config.HF_TOKEN,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
            self.model.eval()
            self.model_id = model_id
            logger.info("Local Llama 3.2 Vision ready")
        except Exception as exc:  # noqa: BLE001
            if load_4bit:
                logger.warning("4-bit load failed (%s) — retrying fp16/bf16", exc)
                try:
                    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    self.processor = AutoProcessor.from_pretrained(
                        model_id, token=config.HF_TOKEN
                    )
                    self.model = MllamaForConditionalGeneration.from_pretrained(
                        model_id,
                        token=config.HF_TOKEN,
                        torch_dtype=dtype,
                        device_map="auto",
                    )
                    self.model.eval()
                    self.model_id = model_id
                    logger.info("Local Llama 3.2 Vision ready (non-quantized fallback)")
                    return
                except Exception as retry_exc:  # noqa: BLE001
                    self._load_error = str(retry_exc)
                    logger.exception("Failed to load local Llama 3.2 Vision")
                    return
            self._load_error = str(exc)
            logger.exception("Failed to load local Llama 3.2 Vision")

    @property
    def ready(self) -> bool:
        return (
            self.groq_ready
            or self.hosted_client is not None
            or (self.model is not None and self.processor is not None)
        )

    def _device(self):
        return next(self.model.parameters()).device

    def _generate_groq(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        *,
        system_prompt: str | None = None,
    ) -> str:
        import httpx

        image_url = _image_to_data_url(image, quality=80)
        payload: dict[str, Any] = {
            "model": config.GROQ_MODEL_ID,
            "temperature": 0.0,
            "max_completion_tokens": max(max_new_tokens, 1200),
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    or (
                        "You extract structured layout boxes from architectural drawings. "
                        "Reply with a single JSON object only. No markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        # Qwen 3.6 defaults to thinking; that breaks Groq json_object validation.
        if "qwen" in config.GROQ_MODEL_ID.lower():
            payload["reasoning_effort"] = "none"

        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        import time as _time

        max_retries = 5
        for attempt in range(max_retries):
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if (
                    response.status_code == 400
                    and "json_validate_failed" in (response.text or "")
                ):
                    payload.pop("response_format", None)
                    payload["reasoning_effort"] = "none"
                    response = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
            if response.status_code == 429 and attempt < max_retries - 1:
                wait = min(2 ** attempt + 1, 30)
                logger.warning("Groq 429 rate-limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                _time.sleep(wait)
                continue
            break
        if response.status_code >= 400:
            detail = response.text[:800]
            raise RuntimeError(f"Groq API {response.status_code}: {detail}")
        data = response.json()
        message = data["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        return content

    def _generate_hf_hosted(
        self, image: Image.Image, prompt: str, max_new_tokens: int
    ) -> str:
        image_url = _image_to_data_url(image)
        with _LOCK:
            output = self.hosted_client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
        return str(output.choices[0].message.content or "").strip()

    def _generate_local(
        self, image: Image.Image, prompt: str, max_new_tokens: int
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        device = self._device()
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with _LOCK:
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        return self.processor.decode(
            output[0][prompt_len:], skip_special_tokens=True
        ).strip()

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 800,
        *,
        system_prompt: str | None = None,
    ) -> str:
        if not self.ready:
            raise RuntimeError(self._load_error or "Vision LLM is not loaded")

        work = downscale_for_llama(image)
        if self.groq_ready:
            return self._generate_groq(
                work, prompt, max_new_tokens, system_prompt=system_prompt
            )
        if self.hosted_client is not None:
            return self._generate_hf_hosted(work, prompt, max_new_tokens)
        return self._generate_local(work, prompt, max_new_tokens)

    def _ask_json(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        raw = self._generate(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )
        try:
            return parse_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Invalid JSON from vision model, retrying once")
            raw = self._generate(
                image,
                f"{prompt}\n\n{RETRY_PROMPT}",
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )
            return parse_json_object(raw)

    def classify_page(self, image: Image.Image) -> dict[str, Any]:
        """Return has_diagram and a single plan_area box covering all plan linework."""
        data = self._ask_json(image, CLASSIFY_PROMPT)
        plan_area = _normalize_box(data.get("plan_area"))
        if plan_area is None:
            plan_area = _normalize_box(data.get("diagram_box"))
        # Legacy multi-drawing responses: union all boxes into one plan_area.
        if plan_area is None:
            drawings_raw = data.get("drawings")
            if isinstance(drawings_raw, list):
                boxes: list[list[float]] = []
                for item in drawings_raw:
                    if isinstance(item, dict):
                        box = _normalize_box(item.get("box"))
                        if box is not None:
                            boxes.append(box)
                if boxes:
                    plan_area = [
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    ]
        has_diagram = bool(data.get("has_diagram")) and plan_area is not None
        reason = str(data.get("reason") or "").strip()
        return {
            "has_diagram": has_diagram,
            "plan_area": plan_area,
            "reason": reason,
            "raw": data,
        }

    def locate_wings(self, diagram: Image.Image) -> list[dict[str, Any]]:
        """Return [{name, box, source}] with normalized boxes on the cropped diagram."""
        data = self._ask_json(diagram, WINGS_PROMPT)
        wings_raw = data.get("wings")
        if not isinstance(wings_raw, list):
            return []
        from pipeline.wing_crop import assign_wing_instance_ids, slugify

        wings: list[dict[str, Any]] = []
        for item in wings_raw:
            if not isinstance(item, dict):
                continue
            name = slugify(str(item.get("name") or "FULL-PLAN"))
            box = _normalize_box(item.get("box"))
            if box is None:
                continue
            wings.append({"name": name, "box": box, "source": "vision_model"})
        return assign_wing_instance_ids(wings)

    def detect_symbols_in_tile(
        self,
        tile_image: Image.Image,
        legend_entries: list[Any],
    ) -> list[dict[str, Any]]:
        """Detect catalog symbol instances in one plan zoom tile."""
        if not legend_entries:
            return []
        prompt = _build_symbol_detect_prompt(legend_entries)
        data = self._ask_json(
            tile_image,
            prompt,
            max_new_tokens=2000,
            system_prompt=SYMBOL_DETECT_SYSTEM,
        )
        detections_raw = data.get("detections")
        if not isinstance(detections_raw, list):
            return []
        valid_symbols: set[str] = set()
        for entry in legend_entries:
            if isinstance(entry, dict):
                sym = str(entry.get("symbol") or "").strip()
                desc = str(entry.get("description") or "").strip()
                part = str(entry.get("part_number") or "").strip()
                if sym:
                    valid_symbols.add(sym)
                elif desc and part:
                    valid_symbols.add(f"{desc} ({part})")
                elif desc:
                    valid_symbols.add(desc)
            else:
                cid = catalog_id_for_entry(entry)
                if cid:
                    valid_symbols.add(cid)
                sym = str(getattr(entry, "symbol", None) or "").strip()
                desc = str(getattr(entry, "description", None) or "").strip()
                part = str(getattr(entry, "part_number", None) or "").strip()
                if sym:
                    valid_symbols.add(sym)
                if desc:
                    valid_symbols.add(desc)
                    valid_symbols.add(desc.upper())
                if desc and part:
                    valid_symbols.add(f"{desc} ({part})")
        out: list[dict[str, Any]] = []
        for item in detections_raw:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").strip()
            box = _normalize_box(item.get("box"))
            if symbol not in valid_symbols or box is None:
                continue
            try:
                confidence = float(item.get("confidence", item.get("score", 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            out.append(
                {
                    "symbol": symbol,
                    "box": box,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return out


def get_llama_client() -> LlamaVisionClient:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = LlamaVisionClient()
    return _INSTANCE


def reset_llama_client() -> None:
    """Drop the singleton so a new runtime/key can be loaded after .env changes."""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None
