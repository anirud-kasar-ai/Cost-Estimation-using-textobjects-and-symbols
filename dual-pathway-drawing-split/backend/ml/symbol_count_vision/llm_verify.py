"""Vision LLM verification for CV symbol counts.

Supports Gemini (default, free-tier friendly) and Groq. The CV pipeline
produces deterministic counts; this module sends the plan image, legend, and
CV counts to a multimodal model for review. On any failure the caller keeps
the CV counts.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Any

from PIL import Image

from pipeline import config
from pipeline.cv.tag_match import count_key
from pipeline.extraction.symbol_table_extractor import SymbolTableInfo

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You verify symbol counts on architectural/technology floor plan drawings. "
    "Reply with a single JSON object only. No markdown, no prose."
)


def _encode_image(image: Image.Image) -> str:
    img = image.convert("RGB")
    max_dim = max(img.size)
    limit = int(config.LLM_MAX_IMAGE_DIM)
    if max_dim > limit:
        scale = limit / max_dim
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_prompt(legend: SymbolTableInfo, cv_counts: dict[str, int]) -> str:
    lines = [
        "This image is a floor plan (or wing crop) from a construction drawing set.",
        "A computer-vision pass already counted legend symbols on it. Review the",
        "image and return your own best count for every legend entry below.",
        "",
        "Return JSON only:",
        '{"counts": {"<display_key>": <int>, ...}, "notes": "<short remark>"}',
        "",
        "Rules:",
        "- Keys must be copied EXACTLY from the legend list below (display_key).",
        "- Count visible instances on the plan linework only. Ignore the title,",
        "  key plan, dimensions, room labels, and any legend table on the image.",
        "- Numbers like 2300 / 5400 / 5500 printed next to raceway are Wiremold",
        "  part labels (WM2300 / WM5400 / WM5500) — count them under the matching",
        "  SURFACE RACEWAY (WMxxxx) legend key, one per visible label.",
        "- The small solid black triangle with an optional digit beside it is the",
        "  '#' DATA PERMANENT LINK mark; the digit is its quantity (# = QTY).",
        "- Letter tags on the plan (G, J, AP, NVR, STC, TGB, FACP, …) count under",
        "  their legend tag key.",        "- If a symbol does not appear, use 0. Include every legend entry.",
        "- Linear items (conduit, raceway) are drawn as long runs: count separate",
        "  runs/segments, not repeated pattern marks along one run.",
        "- The CV counts below are hints; correct them where they look wrong.",
        "",
        "Legend entries (display_key | description | tag | part):",
    ]
    for entry in legend.entries:
        display = count_key(entry)
        if not display:
            continue
        tag = (entry.symbol or "").strip()
        part = (entry.part_number or "").strip()
        lines.append(
            f"- {display!r} | {str(entry.description or '').strip()!r} | "
            f"tag={tag or '-'} | part={part or '-'}"
        )
    lines.append("")
    lines.append("CV counts (hints):")
    lines.append(json.dumps(cv_counts, ensure_ascii=False))
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _call_groq(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        config.GROQ_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "symbol-count-vision/0.2 (python-urllib)",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_gemini(image_b64: str, prompt: str) -> dict[str, Any]:
    model = config.GEMINI_MODEL
    url = (
        f"{config.GEMINI_API_BASE}/models/{urllib.parse.quote(model, safe='')}"
        f":generateContent?key={urllib.parse.quote(config.GEMINI_API_KEY)}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "symbol-count-vision/0.2 (python-urllib)",
        },
        method="POST",
    )
    # Free-tier Flash models briefly 503 under load; retry a couple times.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 503} or attempt == 2:
                raise
            import time

            time.sleep(2.0 * (attempt + 1))
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "symbol-count-vision/0.2 (python-urllib)",
                },
                method="POST",
            )
    assert last_exc is not None
    raise last_exc

def _gemini_text(response: dict[str, Any]) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    texts: list[str] = []
    for part in parts or []:
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts)


def _normalize_counts(
    parsed: dict[str, Any], valid_keys: set[str]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw = parsed.get("counts")
    if not isinstance(raw, dict):
        return counts
    for key, value in raw.items():
        key = str(key).strip()
        if key not in valid_keys:
            continue
        try:
            counts[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return counts


def verify_counts_with_llm(
    image: Image.Image,
    legend: SymbolTableInfo,
    cv_counts: dict[str, int],
) -> dict[str, Any]:
    """Ask the configured vision model to review CV counts.

    Returns {"ok": bool, "counts": {...} | None, "model": str,
             "provider": str, "notes": str | None, "error": str | None}.
    """
    provider = config.llm_provider()
    result: dict[str, Any] = {
        "ok": False,
        "counts": None,
        "model": config.llm_model(),
        "provider": provider,
        "notes": None,
        "error": None,
    }
    if not config.llm_enabled():
        result["error"] = (
            "LLM verification disabled (no API key for "
            f"{provider}, or SYMBOL_COUNT_USE_LLM=false)."
        )
        return result

    valid_keys = {
        count_key(e)
        for e in legend.entries
        if count_key(e)
    }
    prompt = _build_prompt(legend, cv_counts)
    image_b64 = _encode_image(image)

    try:
        if provider == "gemini":
            response = _call_gemini(image_b64, prompt)
            content = _gemini_text(response)
        else:
            payload = {
                "model": config.GROQ_MODEL,
                "temperature": 0.1,
                "max_completion_tokens": 4096,
                "reasoning_effort": "none",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
            }
            response = _call_groq(payload)
            content = response["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        result["error"] = f"{provider} API HTTP {exc.code}: {detail or exc.reason}"
        logger.warning("LLM verification failed: %s", result["error"])
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{provider} API call failed: {exc}"
        logger.warning("LLM verification failed: %s", result["error"])
        return result

    parsed = _extract_json(str(content))
    if not parsed or not isinstance(parsed.get("counts"), dict):
        result["error"] = "Model reply was not valid counts JSON."
        return result

    counts = _normalize_counts(parsed, valid_keys)
    if not counts:
        result["error"] = "Model returned no counts matching legend keys."
        return result

    result["ok"] = True
    result["counts"] = counts
    note = parsed.get("notes")
    if isinstance(note, str) and note.strip():
        result["notes"] = note.strip()[:400]
    return result


__all__ = ["verify_counts_with_llm"]
