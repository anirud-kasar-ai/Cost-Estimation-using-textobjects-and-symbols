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
    "You are an expert CAD / BIM low-voltage plan reviewer specializing in "
    "telecommunications, electrical, and technology floor plans on construction "
    "bid sets (AutoCAD/Revit plots). "
    "You understand contractor drafting conventions: solid filled triangles are "
    "DATA PERMANENT LINK drops (# = QTY), not direction arrows; open rectangles on "
    "walls with 2300/5400/5500 are Wiremold surface raceway; short letter tags "
    "(J, AP, WP, G) are equipment; +48 is elevation AFF and must be ignored; "
    "boxed keynote digits and room bubbles are not equipment counts. "
    "You read upright, rotated, and vertical text; letter tags; and numeric part labels "
    "with the same care as a licensed senior architect marking a bid set. "
    "Reply with a single JSON object only. No markdown, no prose."
)

_ARCHITECT_OCR_SYSTEM = (
    "You are a licensed senior architect and CAD/BIM blueprint reviewer "
    "specializing in low-voltage, electrical, telecommunications, and structural "
    "text extraction from construction drawing sets. "
    "You are expert at reading AutoCAD / Revit plot styles: thin linework, "
    "rotated annotation, vertical raceway labels, single-letter tags, and "
    "small numeric callouts that generic OCR often misses or confuses."
)

_ARCHITECT_OCR_TASK = (
    "Perform precise OCR and semantic parsing on architectural floor plan crops, "
    "callout bubbles, keynotes, equipment tags, and schedule fragments. "
    "Prioritize (A) alphabetic equipment / junction tags and (B) numeric raceway "
    "and quantity labels that are easy to miss when small, rotated, or near walls."
)

_ARCHITECT_OCR_EXTRACTION_RULES = [
    "1. LITERAL TRANSCRIBE: Extract raw text exactly as drawn. Preserve capitalization, "
    "line breaks, slash separators (e.g., 'RR / B'), and pipe delimiters (e.g., '1|13'). "
    "Do not invent characters that are not inked.",
    "2. NO INFERENCE: Do not autocorrect technical abbreviations (e.g., do not expand "
    "'E' to 'EXISTING' or 'WM' to 'WIREMOLD' unless explicitly written). "
    "Keep raceway numbers as drawn: '2300', '5400', '5500' — do not prepend 'WM' unless "
    "the letters WM are visible on the plan.",
    "3. SPATIAL GROUPING: Group multi-line text blocks by visual proximity "
    "(e.g., separate room name 'A-6E' from an adjacent drop tag '1|5' or QTY digit '3'). "
    "Never merge a room code with a nearby equipment tag.",
    "4. NOISE FILTERING: Ignore hatch, wall polylines, dimension arrows, grid bubbles, "
    "scale bars, and title-block boilerplate. Still extract small tags that sit ON walls "
    "or along raceway (J, AP, WP, 2300) — those are real annotations, not noise.",
    "5. ALPHABETIC TAGS (HIGH PRIORITY): Explicitly hunt for short letter tags drawn on "
    "the plan: single letters (J, G, R, M), two-letter tags (AP, WP, SB), and longer "
    "tags (NVR, STC, TGB, FACP, IDF, MDF). Read them even when tiny, rotated 90°, "
    "or touching a wall/symbol. Emit each instance separately in text_entities with "
    "category equipment_tag (or general_keynote for schedule-style notes).",
    "6. NUMERIC LABELS (HIGH PRIORITY): Explicitly hunt for numeric annotations: "
    "(a) raceway / Wiremold part digits 2300, 5400, 5500 (often vertical beside raceway); "
    "(b) boxed or free QTY digits 1–9 next to drop triangles / speakers; "
    "(c) pipe callouts like 1|5, 1|8, 1|13; "
    "(d) other part numbers with digits (WM2300, CAT6A). "
    "Emit raceway digits under raceway_conduit_notes and as text_entities with "
    "category raceway_conduit_note. Do not skip vertical or sideways numbers.",
    "7. ORIENTATION: Scan upright, 90° CW, 90° CCW, and near-vertical strings. "
    "On telecom plans, raceway model numbers are frequently stacked vertically; "
    "letter tags may sit below glyphs (AP under a circle) or beside horns (WP).",
    "8. OCR CONFUSION GUARD: Prefer plan context over look-alike glyphs. "
    "Common swaps: O↔0, I↔1↔l, S↔5, B↔8, Z↔2, G↔6. "
    "Near raceway prefer 2300/5400/5500 over letter-like misreads; "
    "near device glyphs prefer AP/WP/J over digit look-alikes. "
    "If a boxed digit sits beside a solid triangle, it is a QTY digit — transcribe the digit.",
    "9. OCR BOUNDING & CONFIDENCE: For EVERY text_entity provide bbox_rel as "
    "normalized 0.0–1.0 coordinates (origin top-left of the crop) tightly around the "
    "ink, plus confidence 0–1. Prefer many precise entities over one fused string.",
    "10. DENSE OVERLAP (NO DOUBLE ENTITIES): On zoom tiles, symbols/tags/QTY digits/"
    "pipe callouts (1|5) and elevations (+48) cluster tightly. Emit ONE text_entity "
    "per distinct ink cluster. Do not duplicate the same letters/digits because "
    "leader lines cross them or CV would also see them. Separate a triangle QTY digit "
    "from a nearby pipe tag and from room labels. Skip elevations (+48/+49) and "
    "sheet bubbles (1/D5) entirely.",
    "11. CAD CONSTRUCTION CONVENTIONS: Solid filled triangles on/near walls are DATA "
    "PERMANENT LINK drops (#), not arrows — always capture nearby QTY digits. Open "
    "rectangles on wall lines with 2300/5400/5500 are surface raceway. Boxed lone '1' "
    "keynotes and '+48' heights are not equipment tags. Map digits-only raceway labels "
    "to the matching WM series without inventing the letters WM.",
]


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


def _build_prompt(
    legend: SymbolTableInfo,
    cv_counts: dict[str, int],
    *,
    linear_keys: set[str] | None = None,
    extra_rules: list[str] | None = None,
    detection_hints: list[dict[str, Any]] | None = None,
) -> str:
    linear_keys = linear_keys or set()
    lines = [
        "This image is a technology / low-voltage floor plan (or dense zoom tile) from a CAD bid set.",
        "A computer-vision pass already counted legend symbols. Review the image like an",
        "expert CAD reviewer and return your best UNIQUE-INSTANCE count for every legend entry.",
        "",
        "Return JSON only:",
        '{"counts": {"<display_key>": <int>, ...}, "notes": "<short remark>"}',
        "",
        "Rules:",
        "- Keys must be copied EXACTLY from the legend list below (display_key).",
        "- Count visible instances on the plan linework only. Ignore the title block,",
        "  key plan, KEY PLAN inset, sheet callouts (1/D5), elevations (+48/+49),",
        "  dimension strings used as lengths, room labels (A-1, A-2, A-6E, RR/B), and any",
        "  legend table printed on the sheet.",
        "",
        "ALPHABETIC TAGS (count carefully — these are often missed):",
        "- Short letter tags on the plan (J, G, R, AP, WP, SB, NVR, STC, TGB, FACP, …)",
        "  each count once per visible instance under their legend tag key.",
        "- Read tags even when small, rotated, vertical, or sitting under/beside a glyph",
        "  (e.g. 'AP' under a circle-crosshair; 'WP' beside an exterior horn; a row of 'J'",
        "  along a wall or cable path).",
        "- Do not confuse letter tags with room codes (A-6E) or grid bubbles.",
        "",
        "NUMERIC LABELS (count carefully — these are often missed):",
        "- Numbers 2300 / 5400 / 5500 printed next to raceway (often VERTICAL) are Wiremold",
        "  part labels for SURFACE RACEWAY (WM2300 / WM5400 / WM5500). Count ONE per",
        "  visible raceway BOX + label cluster under the matching SURFACE RACEWAY (WMxxxx)",
        "  key — even if 'WM' letters are not printed.",
        "- Do NOT treat those raceway digits as architectural dimensions or room IDs.",
        "- The small SOLID FILLED black triangle with an optional digit beside it is the '#'",
        "  DATA PERMANENT LINK mark (telecom drop) — NOT a direction arrow / arrowhead.",
        "  The digit is quantity (# = QTY). Count each filled triangle as one mark.",
        "- Pipe-style / spaced callouts (1|5, 1 13, 1|13) are data-drop keynotes next to",
        "  drops; they are not raceway models and not a second '#' by themselves.",
        "- Elevations '+48'/'+49' are mounting height AFF only — never count them.",
        "- A square containing only '1' (keyed note) is usually NOT equipment unless the",
        "  legend explicitly uses that glyph as a counted tag.",
        "",
        "OTHER:",
        "- If a symbol does not appear, use 0. Include every legend entry.",
        "- Linear items (conduit, raceway runs, J-hook, ladder rack, data pole,",
        "  underground conduit) are NEVER counted as discrete glyph counts — always 0",
        "  for those keys (raceway TEXT labels like 2300 are still counted under SURFACE",
        "  RACEWAY keys when that key is a text/OCR entry, not [LINEAR]).",
        "- NEVER invent counts for keys marked [LINEAR — always 0].",
        "- Watch OCR look-alikes: O/0, I/1, S/5, B/8, Z/2 — prefer legend-consistent reads.",
        "- The CV counts / detection hints below are location aids; correct over-counts",
        "  from overlapping boxes and fill clear misses (especially missed '#' triangles",
        "  and missed 5400/2300 raceway labels on dense wall runs).",
    ]
    for block in extra_rules or []:
        if block and str(block).strip():
            lines.append("")
            lines.append(str(block).strip())
    lines.append("")
    lines.append("Legend entries (display_key | description | tag | part):")
    for entry in legend.entries:
        display = count_key(entry)
        if not display:
            continue
        tag = (entry.symbol or "").strip()
        part = (entry.part_number or "").strip()
        linear_note = " [LINEAR — always 0]" if display in linear_keys else ""
        lines.append(
            f"- {display!r}{linear_note} | {str(entry.description or '').strip()!r} | "
            f"tag={tag or '-'} | part={part or '-'}"
        )
    lines.append("")
    lines.append("CV counts (hints — may include overlap double-fires):")
    lines.append(json.dumps(cv_counts, ensure_ascii=False))
    if detection_hints:
        capped = detection_hints[:120]
        lines.append("")
        lines.append(
            "CV detection centers (symbol, source, qty, cx, cy) — merge overlaps to UNIQUE instances:"
        )
        lines.append(json.dumps(capped, ensure_ascii=False))
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
    parsed: dict[str, Any],
    valid_keys: set[str],
    *,
    linear_keys: set[str] | None = None,
) -> dict[str, int]:
    linear_keys = linear_keys or set()
    counts: dict[str, int] = {}
    raw = parsed.get("counts")
    if not isinstance(raw, dict):
        return counts
    for key, value in raw.items():
        key = str(key).strip()
        if key not in valid_keys or key in linear_keys:
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
    *,
    linear_keys: set[str] | None = None,
    extra_rules: list[str] | None = None,
    detection_hints: list[dict[str, Any]] | None = None,
    extra_429_retries: int | None = None,
) -> dict[str, Any]:
    """Ask the configured vision model to review CV counts.

    Returns {"ok": bool, "counts": {...} | None, "model": str,
             "provider": str, "notes": str | None, "error": str | None}.
    """
    import time

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
    prompt = _build_prompt(
        legend,
        cv_counts,
        linear_keys=linear_keys or set(),
        extra_rules=extra_rules,
        detection_hints=detection_hints,
    )
    image_b64 = _encode_image(image)
    retries = int(
        extra_429_retries
        if extra_429_retries is not None
        else 0
    )
    last_http: urllib.error.HTTPError | None = None
    content = ""

    for attempt in range(retries + 1):
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
            last_http = None
            break
        except urllib.error.HTTPError as exc:
            last_http = exc
            if exc.code == 429 and attempt < retries:
                time.sleep(float(config.FOLDER_VISION_429_BACKOFF_S))
                continue
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 429:
                result["error"] = (
                    f"{provider} API HTTP 429 (quota). CV counts kept — "
                    "retry later or set LLM_PROVIDER=groq. "
                    f"{detail or exc.reason}"
                )
            else:
                result["error"] = (
                    f"{provider} API HTTP {exc.code}: {detail or exc.reason}"
                )
            logger.warning("LLM verification failed: %s", result["error"])
            return result
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{provider} API call failed: {exc}"
            logger.warning("LLM verification failed: %s", result["error"])
            return result
    else:
        if last_http is not None:
            result["error"] = (
                f"{provider} API HTTP {last_http.code} after retries. CV counts kept."
            )
            return result

    parsed = _extract_json(str(content))
    if not parsed or not isinstance(parsed.get("counts"), dict):
        result["error"] = "Model reply was not valid counts JSON."
        return result

    counts = _normalize_counts(parsed, valid_keys, linear_keys=linear_keys or set())
    if not counts:
        result["error"] = "Model returned no counts matching legend keys."
        return result

    result["ok"] = True
    result["counts"] = counts
    note = parsed.get("notes")
    if isinstance(note, str) and note.strip():
        result["notes"] = note.strip()[:400]
    return result


def _build_judge_prompt(symbol_key: str, entry: SymbolEntry | None) -> str:
    desc = str(entry.description or "").strip() if entry else ""
    tag = (entry.symbol or "").strip() if entry else ""
    part = (entry.part_number or "").strip() if entry else ""
    lines = [
        "You are an expert CAD / BIM vision judge validating ONE detection on a",
        "low-voltage architectural floor plan crop.",
        "",
        "Images: (1) plan crop around a CV detection, (2) optional legend glyph reference.",
        "",
        "Legend target:",
        f"- display_key: {symbol_key!r}",
        f"- description: {desc!r}",
        f"- tag: {tag or '-'}",
        f"- part: {part or '-'}",
        "",
        "Reply JSON only:",
        '{"present": <bool>, "symbol_matches": <bool>, "confidence": <0..1>, "reason": "<short>"}',
        "",
        "Rules:",
        "- present=true only if real plan ink for a symbol/tag/label is visible",
        "  (not blank wall, hatch, dimension leader, room label alone, title block,",
        "  elevation +48, or a keyed-note square with only '1').",
        "- symbol_matches=true only if it matches the legend target (glyph, letter tag,",
        "  or numeric part label such as 2300/5400/WM2300).",
        "- '#' / DATA DROP: accept a SOLID FILLED triangle (any tip direction) as a drop",
        "  mark — on telecom plans these are NOT direction arrows. Optional adjacent",
        "  digit 1–9 is QTY, still one mark.",
        "- SURFACE RACEWAY: accept open rectangle on the wall line and/or digits",
        "  2300/5400/5500 (often vertical) for WM* targets.",
        "- ALPHABETIC: accept short tags (J, AP, WP, …) even if small/rotated when the",
        "  letters match the legend tag. Reject if the crop is only a room code (A-6E).",
        "- NUMERIC: accept raceway digits 2300/5400/5500 for SURFACE RACEWAY / WM*",
        "  targets. Reject pure architectural length dimensions and elevations (+48).",
        "- OCR text-only hits must match the expected tag or part digits to count.",
        "- Be strict on blank/ambiguous crops; when unsure, set present=false.",
        "- Prefer legend context over look-alike OCR swaps (O/0, I/1, S/5, B/8).",
    ]
    return "\n".join(lines)


def judge_detection_crop(
    crop: Image.Image,
    entry: SymbolEntry | None,
    symbol_key: str,
    glyph: Image.Image | None = None,
) -> dict[str, Any]:
    """Vision-judge a single merged detection crop against the legend entry."""
    provider = config.llm_provider()
    result: dict[str, Any] = {
        "ok": False,
        "present": False,
        "symbol_matches": False,
        "confidence": 0.0,
        "reason": None,
        "provider": provider,
        "model": config.llm_model(),
        "error": None,
    }
    if not config.llm_enabled():
        result["error"] = "LLM disabled"
        return result

    prompt = _build_judge_prompt(symbol_key, entry)
    crop_b64 = _encode_image(crop)
    glyph_b64 = _encode_image(glyph) if glyph is not None else None

    try:
        if provider == "gemini":
            parts: list[dict[str, Any]] = [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": crop_b64}}]
            if glyph_b64:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": glyph_b64}})
            payload = {
                "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 512,
                    "responseMimeType": "application/json",
                },
            }
            url = (
                f"{config.GEMINI_API_BASE}/models/{urllib.parse.quote(config.GEMINI_MODEL, safe='')}"
                f":generateContent?key={urllib.parse.quote(config.GEMINI_API_KEY)}"
            )
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "symbol-count-vision/0.2"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_S) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            content = _gemini_text(response)
        else:
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            ]
            if glyph_b64:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{glyph_b64}"}}
                )
            payload = {
                "model": config.GROQ_MODEL,
                "temperature": 0.1,
                "max_completion_tokens": 512,
                "reasoning_effort": "none",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            }
            response = _call_groq(payload)
            content = response["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result

    parsed = _extract_json(str(content))
    if not parsed:
        result["error"] = "Invalid judge JSON"
        return result

    result["ok"] = True
    result["present"] = bool(parsed.get("present"))
    result["symbol_matches"] = bool(parsed.get("symbol_matches"))
    try:
        result["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    reason = parsed.get("reason")
    if isinstance(reason, str) and reason.strip():
        result["reason"] = reason.strip()[:300]
    return result


def architect_ocr_prompt_spec(
    *,
    expected_context: str | None = None,
) -> dict[str, Any]:
    """Return the architect OCR prompt spec (for logging / API docs)."""
    ctx = (expected_context or config.VISION_OCR_EXPECTED_CONTEXT).strip()
    return {
        "system_instruction": _ARCHITECT_OCR_SYSTEM,
        "task": _ARCHITECT_OCR_TASK,
        "inputs": {
            "crop_image": (
                "Floor plan crop containing text annotations, alphabetic equipment tags "
                "(AP, WP, J), numeric raceway labels (2300/5400), room codes, or legend entries."
            ),
            "expected_context": ctx,
        },
        "extraction_rules": _ARCHITECT_OCR_EXTRACTION_RULES,
        "response_format": {
            "raw_text": "<string — all visible annotation text, line-separated>",
            "parsed_entities": {
                "room_labels": ["<e.g., A-WING (EAST), A-6E, RR/B>"],
                "equipment_tags": ["<e.g., AP, WP, J, G, SB, NVR, TGB>"],
                "data_drop_tags": ["<e.g., 1|5, 1|8, 1|13>"],
                "qty_digits": ["<e.g., 2, 3, 4 beside drop triangles>"],
                "raceway_conduit_notes": ["<e.g., 2300, 5400, 5500, WM2300>"],
                "general_keynotes": ["<e.g., T-800, T-801, OFCI>"],
            },
            "text_entities": [
                {
                    "text": "<literal transcribed string>",
                    "category": (
                        "room_label|equipment_tag|data_drop_tag|qty_digit|"
                        "raceway_conduit_note|general_keynote|other"
                    ),
                    "orientation": "upright|rotated_90|rotated_270|vertical|skewed",
                    "bbox_rel": {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0},
                    "confidence": 0.0,
                }
            ],
            "confidence": 0.0,
            "reasoning": (
                "<short sentence on OCR clarity, missed tiny tags/numbers, "
                "orientation, or look-alike risks>"
            ),
        },
    }


def _build_architect_ocr_prompt(*, expected_context: str | None = None) -> str:
    spec = architect_ocr_prompt_spec(expected_context=expected_context)
    ctx = spec["inputs"]["expected_context"]
    lines = [
        spec["task"],
        "",
        "Inputs:",
        "- crop_image: floor plan crop (attached)",
        f"- expected_context: {ctx!r}",
        "",
        "CAD focus checklist (do not skip):",
        "- Alphabetic: J, G, AP, WP, SB, NVR, STC, TGB, FACP and similar tags.",
        "- Numeric: 2300, 5400, 5500 (often vertical), QTY 1–9, pipe tags 1|n.",
        "- Orientations: upright + rotated + vertical stacks.",
        "- Keep room labels separate from equipment / raceway text.",
        "",
        "Extraction rules:",
        *[f"- {rule}" for rule in spec["extraction_rules"]],
        "",
        "Reply with a single JSON object only. No markdown, no prose.",
        "Use this schema exactly:",
        json.dumps(spec["response_format"], indent=2),
        "",
        "Every alphabetic tag and numeric raceway/QTY label that is visibly inked MUST",
        "appear in both parsed_entities (correct bucket) and text_entities (with bbox_rel).",
    ]
    return "\n".join(lines)


def _call_vision_json(
    *,
    prompt: str,
    image_b64: str,
    system_prompt: str,
    max_tokens: int = 4096,
) -> tuple[str | None, str | None]:
    """Call configured vision provider; return (content, error)."""
    provider = config.llm_provider()
    try:
        if provider == "gemini":
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
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
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                },
            }
            url = (
                f"{config.GEMINI_API_BASE}/models/{urllib.parse.quote(config.GEMINI_MODEL, safe='')}"
                f":generateContent?key={urllib.parse.quote(config.GEMINI_API_KEY)}"
            )
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "symbol-count-vision/0.2"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_S) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            return _gemini_text(response), None
        payload = {
            "model": config.GROQ_MODEL,
            "temperature": 0.1,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }
        response = _call_groq(payload)
        return str(response["choices"][0]["message"]["content"]), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def detect_legend_symbols_with_llm(
    image: Image.Image,
    legend: SymbolTableInfo,
    *,
    linear_keys: set[str] | None = None,
    max_instances: int = 80,
) -> dict[str, Any]:
    """Ask the vision model to locate legend symbols on a plan crop.

    Returns {"ok": bool, "detections": [{"symbol", "cx", "cy", "bbox_rel", ...}],
             "error": str|None, "model": str, "provider": str}.
    """
    linear_keys = linear_keys or set()
    provider = config.llm_provider()
    result: dict[str, Any] = {
        "ok": False,
        "detections": [],
        "error": None,
        "model": config.llm_model(),
        "provider": provider,
    }
    if not config.llm_enabled():
        result["error"] = "LLM disabled"
        return result

    legend_lines: list[str] = []
    valid: set[str] = set()
    for entry in legend.entries:
        key = count_key(entry)
        if not key:
            continue
        valid.add(key)
        tag = (entry.symbol or "").strip() or "-"
        part = (entry.part_number or "").strip() or "-"
        linear = " [LINEAR — do not emit boxes]" if key in linear_keys else ""
        legend_lines.append(
            f"- {key!r}{linear} | tag={tag} | part={part} | "
            f"desc={(entry.description or '').strip()!r}"
        )
    if not legend_lines:
        result["error"] = "Empty legend"
        return result

    prompt = "\n".join(
        [
            "Locate EVERY visible instance of the legend symbols on this CAD zoom tile.",
            "Return JSON only:",
            '{"detections":[{"symbol":"<display_key>","kind":"tag|drop|device|raceway|camera|other",'
            '"qty":1,"confidence":0.0,"bbox_rel":{"x1":0,"y1":0,"x2":0,"y2":0}}],'
            '"notes":"<short>"}',
            "",
            "Rules:",
            "- symbol MUST be copied EXACTLY from the legend display_key list.",
            "- bbox_rel is normalized 0–1 of THIS image (origin top-left), tight around the ink.",
            "- '#' DATA DROP = solid FILLED black triangle (any direction) + optional QTY digit.",
            "  Do NOT label direction arrows; filled triangles on walls/raceway are drops.",
            "  Set qty to the adjacent digit 1–9 when present, else 1.",
            "- Letter tags (J, AP, WP, G, M, …): box the LETTERS (and glyph if attached).",
            "  J on a DASHED cable path is often a J-HOOK run — still emit symbol 'J' only if",
            "  the legend tag J (JUNCTION BOX) matches a wall box/letter-J device, not a hook",
            "  string along dashed path. Prefer wall junction 'J' glyphs.",
            "- SURFACE RACEWAY: open rectangle on wall + digits 2300/5400/5500 → matching",
            "  SURFACE RACEWAY (WMxxxx) key.",
            "- IP CLOCK/SPEAKER (Bogen): horn/speaker glyph with '12:00' text → the CAT6A",
            "  IP CLOCK/SPEAKER legend key if present.",
            "- NETWORK CAMERA: hourglass (two opposing triangles + X-box).",
            "- IGNORE: +48/+49 elevations, room labels (A-6E), KEY PLAN, hexagon/square",
            "  keynote digits alone, doors, furniture.",
            "- Skip every [LINEAR — do not emit boxes] key.",
            f"- Cap at {max_instances} detections; prefer precision over inventing marks.",
            "",
            "Legend display_keys:",
            *legend_lines,
        ]
    )
    image_b64 = _encode_image(image)
    content, err = _call_vision_json(
        prompt=prompt,
        image_b64=image_b64,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=8192,
    )
    if err:
        result["error"] = err
        return result
    parsed = _extract_json(str(content or ""))
    if not parsed or not isinstance(parsed.get("detections"), list):
        result["error"] = "Model reply was not valid detections JSON."
        return result

    img_w, img_h = image.size
    out: list[dict[str, Any]] = []
    for item in parsed["detections"]:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if symbol not in valid or symbol in linear_keys:
            continue
        bbox = item.get("bbox_rel")
        if not isinstance(bbox, dict):
            continue
        try:
            x1 = float(bbox["x1"]) * img_w
            y1 = float(bbox["y1"]) * img_h
            x2 = float(bbox["x2"]) * img_w
            y2 = float(bbox["y2"]) * img_h
        except (KeyError, TypeError, ValueError):
            continue
        xa, xb = sorted((x1, x2))
        ya, yb = sorted((y1, y2))
        if xb - xa < 2 or yb - ya < 2:
            continue
        try:
            conf = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        try:
            qty = max(1, int(item.get("qty") or 1))
        except (TypeError, ValueError):
            qty = 1
        out.append(
            {
                "symbol": symbol,
                "kind": str(item.get("kind") or "other"),
                "qty": qty,
                "confidence": max(0.0, min(1.0, conf)),
                "x1": xa,
                "y1": ya,
                "x2": xb,
                "y2": yb,
            }
        )
        if len(out) >= max_instances:
            break

    result["ok"] = True
    result["detections"] = out
    note = parsed.get("notes")
    if isinstance(note, str) and note.strip():
        result["notes"] = note.strip()[:400]
    return result


def parse_crop_text_with_llm(
    crop: Image.Image,
    *,
    expected_context: str | None = None,
) -> dict[str, Any]:
    """Vision OCR + semantic parsing on a floor plan crop (architect prompt)."""
    result: dict[str, Any] = {
        "ok": False,
        "raw_text": None,
        "parsed_entities": None,
        "text_entities": None,
        "confidence": None,
        "reasoning": None,
        "provider": config.llm_provider(),
        "model": config.llm_model(),
        "error": None,
    }
    if not config.vision_ocr_enabled():
        result["error"] = "Vision OCR disabled (SYMBOL_COUNT_USE_VISION_OCR=false or no API key)."
        return result

    prompt = _build_architect_ocr_prompt(expected_context=expected_context)
    image_b64 = _encode_image(crop)
    content, err = _call_vision_json(
        prompt=prompt,
        image_b64=image_b64,
        system_prompt=_ARCHITECT_OCR_SYSTEM,
        max_tokens=4096,
    )
    if err:
        result["error"] = err
        return result

    parsed = _extract_json(str(content or ""))
    if not parsed:
        result["error"] = "Model reply was not valid architect OCR JSON."
        return result

    result["ok"] = True
    raw = parsed.get("raw_text")
    if isinstance(raw, str):
        result["raw_text"] = raw.strip()
    entities = parsed.get("parsed_entities")
    if isinstance(entities, dict):
        result["parsed_entities"] = entities
    text_entities = parsed.get("text_entities")
    if isinstance(text_entities, list):
        result["text_entities"] = text_entities
    try:
        result["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    reasoning = parsed.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        result["reasoning"] = reasoning.strip()[:400]
    return result


__all__ = [
    "architect_ocr_prompt_spec",
    "detect_legend_symbols_with_llm",
    "judge_detection_crop",
    "parse_crop_text_with_llm",
    "verify_counts_with_llm",
]
