"""Vision LLM verification for CV symbol counts.

Supports Gemini (default, free-tier friendly) and Groq. The CV pipeline
produces deterministic counts; this module sends the plan image, legend, and
CV counts to a multimodal model for review. On any failure the caller keeps
the CV counts.
"""

from __future__ import annotations

import base64
import hashlib
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
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert CAD / BIM low-voltage plan reviewer specializing in "
    "telecommunications, electrical, and technology floor plans on construction "
    "bid sets (AutoCAD/Revit plots). "
    "You understand contractor drafting conventions: a STANDALONE solid filled "
    "triangle is a DATA PERMANENT LINK drop (# = QTY), not a direction arrow, and "
    "not every filled triangle (AP circles, DATA POLE bowties, and camera "
    "hourglasses also contain triangles); "
    "open rectangles on "
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
    glyph_descs: dict[str, str] | None = None,
    reference_glyphs_attached: bool = False,
) -> str:
    linear_keys = linear_keys or set()
    glyph_descs = glyph_descs or {}
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
        "- Linear items (continuous conduit, raceway runs, ladder rack,",
        "  underground conduit) are NEVER counted as discrete glyph counts — always 0",
        "  for those keys (raceway TEXT labels like 2300 are still counted under SURFACE",
        "  RACEWAY keys when that key is a text/OCR entry, not [LINEAR]).",
        "- J-HOOK, CONDUIT STUB, and DATA POLE are discrete look-alikes — count them",
        "  using the LOOK-ALIKE rules (do not zero them as linear runs).",
        "- NEVER invent counts for keys marked [LINEAR — always 0].",
        "- Watch OCR look-alikes: O/0, I/1, S/5, B/8, Z/2 — prefer legend-consistent reads.",
        "- The CV counts / detection hints below are location aids; correct over-counts",
        "  from overlapping boxes and fill clear misses (especially missed '#' triangles",
        "  and missed 5400/2300 raceway labels on dense wall runs).",
        "",
        _LOOKALIKE_CONFUSION_TABLE,
    ]
    for block in extra_rules or []:
        if block and str(block).strip():
            lines.append("")
            lines.append(str(block).strip())
    lines.append("")
    lines.append(
        "Legend entries from the uploaded technical-symbol sheet "
        "(display_key | description | tag | part | glyph shape):"
    )
    for entry in legend.entries:
        display = count_key(entry)
        if not display:
            continue
        tag = (entry.symbol or "").strip()
        part = (entry.part_number or "").strip()
        linear_note = " [LINEAR — always 0]" if display in linear_keys else ""
        glyph_note = (glyph_descs.get(display) or "").strip()
        lines.append(
            f"- {display!r}{linear_note} | {str(entry.description or '').strip()!r} | "
            f"tag={tag or '-'} | part={part or '-'} | glyph={glyph_note or '-'}"
        )
    if reference_glyphs_attached:
        lines.append("")
        lines.append(
            "The actual legend glyph artwork (cropped from the uploaded technical-symbol"
        )
        lines.append(
            "sheet) is attached after the plan image, each labeled with its legend key."
        )
        lines.append(
            "Those reference images define what each symbol looks like — count only marks"
        )
        lines.append(
            "matching them (or the exact tag/part text). Never count the references themselves."
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


def _reference_image_parts_gemini(
    reference_images: list[tuple[str, str]] | None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for label, ref_b64 in reference_images or []:
        parts.append(
            {
                "text": (
                    f"REFERENCE GLYPH — legend key {label!r} "
                    "(legend artwork only, NOT part of the plan):"
                )
            }
        )
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}})
    return parts


def _post_gemini(
    model: str,
    payload: dict[str, Any],
    *,
    retry_429: bool = True,
) -> dict[str, Any]:
    url = (
        f"{config.GEMINI_API_BASE}/models/{urllib.parse.quote(model, safe='')}"
        f":generateContent?key={urllib.parse.quote(config.GEMINI_API_KEY)}"
    )
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "symbol-count-vision/0.2 (python-urllib)",
    }
    # 503: free-tier Flash briefly overloaded — short retry.
    # 429 (when retried): usually a per-minute token window — wait it out,
    # since folder jobs fire many image-heavy calls back to back.
    last_exc: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            retriable = exc.code == 503 or (exc.code == 429 and retry_429)
            if not retriable or attempt == 2:
                raise
            import time

            if exc.code == 429:
                time.sleep(20.0 * (attempt + 1))
            else:
                time.sleep(2.0 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


# After the primary model fails with quota/not-found, skip straight to the
# fallback for a while instead of re-trying (and re-backing-off) on every call.
_PRIMARY_SKIP_UNTIL = 0.0
_PRIMARY_SKIP_S = 600.0
_LAST_GEMINI_MODEL = ""


def _fallback_models(primary: str) -> list[str]:
    out: list[str] = []
    for part in (config.GEMINI_FALLBACK_MODEL or "").split(","):
        name = part.strip()
        if name and name != primary and name not in out:
            out.append(name)
    return out


def gemini_model_used() -> str:
    """Model that last succeeded (may be a fallback), else the configured primary."""
    return _LAST_GEMINI_MODEL or config.GEMINI_MODEL


def _stamp_model(result: dict[str, Any]) -> None:
    if result.get("provider") == "gemini":
        result["model"] = gemini_model_used()


def _gemini_generate(payload: dict[str, Any]) -> dict[str, Any]:
    """POST to Gemini 2.5 Flash, then fallbacks, on 404/429.

    Primary is GEMINI_MODEL (gemini-2.5-flash). There is no Gemini 2.6; if
    2.5-flash is quota-blocked we try GEMINI_FALLBACK_MODEL (comma-separated).
    """
    global _PRIMARY_SKIP_UNTIL, _LAST_GEMINI_MODEL
    import time

    primary = config.GEMINI_MODEL
    fallbacks = _fallback_models(primary)
    if fallbacks and time.time() < _PRIMARY_SKIP_UNTIL:
        last_exc: Exception | None = None
        for model in fallbacks:
            try:
                response = _post_gemini(model, payload)
                _LAST_GEMINI_MODEL = model
                return response
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code not in {404, 429}:
                    raise
        if last_exc is not None:
            raise last_exc
    try:
        response = _post_gemini(primary, payload, retry_429=not bool(fallbacks))
        _LAST_GEMINI_MODEL = primary
        return response
    except urllib.error.HTTPError as exc:
        if exc.code not in {404, 429} or not fallbacks:
            raise
        _PRIMARY_SKIP_UNTIL = time.time() + _PRIMARY_SKIP_S
        last_exc = exc
        for model in fallbacks:
            logger.warning(
                "Gemini model %s failed (HTTP %s); trying %s",
                primary if model == fallbacks[0] else _LAST_GEMINI_MODEL or primary,
                exc.code if model == fallbacks[0] else getattr(last_exc, "code", "?"),
                model,
            )
            try:
                response = _post_gemini(model, payload)
                _LAST_GEMINI_MODEL = model
                return response
            except urllib.error.HTTPError as inner:
                last_exc = inner
                if inner.code not in {404, 429}:
                    raise
        raise last_exc


def _call_gemini(
    image_b64: str,
    prompt: str,
    reference_images: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
    ]
    parts.extend(_reference_image_parts_gemini(reference_images))
    payload = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    return _gemini_generate(payload)

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
    # Text shape descriptions from the uploaded sheet — do NOT attach glyph
    # JPEGs here. Count-verify already sends the (often huge) full-wing image;
    # extra reference images blow the Gemini input-token quota and the whole
    # call is skipped (glyphs never reach the model).
    reference_images: list[tuple[str, str]] = []
    glyph_descs = cached_glyph_descriptions(legend, linear_keys or set())
    prompt = _build_prompt(
        legend,
        cv_counts,
        linear_keys=linear_keys or set(),
        extra_rules=extra_rules,
        detection_hints=detection_hints,
        glyph_descs=glyph_descs,
        reference_glyphs_attached=bool(reference_images),
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
                response = _call_gemini(
                    image_b64, prompt, reference_images=reference_images
                )
                content = _gemini_text(response)
            else:
                user_content: list[dict[str, Any]] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ]
                for label, ref_b64 in reference_images:
                    user_content.append(
                        {
                            "type": "text",
                            "text": (
                                f"REFERENCE GLYPH — legend key {label!r} "
                                "(legend artwork only, NOT part of the plan):"
                            ),
                        }
                    )
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"},
                        }
                    )
                payload = {
                    "model": config.GROQ_MODEL,
                    "temperature": 0.1,
                    "max_completion_tokens": 4096,
                    "reasoning_effort": "none",
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
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
    _stamp_model(result)
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
        "This is a BINARY confirmation pass: answer only whether the crop matches the",
        "single legend target above. 'Is this a match' — nothing else.",
        "",
        "Rules:",
        "- present=true only if real plan ink for a symbol/tag/label is visible",
        "  (not blank wall, hatch, dimension leader, room label alone, title block,",
        "  elevation +48, or a keyed-note square with only '1').",
        "- symbol_matches=true only if it matches the legend target (glyph, letter tag,",
        "  or numeric part label such as 2300/5400/WM2300).",
        "- TEXT IS NOT A GLYPH: when the target is a drawn glyph (shape), printed",
        "  words/numbers alone in the crop are NOT a match — set symbol_matches=false.",
        "  Room labels (A-6E), wing titles, dimensions, and +48 are never matches.",
        "- When a legend glyph reference image is attached, compare the crop's shape",
        "  against it; the shapes must visibly agree.",
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
        "- Is this crop the FULL glyph for the target key, not a fragment shared with",
        "  another legend row? (E≠stub unless long middle bar; one J≠j-hook; triangle",
        "  inside AP/pole/camera ≠ '#' drop).",
        "",
        _LOOKALIKE_CONFUSION_TABLE,
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
            response = _gemini_generate(payload)
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
    _stamp_model(result)
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
    reference_images: list[tuple[str, str]] | None = None,
) -> tuple[str | None, str | None]:
    """Call configured vision provider; return (content, error).

    ``reference_images`` is a list of (label, jpeg_b64) legend glyph crops
    attached AFTER the plan image so the model does visual matching against
    the exact legend artwork instead of generative detection.
    """
    provider = config.llm_provider()
    try:
        if provider == "gemini":
            parts: list[dict[str, Any]] = [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ]
            for label, ref_b64 in reference_images or []:
                parts.append(
                    {
                        "text": (
                            f"REFERENCE GLYPH — legend key {label!r} "
                            "(legend artwork only, NOT part of the plan):"
                        )
                    }
                )
                parts.append(
                    {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}}
                )
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                },
            }
            response = _gemini_generate(payload)
            return _gemini_text(response), None
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
        for label, ref_b64 in reference_images or []:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"REFERENCE GLYPH — legend key {label!r} "
                        "(legend artwork only, NOT part of the plan):"
                    ),
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"},
                }
            )
        payload = {
            "model": config.GROQ_MODEL,
            "temperature": 0.1,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        response = _call_groq(payload)
        return str(response["choices"][0]["message"]["content"]), None
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        return None, f"HTTP {exc.code}: {detail or exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# Discrete device glyphs first. Linear raceway/conduit line art is skipped
# except look-alike glyphs that Gemini must see (stub, J-hook, #, AP).
_PRIORITY_REF_KEYS = (
    "#",
    "AP",
    "CONDUIT STUB",
    "J-HOOK (SINGLE/STACKED)",
    "NETWORK CAMERA",
    "J",
    "G",
    "R",
    "WP",
    "NVR",
    "FACP",
    "TGB",
    "IACP",
    "STC",
)
_FORCE_GLYPH_REF_RE = re.compile(r"CONDUIT\s+STUB|J-?HOOK", re.I)


def _ref_sort_key(key: str) -> tuple[int, str]:
    upper = key.upper()
    if key == "#" or upper.startswith("#"):
        return (0, key)
    if upper == "AP":
        return (1, key)
    if "STUB" in upper:
        return (2, key)
    if "J-HOOK" in upper or "J HOOK" in upper:
        return (3, key)
    try:
        return (10 + _PRIORITY_REF_KEYS.index(key), key)
    except ValueError:
        if "CAMERA" in upper:
            return (12, key)
        return (100, key)


def _legend_reference_images(
    legend: SymbolTableInfo,
    linear_keys: set[str],
    *,
    max_refs: int | None = None,
) -> list[tuple[str, str]]:
    """(display_key, jpeg_b64) for legend rows carrying real glyph artwork.

    These come from the uploaded technical-symbol PDF (or the symbol library)
    and are attached to the detect call so the model matches shapes visually
    instead of guessing from text descriptions.

    Linear glyphs (raceway/conduit linework) are never attached: they do not
    match plan marks and they inflate Gemini input tokens until the call 429s.
    Discrete device glyphs (#, AP, cameras, letter tags) are attached first.
    """
    from pipeline.cv.template_match import LINEAR_GLYPH_RE

    limit = int(max_refs if max_refs is not None else config.VISION_DETECT_MAX_REFS)
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in legend.entries:
        key = count_key(entry)
        if not key or key in seen:
            continue
        force_ref = bool(_FORCE_GLYPH_REF_RE.search(key))
        if not force_ref:
            if key in linear_keys:
                continue
            if LINEAR_GLYPH_RE.search(key) and (entry.symbol or "").strip() != "#":
                continue
        png = entry.symbol_image_png
        if not png:
            continue
        try:
            img = Image.open(BytesIO(png)).convert("RGB")
        except OSError:
            continue
        # Upscale tiny glyph crops so the shape is legible to the model.
        if max(img.size) < 96:
            scale = 96 / max(img.size)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
        candidates.append((key, _encode_image(img)))
        seen.add(key)
    candidates.sort(key=lambda item: _ref_sort_key(item[0]))
    return candidates[: max(0, limit)]


# One cached shape-description set per uploaded technical-symbol sheet.
_GLYPH_DESC_CACHE: dict[str, dict[str, str]] = {}


def _glyph_refs_cache_key(refs: list[tuple[str, str]]) -> str:
    h = hashlib.sha1()
    for label, b64 in refs:
        h.update(label.encode("utf-8"))
        h.update(b64[:64].encode("ascii"))
        h.update(str(len(b64)).encode("ascii"))
    return h.hexdigest()


def cached_glyph_descriptions(
    legend: SymbolTableInfo,
    linear_keys: set[str] | None = None,
) -> dict[str, str]:
    """Shape descriptions for this legend without a new LLM call.

    Prefers the in-process cache, then the on-disk reference legend that
    matches this sheet's entries (artwork hash / full row) — not key-only.
    """
    refs = _legend_reference_images(legend, linear_keys or set())
    cached = _GLYPH_DESC_CACHE.get(_glyph_refs_cache_key(refs)) if refs else None
    if cached:
        return dict(cached)
    try:
        from pipeline.legend_reference import lookup_descriptions_for_legend

        return lookup_descriptions_for_legend(legend)
    except Exception:  # noqa: BLE001
        return {}


def describe_legend_glyphs_with_llm(
    legend: SymbolTableInfo,
    linear_keys: set[str] | None = None,
    *,
    reference_images: list[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Ask the vision model to describe each uploaded legend glyph's SHAPE once.

    Runs a single call per technical-symbol sheet (cached for the server's
    lifetime) and returns {display_key: short shape description} — e.g.
    '#' → 'small solid filled triangle'. The descriptions are injected into
    VALID_SYMBOLS so the model knows exactly what each glyph looks like.
    """
    linear_keys = linear_keys or set()
    refs = (
        reference_images
        if reference_images is not None
        else _legend_reference_images(legend, linear_keys)
    )
    if not refs:
        return {}
    cache_key = _glyph_refs_cache_key(refs)
    if cache_key in _GLYPH_DESC_CACHE:
        return dict(_GLYPH_DESC_CACHE[cache_key])

    # Curated reference legends (storage/reference_legends/) win over the LLM,
    # but ONLY for entries that provably match the stored sheet (same glyph
    # artwork hash or identical legend row) — other projects' sheets may reuse
    # the same key for different symbols.
    try:
        from pipeline.legend_reference import lookup_descriptions_for_legend

        stored = lookup_descriptions_for_legend(legend)
    except Exception:  # noqa: BLE001
        stored = {}
    descs: dict[str, str] = {
        label: stored[label] for label, _b64 in refs if label in stored
    }
    labels = [label for label, _b64 in refs if label not in descs]
    if not labels or not config.llm_enabled():
        _GLYPH_DESC_CACHE[cache_key] = dict(descs)
        return descs
    prompt = "\n".join(
        [
            "The attached images are cropped glyph artworks from a construction",
            "technical-symbol legend sheet, each labeled with its legend key.",
            "For EVERY labeled reference glyph, describe ONLY the drawn shape in",
            "4–12 words, the way a drafter would (e.g. 'small solid filled triangle',",
            "'open rectangle on wall line', 'hexagon outline with letter T',",
            "'two opposing triangles with X box between').",
            "Do not describe colors, image quality, or meaning — shape only.",
            "",
            "Return JSON only:",
            '{"glyphs": {"<legend key>": "<shape description>", ...}}',
            "",
            "Legend keys to describe (copy EXACTLY as JSON keys):",
            *[f"- {label!r}" for label in labels],
        ]
    )
    missing_refs = [(label, b64) for label, b64 in refs if label in set(labels)]
    content, err = _call_vision_json(
        prompt=prompt,
        image_b64=missing_refs[0][1],
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=2048,
        reference_images=missing_refs,
    )
    parsed = _extract_json(str(content or ""))
    if parsed and isinstance(parsed.get("glyphs"), dict):
        label_set = set(labels)
        for k, v in parsed["glyphs"].items():
            k = str(k).strip()
            if k in label_set and isinstance(v, str) and v.strip():
                descs[k] = v.strip()[:120]
    if err:
        logger.warning("Legend glyph description call failed: %s", err)
    # Cache even on failure so folder jobs never repeat a failing call per tile.
    _GLYPH_DESC_CACHE[cache_key] = dict(descs)
    return descs


_LOOKALIKE_CONFUSION_TABLE = """
LOOK-ALIKE GLYPHS (mandatory — do not confuse fragments):
- CONDUIT STUB = capital E whose MIDDLE bar is much longer than the top/bottom,
  sitting on conduit. A plain letter E, room code A-6E, or the word EAST is NOT a stub.
- J-HOOK = two or more letter J's on ONE horizontal line (—J—J—J—). That cluster is
  ONE j-hook run, never JUNCTION BOX. JUNCTION BOX is a single boxed/standalone J.
- '#' DATA DROP = a STANDALONE filled triangle + optional digit. A filled triangle
  INSIDE a circle-crosshair is AP (not #). Inside a square bowtie/X it is DATA POLE
  (not #). Two opposing triangles + X-box is ONE NETWORK CAMERA (not two # drops).
  A filled triangle sitting on a hollow camera body rectangle is the camera glyph.
""".strip()

# Failure modes seen on real bid-set tiles — the model must never report these.
_DETECT_NEGATIVE_LIST = [
    "room labels / space IDs (A-1, A-2, A-6E, RR/B, C-5B, 1.03-in-circle)",
    "wing titles and sheet text (A-WING (EAST), TECHNOLOGY FLOOR PLANS …)",
    "KEY PLAN inset and everything inside it",
    "dimension lines, dimension text, leader lines, grid bubbles, scale bars",
    "elevation heights +48 / +49 (mounting height AFF, never equipment)",
    "hexagon / square keynote bubbles containing only a digit",
    "door swings, doors, windows, furniture outlines, plumbing fixtures",
    "wall hatch, break-line zigzags, north arrows",
    "general note text, title block, legend table printed on the sheet",
    "pipe callouts (1|5, 1|13) as standalone symbols — they only locate '#' drops",
    "plain letter E / EAST / A-6E as CONDUIT STUB",
    "a single J that is part of a —J—J—J— j-hook run as JUNCTION BOX",
    "filled triangle inside AP circle, DATA POLE bowtie, or camera hourglass as '#'",
]


def detect_legend_symbols_with_llm(
    image: Image.Image,
    legend: SymbolTableInfo,
    *,
    linear_keys: set[str] | None = None,
    max_instances: int = 80,
) -> dict[str, Any]:
    """Visual-match legend symbols on a plan crop (not generative detection).

    The prompt enumerates the literal legend (VALID_SYMBOLS) and the call
    attaches the actual legend glyph crops as reference images. Detections
    that do not name a legend key are dropped; oversized boxes are clamped
    so they cannot suppress neighbouring symbols during NMS.

    Returns {"ok": bool, "detections": [{"symbol", "x1", "y1", ...}],
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

    reference_images = _legend_reference_images(legend, linear_keys)
    ref_keys = {label for label, _b64 in reference_images}
    # One cached call per uploaded technical-symbol sheet: shape description
    # for every glyph, injected below so the model knows what to match.
    glyph_descs = describe_legend_glyphs_with_llm(
        legend, linear_keys, reference_images=reference_images
    )
    try:
        from pipeline.legend_reference import lookup_descriptions_for_legend

        stored = lookup_descriptions_for_legend(legend)
        glyph_descs = {**stored, **glyph_descs}
    except Exception:  # noqa: BLE001
        pass

    valid_symbols: list[dict[str, Any]] = []
    valid: set[str] = set()
    for entry in legend.entries:
        key = count_key(entry)
        if not key or key in valid:
            continue
        valid.add(key)
        valid_symbols.append(
            {
                "name": key,
                "glyph": glyph_descs.get(key),
                "tag": (entry.symbol or "").strip() or None,
                "part": (entry.part_number or "").strip() or None,
                "description": (entry.description or "").strip() or None,
                "reference_image_attached": key in ref_keys,
                "linear_never_box": key in linear_keys,
            }
        )
    if not valid_symbols:
        result["error"] = "Empty legend"
        return result

    prompt = "\n".join(
        [
            "TASK: visual matching of EXACT legend symbols on a CAD zoom tile.",
            "This is NOT open-ended detection. You may ONLY report marks that match",
            "one of the VALID_SYMBOLS below, by shape (glyphs) or exact letters/digits",
            "(tags / part labels). If a mark does not clearly match one of these exact",
            "entries, DO NOT report it — do not invent new categories.",
            "",
            "VALID_SYMBOLS (the ONLY reportable keys — copy 'name' EXACTLY):",
            json.dumps(valid_symbols, ensure_ascii=False, indent=1),
            "",
            "After the plan image, each entry with reference_image_attached=true has its",
            "actual legend artwork attached, labeled with its key. Those reference images",
            "are the ground truth for what each glyph looks like — match against them.",
            "They are NOT part of the plan; never count the reference images themselves.",
            "",
            "Return JSON only:",
            '{"detections":[{"symbol":"<name>","kind":"tag|drop|device|raceway|camera|other",'
            '"qty":1,"confidence":0.0,"bbox_rel":{"x1":0,"y1":0,"x2":0,"y2":0}}],'
            '"notes":"<short>"}',
            "",
            "MATCHING RULES (strict):",
            "- GLYPH keys: the drawn SHAPE on the plan must match the attached reference",
            "  glyph. Printed words or numbers alone are NEVER a glyph match — text is",
            "  not a symbol.",
            "- TAG keys (J, AP, WP, …): the exact letters must be inked on the plan,",
            "  not part of a longer word or a room code.",
            "- PART keys (2300/5400/5500 → WMxxxx): digits next to raceway linework only.",
            "- '#' DATA DROP: on the PLAN this is ONLY a STANDALONE solid FILLED triangle",
            "  (any direction) plus an optional digit 1–9 beside it (that digit is qty).",
            "  The legend-table crop may also show a '#' character under the triangle —",
            "  that character is the table's KEY COLUMN, not drawn next to plan drops.",
            "  Never require a hash character on the plan. Not a door-swing or north arrow.",
            "- AP: on the PLAN match the circled crosshair (letters AP may sit under it",
            "  or be missing). The triangle INSIDE that circle is part of AP, not a '#' drop.",
            "- DATA POLE: square with opposing filled triangles (bowtie / X). Those inner",
            "  triangles are NOT '#' drops.",
            "- NETWORK CAMERA: hourglass (two opposing triangles + X-box) = ONE camera,",
            "  not two '#' drops. A filled triangle on a hollow camera-body rectangle is",
            "  the camera glyph, not a drop.",
            "- JUNCTION BOX (tag J): a single boxed or standalone letter J.",
            "- J-HOOK: two or more J's on one horizontal line (—J—J—J—) = one hook run,",
            "  never count those letters as JUNCTION BOX.",
            "- CONDUIT STUB: E-shape with a LONG middle bar on conduit. Plain E / EAST /",
            "  A-6E is not a stub.",
            "- IP CLOCK/SPEAKER: horn/speaker glyph with '12:00' text.",
            "- Skip every linear_never_box=true key entirely (continuous conduit/raceway/",
            "  ladder-rack runs). J-HOOK, CONDUIT STUB, and DATA POLE are discrete — report them.",
            "- When unsure whether ink matches a legend entry, leave it out. Precision",
            "  beats recall; a wrong box corrupts counts downstream.",
            "",
            _LOOKALIKE_CONFUSION_TABLE,
            "",
            "NEGATIVE LIST — NEVER report any of these as detections:",
            *[f"- {item}" for item in _DETECT_NEGATIVE_LIST],
            "",
            "BOX RULES (critical — bad boxes disturb neighbouring symbols):",
            "- bbox_rel is normalized 0–1 of the PLAN image (origin top-left).",
            "- One box = ONE symbol's ink, drawn TIGHT. Never emit a large box covering",
            "  several symbols, a wall run, or a whole room — if symbols cluster, emit",
            "  one small tight box per symbol instance.",
            "- A box wider/taller than ~10% of the image is almost certainly wrong for",
            "  a single glyph; shrink it to the symbol's ink.",
            f"- Cap at {max_instances} detections.",
        ]
    )
    image_b64 = _encode_image(image)
    content, err = _call_vision_json(
        prompt=prompt,
        image_b64=image_b64,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=8192,
        reference_images=reference_images,
    )
    if err:
        result["error"] = err
        return result
    parsed = _extract_json(str(content or ""))
    if not parsed or not isinstance(parsed.get("detections"), list):
        result["error"] = "Model reply was not valid detections JSON."
        return result

    img_w, img_h = image.size
    # A single legend glyph on a 653–768px zoom tile is ~12–60px. Clamp anything
    # bigger so one loose vision box cannot swallow neighbouring symbols in NMS.
    max_side = max(28.0, float(config.VISION_DETECT_MAX_BOX_FRAC) * min(img_w, img_h))

    def _clamp_span(a: float, b: float, upper: float) -> tuple[float, float, bool]:
        if b - a <= max_side:
            return a, b, False
        center = (a + b) / 2.0
        half = max_side / 2.0
        return max(0.0, center - half), min(upper, center + half), True

    out: list[dict[str, Any]] = []
    clamped_n = 0
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
        xa, xb, cx_clamped = _clamp_span(xa, xb, float(img_w))
        ya, yb, cy_clamped = _clamp_span(ya, yb, float(img_h))
        if cx_clamped or cy_clamped:
            clamped_n += 1
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
    _stamp_model(result)
    result["detections"] = out
    result["reference_glyphs_attached"] = len(reference_images)
    if glyph_descs:
        result["glyph_descriptions"] = glyph_descs
    if clamped_n:
        result["clamped_boxes"] = clamped_n
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
    _stamp_model(result)
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
    "cached_glyph_descriptions",
    "describe_legend_glyphs_with_llm",
    "detect_legend_symbols_with_llm",
    "judge_detection_crop",
    "parse_crop_text_with_llm",
    "verify_counts_with_llm",
]
