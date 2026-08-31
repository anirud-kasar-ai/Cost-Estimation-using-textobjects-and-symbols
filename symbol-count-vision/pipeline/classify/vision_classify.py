"""Label-only vision classification for ambiguous CV candidates.

Never returns coordinates. The bbox always stays the CV contour from Step 2.
On error/timeout, keep the CV shape_class_guess (fail toward CV).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from PIL import Image

from pipeline import config
from pipeline.detect.candidates import Candidate
from pipeline.taxonomy.schema import SymbolTaxonomy, taxonomy_rules_prompt_block

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = (
    "Which of these reference symbols does the cropped image match, if any? "
    "Respond with the exact key or 'none'. Do not describe a location or "
    "bounding box — you are only choosing from the list provided."
)


def _crop_with_margin(image: Image.Image, cand: Candidate, *, margin: int) -> Image.Image:
    w, h = image.size
    x1 = max(0, int(cand.bbox.x1) - margin)
    y1 = max(0, int(cand.bbox.y1) - margin)
    x2 = min(w, int(cand.bbox.x2) + margin)
    y2 = min(h, int(cand.bbox.y2) + margin)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return image
    return image.crop((x1, y1, x2, y2))


def _parse_key_choice(raw: str | None, allowed: set[str]) -> str | None:
    if not raw:
        return None
    parsed = None
    try:
        from pipeline.llm_verify import _extract_json

        parsed = _extract_json(raw)
    except Exception:  # noqa: BLE001
        parsed = None
    if isinstance(parsed, dict):
        # Refuse any coordinate-bearing payload.
        for banned in ("bbox", "bbox_rel", "box", "x1", "y1", "x2", "y2"):
            if banned in parsed:
                logger.warning("vision_classify ignored coordinate field %s", banned)
        key = parsed.get("key") or parsed.get("symbol") or parsed.get("choice")
        if isinstance(key, str):
            key = key.strip()
            if key.lower() == "none":
                return "none"
            if key in allowed:
                return key
    text = (raw or "").strip().strip('"').strip("'")
    if text.lower() == "none":
        return "none"
    if text in allowed:
        return text
    return None


def classify_one(
    image: Image.Image,
    cand: Candidate,
    taxonomy: SymbolTaxonomy,
    *,
    margin_px: int | None = None,
    legend: Any | None = None,
) -> Candidate:
    """Classify one ambiguous candidate. Mutates and returns it."""
    margin = int(margin_px if margin_px is not None else config.VISION_CLASSIFY_MARGIN_PX)
    allowed = [k for k in cand.candidate_keys if taxonomy.by_key(k)]
    if not allowed:
        # Fail toward CV: leave unresolved/ambiguous.
        cand.classify_source = "cv_fallback"
        return cand
    if not config.llm_enabled():
        cand.classify_source = "cv_fallback"
        return cand

    from pipeline.llm_verify import (
        _SYSTEM_PROMPT,
        _call_vision_json,
        _encode_image,
        _legend_reference_images,
        gemini_quota_blocked,
    )

    if gemini_quota_blocked():
        cand.classify_source = "cv_fallback"
        return cand

    crop = _crop_with_margin(image, cand, margin=margin)
    linear_keys = {
        e.key for e in taxonomy.entries if e.count_semantics == "linear_suppressed"
    }
    refs = _legend_reference_images(
        legend,
        linear_keys,
        taxonomy=taxonomy,
        prefer_keys=allowed,
        max_refs=max(len(allowed), 1),
    )

    prompt = "\n".join(
        [
            CLASSIFY_PROMPT,
            "",
            "Return JSON only: {\"key\": \"<exact key or none>\"}",
            "Do not include bbox, coordinates, or a location description.",
            "",
            "CANDIDATE KEYS (copy EXACTLY if matching):",
            json.dumps(allowed, ensure_ascii=False),
            "",
            "Legend glyph artwork is attached after the crop, each labeled with",
            "its legend key. Match the crop against those glyphs.",
            "",
            taxonomy_rules_prompt_block(taxonomy),
        ]
    )
    try:
        content, err = _call_vision_json(
            prompt=prompt,
            image_b64=_encode_image(crop),
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=256,
            reference_images=refs or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision_classify error: %s", exc)
        cand.classify_source = "cv_fallback"
        return cand
    if err or not content:
        logger.warning("vision_classify failed: %s", err)
        cand.classify_source = "cv_fallback"
        return cand
    choice = _parse_key_choice(content, set(allowed))
    if choice and choice != "none" and choice in allowed:
        cand.resolved_key = choice
        cand.status = "resolved"
        cand.candidate_keys = [choice]
        cand.classify_source = "vision_classify"
        return cand
    # none / unparseable → keep CV guess (first candidate_keys or unmatched).
    cand.classify_source = "cv_fallback"
    return cand


def classify_ocr_text_fallback(
    image: Image.Image,
    cand: Candidate,
    taxonomy: SymbolTaxonomy,
    *,
    margin_px: int | None = None,
) -> Candidate:
    """Low-confidence Tesseract → vision OCR on the same tight crop (text only)."""
    if not config.vision_ocr_enabled():
        return cand
    margin = int(margin_px if margin_px is not None else config.VISION_CLASSIFY_MARGIN_PX)
    crop = _crop_with_margin(image, cand, margin=margin)
    try:
        from pipeline.llm_verify import parse_crop_text_with_llm

        parsed = parse_crop_text_with_llm(crop, expected_context="Equipment Tags")
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision OCR fallback error: %s", exc)
        return cand
    if not parsed.get("ok"):
        return cand
    texts: list[str] = []
    entities = parsed.get("text_entities")
    if isinstance(entities, list):
        for item in entities:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]).strip())
    grouped = parsed.get("parsed_entities")
    if isinstance(grouped, dict):
        for values in grouped.values():
            if isinstance(values, list):
                texts.extend(str(v).strip() for v in values if v)
    raw = parsed.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        texts.append(raw.strip())
    alias = (cand.ocr_text or "").strip().upper()
    for text in texts:
        upper = text.upper().strip("-–—_/ ")
        if not upper:
            continue
        for entry in taxonomy.entries:
            tag = (entry.tag or "").strip().upper()
            part = (entry.part_number or "").strip().upper()
            if upper == tag or upper == entry.key.upper() or (part and part.endswith(upper)):
                cand.ocr_text = upper
                cand.resolved_key = entry.key
                cand.status = "resolved"
                cand.candidate_keys = [entry.key]
                cand.classify_source = "vision_ocr_fallback"
                return cand
        if alias and upper == alias and cand.candidate_keys:
            cand.classify_source = "vision_ocr_fallback"
            return cand
    return cand


def classify_ambiguous(
    image: Image.Image,
    candidates: list[Candidate],
    taxonomy: SymbolTaxonomy,
    *,
    margin_px: int | None = None,
    legend: Any | None = None,
) -> list[Candidate]:
    """Vision classify look-alike crops only; OCR fallback for low-conf text.

    Single-key CV guesses skip Gemini. At most VISION_CLASSIFY_MAX_PER_TILE
    look-alike calls run per tile (highest CV score first).
    """
    conf_floor = float(config.VISION_OCR_FALLBACK_CONF)
    cap = max(0, int(config.VISION_CLASSIFY_MAX_PER_TILE))
    out: list[Candidate] = list(candidates)
    vision_idxs: list[int] = []
    for i, cand in enumerate(out):
        if (
            cand.shape_class_guess in {"letter_tag", "digit_label"}
            and cand.ocr_conf is not None
            and cand.ocr_conf < conf_floor
            and cand.status != "resolved"
        ):
            cand = classify_ocr_text_fallback(
                image, cand, taxonomy, margin_px=margin_px
            )
            out[i] = cand
        if cand.status != "ambiguous" or not cand.candidate_keys:
            continue
        allowed = [k for k in cand.candidate_keys if taxonomy.by_key(k)]
        if len(allowed) <= 1:
            cand.classify_source = "cv_fallback"
            continue
        vision_idxs.append(i)
    vision_idxs.sort(key=lambda idx: -float(out[idx].score or 0.0))
    from pipeline.llm_verify import gemini_quota_blocked

    quota_skip = gemini_quota_blocked()
    for n, i in enumerate(vision_idxs):
        if n >= cap or quota_skip:
            out[i].classify_source = "cv_fallback"
            continue
        out[i] = classify_one(
            image, out[i], taxonomy, margin_px=margin_px, legend=legend
        )
        if gemini_quota_blocked():
            quota_skip = True
    return out
