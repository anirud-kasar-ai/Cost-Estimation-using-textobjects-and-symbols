"""Confirm a detected symbol against its own glyph file.

Localization stays a CV box. After CV/taxonomy names a mark, this step looks
that key up in the glyph folder and scores the crop against THAT file only.
It does not scan the rest of the legend.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.cv.tag_match import count_key
from pipeline.cv.template_match import (
    LINEAR_GLYPH_RE,
    _build_glyph_label_map,
    _resolve_glyph_label,
    score_gray_on_gray,
)
from pipeline.detect.candidates import Candidate
from pipeline.extraction.symbol_table_extractor import SymbolTableInfo
from pipeline.taxonomy.schema import SymbolTaxonomy


def load_ident_glyphs(
    legend: SymbolTableInfo,
    glyph_dir: Path | None,
) -> list[tuple[str, Image.Image]]:
    """Legend-key → artwork. Glyph folder is the source of truth when present."""
    by_key: dict[str, Image.Image] = {}
    if glyph_dir is not None and glyph_dir.is_dir():
        entry_by_key = _build_glyph_label_map(legend)
        for path in sorted(glyph_dir.glob("*.png")):
            label = _resolve_glyph_label(path.stem, entry_by_key)
            if not label or label in by_key or LINEAR_GLYPH_RE.search(label):
                continue
            try:
                by_key[label] = Image.open(path).convert("RGB")
            except OSError:
                continue
        return list(by_key.items())
    for entry in legend.entries:
        key = count_key(entry)
        if not key or LINEAR_GLYPH_RE.search(key):
            continue
        png = entry.symbol_image_png
        if not png:
            continue
        try:
            by_key[key] = Image.open(BytesIO(png)).convert("RGB")
        except OSError:
            continue
    return list(by_key.items())


def _prepare_glyph_grays(
    glyphs: list[tuple[str, Image.Image]],
) -> dict[str, np.ndarray]:
    return {key: np.array(img.convert("L")) for key, img in glyphs}


def _crop_gray(image: Image.Image, cand: Candidate, *, margin: int) -> np.ndarray:
    import cv2

    w, h = image.size
    x1 = max(0, int(cand.bbox.x1) - margin)
    y1 = max(0, int(cand.bbox.y1) - margin)
    x2 = min(w, int(cand.bbox.x2) + margin)
    y2 = min(h, int(cand.bbox.y2) + margin)
    if x2 - x1 < 4 or y2 - y1 < 4:
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    max_side = 96
    gh, gw = gray.shape[:2]
    if max(gh, gw) > max_side:
        scale = max_side / float(max(gh, gw))
        gray = cv2.resize(
            gray,
            (max(8, int(gw * scale)), max(8, int(gh * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def rank_glyphs_on_gray(
    crop_gray: np.ndarray,
    glyph_grays: list[tuple[str, np.ndarray]],
    *,
    assigned: str | None = None,
    min_score: float = 0.55,
) -> list[tuple[str, float]]:
    """Score crop vs the small set of glyphs passed in (detected keys only)."""
    ranked: list[tuple[str, float]] = []
    if assigned:
        for key, gray in glyph_grays:
            if key != assigned:
                continue
            score = float(score_gray_on_gray(crop_gray, gray))
            if score >= 0.85:
                return [(key, score)]
            if score >= 0.02:
                ranked.append((key, score))
            break
    for key, gray in glyph_grays:
        if assigned and key == assigned:
            continue
        score = float(score_gray_on_gray(crop_gray, gray))
        if score >= 0.02:
            ranked.append((key, score))
            if score >= 0.92 and score >= min_score + 0.2:
                break
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _is_qty_digit_key(key: str) -> bool:
    text = (key or "").strip()
    return text.isdigit() and 1 <= len(text) <= 2


def _detected_keys(cand: Candidate, valid: set[str]) -> list[str]:
    """Keys CV already proposed for this box — never the full legend."""
    keys: list[str] = []
    if cand.status == "resolved" and cand.resolved_key:
        keys.append(cand.resolved_key)
        return keys
    ocr = (cand.ocr_text or "").strip()
    if ocr in valid:
        keys.append(ocr)
    for key in cand.candidate_keys:
        if key in valid and key not in keys:
            keys.append(key)
    return keys


def identify_candidates_by_glyphs(
    image: Image.Image,
    candidates: list[Candidate],
    *,
    legend: SymbolTableInfo,
    glyph_dir: Path | None,
    taxonomy: SymbolTaxonomy,
    valid_keys: set[str],
) -> tuple[list[Candidate], dict[str, int]]:
    """Match each detected mark to its glyph file, if that file exists.

    Numbers are glyphs too when the folder has that key (``4``, ``2300``).
    A QTY digit next to a drop triangle is not promoted to a numbered key.
    """
    stats = {
        "compared": 0,
        "skipped": 0,
        "no_glyph": 0,
        "confirmed": 0,
        "relabeled": 0,
        "assigned": 0,
    }
    glyphs = load_ident_glyphs(legend, glyph_dir)
    if not glyphs:
        return candidates, stats
    glyph_grays = _prepare_glyph_grays(glyphs)

    valid = set(valid_keys) | set(taxonomy.keys())
    min_score = float(config.GLYPH_IDENT_MIN)
    margin = float(config.GLYPH_IDENT_MARGIN)
    crop_margin = max(4, int(config.VISION_CLASSIFY_MARGIN_PX))

    from pipeline.classify.context_resolver import _mark_resolved

    for cand in candidates:
        wanted = [
            key
            for key in _detected_keys(cand, valid)
            if key in glyph_grays and not LINEAR_GLYPH_RE.search(key)
        ]
        if not wanted:
            stats["no_glyph"] += 1
            continue
        src = (cand.source or "").strip().lower()
        if src in {"triangle", "hourglass", "bowtie", "stub", "jhook", "callout_drop"}:
            stats["skipped"] += 1
            continue
        crop_gray = _crop_gray(image, cand, margin=crop_margin)
        assigned = cand.resolved_key if cand.status == "resolved" else None
        ranking = rank_glyphs_on_gray(
            crop_gray,
            [(key, glyph_grays[key]) for key in wanted],
            assigned=assigned,
            min_score=min_score,
        )
        stats["compared"] += 1
        if not ranking:
            continue
        best_key, best_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        if best_key not in valid or best_score < min_score:
            continue
        if LINEAR_GLYPH_RE.search(best_key):
            continue
        if cand.shape_class_guess == "triangle_like" and _is_qty_digit_key(best_key):
            continue
        ocr = (cand.ocr_text or "").strip()
        if (
            cand.source == "ocr"
            and ocr.isdigit()
            and best_key.strip() == "#"
            and ocr not in {"#"}
        ):
            continue

        assigned_score = 0.0
        if assigned:
            for key, score in ranking:
                if key == assigned:
                    assigned_score = score
                    break

        if assigned == best_key:
            stats["confirmed"] += 1
            if cand.classify_source in {"unresolved", "unmatched", "ambiguous", ""}:
                cand.classify_source = "glyph_identify"
            continue

        clear_winner = best_score >= assigned_score + margin and (
            assigned is None or best_score - second_score >= margin * 0.5
        )
        if not clear_winner and assigned:
            continue

        _mark_resolved(cand, best_key, source="glyph_identify")
        if assigned:
            stats["relabeled"] += 1
        else:
            stats["assigned"] += 1

    return candidates, stats


def rank_glyphs_on_crop(
    crop: Image.Image,
    glyphs: list[tuple[str, Image.Image]],
) -> list[tuple[str, float]]:
    """Test helper: rank PIL crop against PIL glyphs."""
    import cv2

    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    items = list(_prepare_glyph_grays(glyphs).items())
    return rank_glyphs_on_gray(gray, items)


__all__ = [
    "identify_candidates_by_glyphs",
    "load_ident_glyphs",
    "rank_glyphs_on_crop",
]
