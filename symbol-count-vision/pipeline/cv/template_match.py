"""Template matching of legend glyph PNGs against a plan image."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.cv.tag_match import count_key
from pipeline.extraction.symbol_table_extractor import SymbolTableInfo, catalog_id_for_entry

logger = logging.getLogger(__name__)

# Linear items are drawn as continuous runs (walls look identical to their
# glyphs), so template matching produces hundreds of false boxes along any
# straight line. They cannot be counted as discrete instances this way —
# part-number text labels (WM2300 → "2300") are counted via OCR instead.
# Drop-mark glyph is counted geometrically (solid triangles), not by template.
DROP_MARK_GLYPH = "#"
SHORT_TAG_TEMPLATE_RE = re.compile(r"^[A-Z]{2,3}$")
# True wall/raceway RUNS — counting these as discrete glyphs false-fires
# on every straight wall. CONDUIT STUB, DATA POLE, and J-HOOK clusters are
# discrete look-alikes handled separately (not this regex).
LINEAR_GLYPH_RE = re.compile(
    r"RACEWAY|LADDER\s+RACK|PERMANENT\s+LINK|(?<!STUB\s)CONDUIT(?!\s+STUB)",
    re.IGNORECASE,
)
# Three J's on a line would template-match every isolated J — skip that glyph
# file, but still count J-HOOK via OCR clustering (not LINEAR_GLYPH_RE).
_JHOOK_TEMPLATE_SKIP_RE = re.compile(r"J-?HOOK", re.IGNORECASE)

# Camera-family glyphs need finer scales on zoom tiles, but a *higher*
# threshold than the default — at 0.60 they match keynote boxes and wall
# junctions that are not cameras.
CAMERA_GLYPH_RE = re.compile(r"CAMERA|\bMOUNT\b", re.IGNORECASE)
CAMERA_TEMPLATE_THRESHOLD = 0.78
CAMERA_TEMPLATE_SCALES = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]


def _perceptual_hash(gray: np.ndarray) -> int:
    """64-bit average hash of a grayscale image for near-duplicate detection."""
    import cv2

    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = 0
    for value in small.flatten().tolist():
        bits = (bits << 1) | (1 if value > mean else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _build_glyph_label_map(legend: SymbolTableInfo) -> dict[str, str]:
    """Map glyph filename stems / catalog ids → legend count_key only."""
    entry_by_key: dict[str, str] = {}
    for entry in legend.entries:
        label = count_key(entry)
        if not label:
            continue
        key = catalog_id_for_entry(entry)
        if key:
            entry_by_key[key] = label
        if entry.symbol:
            entry_by_key[entry.symbol] = label
        if entry.description:
            entry_by_key[entry.description] = label
        entry_by_key[label] = label
        safe = re.sub(r'[<>:"/\\|?*]', "_", label)[:80]
        entry_by_key[safe] = label
    return entry_by_key


def _resolve_glyph_label(stem: str, entry_by_key: dict[str, str]) -> str | None:
    """Resolve a glyph filename to a legend count_key, or None if unknown."""
    label = entry_by_key.get(stem)
    if label:
        return label
    if stem.rsplit("_", 1)[-1].isdigit():
        base = stem.rsplit("_", 1)[0]
        return entry_by_key.get(base)
    return None


def _match_one_template(
    plan_gray: np.ndarray,
    template_gray: np.ndarray,
    *,
    threshold: float,
    scales: list[float],
) -> list[tuple[float, float, float, float, float]]:
    import cv2

    th, tw = template_gray.shape[:2]
    if th < config.CV_TEMPLATE_MIN_SIZE or tw < config.CV_TEMPLATE_MIN_SIZE:
        return []
    if tw > plan_gray.shape[1] * 0.5 or th > plan_gray.shape[0] * 0.5:
        return []

    hits: list[tuple[float, float, float, float, float]] = []
    for scale in scales:
        sw = max(8, int(tw * scale))
        sh = max(8, int(th * scale))
        if sw >= plan_gray.shape[1] or sh >= plan_gray.shape[0]:
            continue
        resized = cv2.resize(template_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(plan_gray, resized, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            score = float(result[y, x])
            hits.append((float(x), float(y), float(x + sw), float(y + sh), score))
    return hits


def _nms_hits(
    hits: list[tuple[float, float, float, float, float]],
    iou_threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    for hit in ordered:
        x1, y1, x2, y2, score = hit
        suppress = False
        for kx1, ky1, kx2, ky2, _ in kept:
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_b = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
            union = area_a + area_b - inter
            if union > 0 and inter / union >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(hit)
    return kept


def detect_symbols_from_glyphs(
    image: Image.Image,
    legend: SymbolTableInfo,
    glyph_dir: Path,
) -> list[SymbolDetection]:
    if not glyph_dir.is_dir():
        return []

    import cv2

    plan_gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    default_threshold = float(config.CV_TEMPLATE_THRESHOLD)
    default_scales = list(config.CV_TEMPLATE_SCALES)
    nms_iou = float(config.SYMBOL_COUNT_NMS_IOU)

    entry_by_key = _build_glyph_label_map(legend)

    detections: list[SymbolDetection] = []
    glyph_files = sorted(glyph_dir.glob("*.png"))
    if not glyph_files:
        logger.warning("No glyph PNGs found in %s", glyph_dir)
        return []

    seen_hashes: set[int] = set()
    for glyph_path in glyph_files:
        template = cv2.imread(str(glyph_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        stem = glyph_path.stem
        label = _resolve_glyph_label(stem, entry_by_key)
        if not label:
            # Glyph is not in the uploaded legend — never invent a count key.
            continue

        if LINEAR_GLYPH_RE.search(label) or LINEAR_GLYPH_RE.search(stem):
            continue
        if _JHOOK_TEMPLATE_SKIP_RE.search(label) or _JHOOK_TEMPLATE_SKIP_RE.search(stem):
            continue
        if label.strip() == DROP_MARK_GLYPH:
            continue
        # Cameras are counted geometrically (hourglass / X-box), not by
        # template — these glyphs false-match keynotes and junctions.
        if CAMERA_GLYPH_RE.search(label) or CAMERA_GLYPH_RE.search(stem):
            continue

        digest = _perceptual_hash(template)
        if any(_hamming(digest, seen) <= 6 for seen in seen_hashes):
            continue
        seen_hashes.add(digest)

        th, tw = template.shape[:2]
        if th < 16 or tw / max(1, th) > 3.0:
            continue

        is_camera = bool(CAMERA_GLYPH_RE.search(label) or CAMERA_GLYPH_RE.search(stem))
        threshold = CAMERA_TEMPLATE_THRESHOLD if is_camera else default_threshold
        if not is_camera and SHORT_TAG_TEMPLATE_RE.match(label.strip()):
            threshold = min(threshold, float(config.CV_SHORT_TAG_TEMPLATE_THRESHOLD))
        scales = CAMERA_TEMPLATE_SCALES if is_camera else default_scales

        hits = _match_one_template(
            plan_gray,
            template,
            threshold=threshold,
            scales=scales,
        )
        hits = _nms_hits(hits, nms_iou)
        if len(hits) > 500:
            logger.warning("Too many template hits for %s — skipping", glyph_path.name)
            continue
        for x1, y1, x2, y2, score in hits:
            detections.append(
                SymbolDetection(
                    symbol=label,
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=score,
                    source="template",
                )
            )

    return detections


def score_glyph_on_crop(crop: Image.Image, glyph: Image.Image) -> float:
    """Best template-match score of ``glyph`` inside ``crop`` (0..1)."""
    import cv2

    plan_gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    glyph_gray = np.array(glyph.convert("L"))
    if glyph_gray.size == 0 or plan_gray.size == 0:
        return 0.0
    hits = _match_one_template(
        plan_gray,
        glyph_gray,
        threshold=0.0,
        scales=[0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
    )
    if not hits:
        return 0.0
    return max(float(h[4]) for h in hits)
