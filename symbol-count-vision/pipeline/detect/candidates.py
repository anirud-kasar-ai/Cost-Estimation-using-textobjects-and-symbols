"""Pure CV candidate proposal. No legend-key assignment, no LLM boxes.

Every bbox is an exact contour / OCR word box (no padding). Classification
happens later in context_resolver / vision_classify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.cv.lookalike import detect_conduit_stubs
from pipeline.cv.nms import BoundingBox
from pipeline.cv.tag_match import text_legend_keys
from pipeline.cv.triangle_drops import (
    DENSE_ZOOM_MAX_EDGE,
    TRIANGLE_DEDUPE_RADIUS,
    _detect_bowtie_poles,
    _detect_glyph_circles,
    _detect_xbox_cameras,
    _merge_cameras,
    _merge_candidates,
    _pair_hourglass_cameras,
    _pil_to_gray,
    _triangle_candidates,
    _triangle_inside_box,
    _triangle_inside_circle,
    triangle_is_noise,
)
from pipeline.extraction.symbol_table_extractor import SymbolTableInfo
from pipeline.taxonomy.schema import SymbolTaxonomy


@dataclass
class Candidate:
    tile_id: str
    bbox: BoundingBox
    shape_class_guess: str
    has_enclosing_circle: bool = False
    has_opposing_triangle: bool = False
    has_enclosing_square: bool = False
    ocr_text: str | None = None
    ocr_conf: float | None = None
    score: float = 0.8
    source: str = "cv"
    qty: int = 1
    extras: dict[str, Any] = field(default_factory=dict)
    resolved_key: str | None = None
    status: str = "unresolved"
    candidate_keys: list[str] = field(default_factory=list)
    classify_source: str = "unresolved"

    @property
    def cx(self) -> float:
        return (self.bbox.x1 + self.bbox.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox.y1 + self.bbox.y2) / 2.0


def _binarize(
    image: Image.Image, *, exclude_top_pct: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import cv2

    gray = _pil_to_gray(image)
    h, _w = gray.shape[:2]
    y_min = int(h * max(0.0, exclude_top_pct))
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    if y_min > 0:
        binary[:y_min, :] = 0
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return gray, binary, opened, y_min


def _dedupe_circles(
    circles: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = []
    for cx, cy, radius in circles:
        if any(
            (cx - mc[0]) ** 2 + (cy - mc[1]) ** 2 <= max(radius, mc[2]) ** 2
            for mc in merged
        ):
            continue
        merged.append((cx, cy, radius))
    return merged


def _dedupe_poles(poles: list[dict[str, Any]], y_min: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for pole in poles:
        if pole["cy"] < y_min:
            continue
        if any(
            (pole["cx"] - m["cx"]) ** 2 + (pole["cy"] - m["cy"]) ** 2 <= 20**2
            for m in merged
        ):
            continue
        merged.append(pole)
    return merged


def _detect_open_rects(binary: np.ndarray) -> list[dict[str, Any]]:
    """Hollow rectangles that sit on raceway / wall linework."""
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[dict[str, Any]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 10 or height < 8:
            continue
        if width > 120 or height > 80:
            continue
        aspect = width / float(max(1, height))
        if aspect < 0.8 or aspect > 6.0:
            continue
        patch = binary[y : y + height, x : x + width]
        if patch.size == 0:
            continue
        ink = float(np.mean(patch > 0))
        if ink < 0.08 or ink > 0.55:
            continue
        rects.append(
            {
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + width),
                "y2": float(y + height),
                "cx": x + width / 2.0,
                "cy": y + height / 2.0,
            }
        )
    return rects


def _ocr_text_spans(
    image: Image.Image,
    legend: SymbolTableInfo,
    *,
    exclude_top_pct: float,
) -> list[dict[str, Any]]:
    """Tesseract word boxes for legend aliases. Records the alias, not a key."""
    from pipeline.cv.ocr_tags import (
        _is_title_or_keyplan_ocr_span,
        _neighbor_ocr_blob,
        _unrotate_box,
        ocr_page_passes,
        tesseract_available,
    )
    from pipeline.cv.tag_match import _normalize_numeric_part, match_span_to_text_key

    if not tesseract_available():
        return []
    keys = text_legend_keys(legend)
    if not keys:
        return []

    orig_w, orig_h = image.size
    y_min = orig_h * exclude_top_pct
    min_conf = float(config.CV_OCR_MIN_CONF)
    min_h = float(config.CV_OCR_MIN_HEIGHT)
    max_h = float(config.CV_OCR_MAX_HEIGHT) * 1.5
    scale = 2.0
    numeric_aliases = {a for a, _ in keys if a.isdigit()}
    alias_set = {a for a, _ in keys}

    spans: list[dict[str, Any]] = []
    for rot, data in ocr_page_passes(image):
        n = len(data.get("text") or [])
        for i in range(n):
            text = str(data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            text_upper = text.upper().strip("-–—_/ ")
            normalized = _normalize_numeric_part(text, numeric_aliases)
            if normalized:
                text_upper = normalized
            is_part_alias = text_upper in numeric_aliases or text_upper in alias_set
            is_short_tag = bool(__import__("re").fullmatch(r"[A-Z]{1,3}", text_upper))
            zoomish = max(orig_w, orig_h) <= 800
            conf_floor = 35.0 if is_part_alias else min_conf
            if zoomish and is_short_tag:
                conf_floor = min(conf_floor, 45.0)
            if conf < conf_floor:
                continue
            x = float(data["left"][i]) / scale
            y = float(data["top"][i]) / scale
            w = float(data["width"][i]) / scale
            h = float(data["height"][i]) / scale
            if h <= 0 or w <= 0:
                continue
            glyph_size = min(w, h)
            if glyph_size < min_h or glyph_size > max_h:
                continue
            neighbor = _neighbor_ocr_blob(data, i)
            if _is_title_or_keyplan_ocr_span(
                text_upper=text_upper,
                height=h,
                neighbor_blob=neighbor,
                max_tag_h=float(config.CV_OCR_MAX_HEIGHT),
            ):
                continue
            matched = match_span_to_text_key(text_upper, keys, size=glyph_size)
            if not matched:
                continue
            x1, y1, x2, y2 = _unrotate_box(
                rot, x, y, x + w, y + h, orig_w=float(orig_w), orig_h=float(orig_h)
            )
            xa, xb = sorted((x1, x2))
            ya, yb = sorted((y1, y2))
            if ya < y_min or xb - xa < 2 or yb - ya < 2:
                continue
            spans.append(
                {
                    "alias": text_upper,
                    "legend_hint": matched,
                    "conf": conf,
                    "x1": max(0.0, min(float(orig_w), xa)),
                    "y1": max(0.0, min(float(orig_h), ya)),
                    "x2": max(0.0, min(float(orig_w), xb)),
                    "y2": max(0.0, min(float(orig_h), yb)),
                    "is_digit": text_upper.isdigit() or text_upper in numeric_aliases,
                }
            )
    return spans


def propose_candidates(
    image: Image.Image,
    legend: SymbolTableInfo,
    taxonomy: SymbolTaxonomy | None = None,
    *,
    tile_id: str = "tile",
    exclude_top_pct: float = 0.0,
) -> list[Candidate]:
    """CV-only proposals for one native-resolution tile."""
    del taxonomy  # closed-vocab filtering happens at resolve time
    gray, binary, opened, y_min = _binarize(image, exclude_top_pct=exclude_top_pct)
    h, w = gray.shape[:2]
    dense_zoom = max(h, w) <= DENSE_ZOOM_MAX_EDGE

    triangles = _merge_candidates(
        [
            _triangle_candidates(binary, dense_zoom=dense_zoom),
            _triangle_candidates(opened, dense_zoom=dense_zoom),
        ]
    )
    poles = _dedupe_poles(
        _detect_bowtie_poles(binary) + _detect_bowtie_poles(opened), float(y_min)
    )
    pair_cams, hourglass_used = _pair_hourglass_cameras(triangles, blockers=poles)
    xbox_cams = _merge_cameras(
        [_detect_xbox_cameras(binary), _detect_xbox_cameras(opened)]
    )
    glyph_circles = _dedupe_circles(
        _detect_glyph_circles(binary) + _detect_glyph_circles(opened)
    )
    open_rects = _detect_open_rects(binary) + _detect_open_rects(opened)

    out: list[Candidate] = []
    camera_boxes: list[dict[str, Any]] = []

    for cam in pair_cams:
        # _pair_hourglass_cameras pads by 6px; strip it so the box is the
        # union of the two triangle contours.
        box = BoundingBox(
            x1=float(cam["x1"]) + 6.0,
            y1=float(cam["y1"]) + 6.0,
            x2=float(cam["x2"]) - 6.0,
            y2=float(cam["y2"]) - 6.0,
        )
        camera_boxes.append(
            {
                "x1": box.x1 - 8.0,
                "y1": box.y1 - 4.0,
                "x2": box.x2 + 8.0,
                "y2": box.y2 + 4.0,
            }
        )
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=box,
                shape_class_guess="hourglass_like",
                has_opposing_triangle=True,
                score=0.82,
                source="hourglass",
            )
        )

    for cam in xbox_cams:
        cx, cy = float(cam["cx"]), float(cam["cy"])
        if any(
            abs(cx - float(pole["cx"])) < 28 and abs(cy - float(pole["cy"])) < 28
            for pole in poles
        ):
            continue
        if any(
            abs(cx - (b["x1"] + b["x2"]) / 2.0) < 28
            and abs(cy - (b["y1"] + b["y2"]) / 2.0) < 28
            for b in camera_boxes
        ):
            continue
        box = BoundingBox(
            x1=float(cam["x1"]),
            y1=float(cam["y1"]),
            x2=float(cam["x2"]),
            y2=float(cam["y2"]),
        )
        cw = max(12.0, box.x2 - box.x1)
        ch = max(10.0, box.y2 - box.y1)
        camera_boxes.append(
            {
                "x1": box.x1 - max(8.0, cw * 0.9),
                "y1": box.y1 - max(4.0, ch * 0.3),
                "x2": box.x2 + max(8.0, cw * 0.9),
                "y2": box.y2 + max(4.0, ch * 0.3),
            }
        )
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=box,
                shape_class_guess="hourglass_like",
                has_opposing_triangle=True,
                score=0.82,
                source="hourglass",
            )
        )

    for pole in poles:
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=BoundingBox(
                    x1=float(pole["x1"]),
                    y1=float(pole["y1"]),
                    x2=float(pole["x2"]),
                    y2=float(pole["y2"]),
                ),
                shape_class_guess="bowtie_like",
                has_enclosing_square=True,
                score=0.8,
                source="bowtie",
            )
        )

    kept_tri: list[tuple[float, float]] = []
    for idx, cand in enumerate(triangles):
        cx, cy = cand["cx"], cand["cy"]
        if cy < y_min:
            continue
        in_hourglass = idx in hourglass_used
        in_circle = _triangle_inside_circle(cand, glyph_circles)
        in_square = _triangle_inside_box(cand, poles, pad=3.0)
        if not in_hourglass and any(
            box["x1"] <= cx <= box["x2"] and box["y1"] <= cy <= box["y2"]
            for box in camera_boxes
        ):
            continue
        if not in_circle and not in_square and triangle_is_noise(
            cand, dense_zoom=dense_zoom, near_callout=False
        ):
            continue
        if not in_circle:
            # Caption letters under an AP ring (the 'A' in AP) approx as triangles.
            under_ap = False
            for ccx, ccy, radius in glyph_circles:
                dist = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
                if cy > ccy and radius * 1.02 < dist <= radius * 2.4:
                    if any(
                        ((t["cx"] - ccx) ** 2 + (t["cy"] - ccy) ** 2) ** 0.5
                        <= radius * 1.15
                        for t in triangles
                    ):
                        under_ap = True
                        break
            if under_ap:
                continue
        if any(
            (cx - kx) ** 2 + (cy - ky) ** 2 <= TRIANGLE_DEDUPE_RADIUS**2
            for kx, ky in kept_tri
        ):
            continue
        kept_tri.append((cx, cy))
        bbox = BoundingBox(
            x1=float(cand["x"]),
            y1=float(cand["y"]),
            x2=float(cand["x"] + cand["w"]),
            y2=float(cand["y"] + cand["h"]),
        )
        if in_circle:
            for ccx, ccy, radius in glyph_circles:
                if ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5 <= radius * 1.15:
                    bbox = BoundingBox(
                        x1=float(ccx - radius),
                        y1=float(ccy - radius),
                        x2=float(ccx + radius),
                        y2=float(ccy + radius),
                    )
                    break
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=bbox,
                shape_class_guess="triangle_like",
                has_enclosing_circle=bool(in_circle),
                has_opposing_triangle=bool(in_hourglass),
                has_enclosing_square=bool(in_square),
                score=0.85,
                source="triangle",
                extras={"area": cand.get("area"), "dx": cand.get("dx"), "dy": cand.get("dy")},
            )
        )

    for stub in detect_conduit_stubs(image, exclude_top_pct=exclude_top_pct):
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=BoundingBox(
                    x1=float(stub["x1"]),
                    y1=float(stub["y1"]),
                    x2=float(stub["x2"]),
                    y2=float(stub["y2"]),
                ),
                shape_class_guess="conduit_stub_like",
                score=0.78,
                source="stub",
            )
        )

    ocr_spans = _ocr_text_spans(image, legend, exclude_top_pct=exclude_top_pct)
    used_digit_idx: set[int] = set()
    for i, span in enumerate(ocr_spans):
        if not span["is_digit"]:
            continue
        alias = str(span["alias"])
        # Raceway models only count when the digits sit on a nearby box,
        # and the overlay box must stay on the text — not the wall triangle.
        if not (alias.isdigit() and len(alias) >= 4):
            continue
        scx = (span["x1"] + span["x2"]) / 2.0
        scy = (span["y1"] + span["y2"]) / 2.0
        partner = None
        for rect in open_rects:
            dist = ((rect["cx"] - scx) ** 2 + (rect["cy"] - scy) ** 2) ** 0.5
            if dist <= 40.0:
                partner = rect
                break
        if partner is None:
            continue
        used_digit_idx.add(i)
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=BoundingBox(
                    x1=span["x1"],
                    y1=span["y1"],
                    x2=span["x2"],
                    y2=span["y2"],
                ),
                shape_class_guess="digit_label",
                ocr_text=span["alias"],
                ocr_conf=float(span["conf"]),
                score=min(1.0, float(span["conf"]) / 100.0),
                source="ocr",
                extras={"legend_hint": span["legend_hint"], "rect_digit_cluster": True},
            )
        )

    for i, span in enumerate(ocr_spans):
        if i in used_digit_idx:
            continue
        guess = "digit_label" if span["is_digit"] else "letter_tag"
        out.append(
            Candidate(
                tile_id=tile_id,
                bbox=BoundingBox(
                    x1=span["x1"], y1=span["y1"], x2=span["x2"], y2=span["y2"]
                ),
                shape_class_guess=guess,
                ocr_text=span["alias"],
                ocr_conf=float(span["conf"]),
                score=min(1.0, max(0.4, float(span["conf"]) / 100.0)),
                source="ocr",
                extras={"legend_hint": span["legend_hint"]},
            )
        )

    return out
