"""Detect solid triangle data-drop marks and boxed hourglass cameras.

On these drawings a NETWORK CAMERA glyph is two solid triangles pointing at
a small X-box between them, often with quantity digits outside:

    4  ▶▶ [X] ◀◀  4

Those outer digits sit next to *smaller* flanking drop triangles (or on the
outer tips). Morphological opening helps separate touching shapes but can
erase the tiny outer drops on zoomed tiles, so candidates are collected from
both the raw and opened binaries. Camera pairing is scale-independent
(relative size + opposing tips + small gap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config

# Absolute area floors are intentionally low so zoomed tiles still see the
# small outer drop triangles; camera pairing does not rely on absolute size.
TRIANGLE_MIN_AREA = 50
TRIANGLE_MAX_AREA = 1200
TRIANGLE_MIN_AREA_DENSE = 28  # 653px zoom tiles — smaller tips survive
TRIANGLE_MAX_AREA_DENSE = 1600
TRIANGLE_SKIP_RADIUS = 90.0
TRIANGLE_SKIP_RADIUS_DENSE = 36.0  # wall drops sit near false circles on 653 tiles
TRIANGLE_DEDUPE_RADIUS = 22.0
DENSE_ZOOM_MAX_EDGE = 800  # treat image as dense zoom when max(w,h) <= this



def _pil_to_gray(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _tip_direction(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Return (tip_x, tip_y, dir_x, dir_y) for a 3-vertex triangle."""
    # Tip = vertex farthest from the midpoint of the other two (the base).
    best_i = 0
    best_dist = -1.0
    for i in range(3):
        base = (pts[(i + 1) % 3] + pts[(i + 2) % 3]) / 2.0
        dist = float(np.linalg.norm(pts[i] - base))
        if dist > best_dist:
            best_dist = dist
            best_i = i
    tip = pts[best_i]
    base = (pts[(best_i + 1) % 3] + pts[(best_i + 2) % 3]) / 2.0
    direction = tip - base
    norm = float(np.linalg.norm(direction)) or 1.0
    return float(tip[0]), float(tip[1]), float(direction[0] / norm), float(direction[1] / norm)


def _triangle_candidates(
    binary: np.ndarray,
    *,
    dense_zoom: bool = False,
) -> list[dict[str, float]]:
    """Solid triangle blobs with tip direction."""
    import cv2

    min_area = TRIANGLE_MIN_AREA_DENSE if dense_zoom else TRIANGLE_MIN_AREA
    max_area = TRIANGLE_MAX_AREA_DENSE if dense_zoom else TRIANGLE_MAX_AREA
    fill_lo = 0.22 if dense_zoom else 0.30
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict[str, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = None
        for eps in (0.04, 0.06, 0.08, 0.1, 0.12):
            candidate = cv2.approxPolyDP(contour, eps * peri, True)
            if len(candidate) == 3:
                approx = candidate
                break
        if approx is None:
            continue
        x, y, width, height = cv2.boundingRect(approx)
        if width <= 0 or height <= 0:
            continue
        fill = area / float(width * height)
        if fill < fill_lo or fill > 0.85:
            continue
        ratio = width / float(height)
        if ratio < 0.25 or ratio > 3.0:
            continue
        pts = approx.reshape(3, 2).astype(float)
        tip_x, tip_y, dx, dy = _tip_direction(pts)
        out.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(width),
                "h": float(height),
                "area": float(area),
                "cx": x + width / 2.0,
                "cy": y + height / 2.0,
                "tip_x": tip_x,
                "tip_y": tip_y,
                "dx": dx,
                "dy": dy,
            }
        )
    return out


def _merge_candidates(groups: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    """Dedupe triangles found under multiple binarisations."""
    merged: list[dict[str, float]] = []
    for group in groups:
        for cand in group:
            if any(
                (cand["cx"] - m["cx"]) ** 2 + (cand["cy"] - m["cy"]) ** 2
                <= TRIANGLE_DEDUPE_RADIUS**2
                for m in merged
            ):
                # Prefer the larger / better-filled observation.
                for i, m in enumerate(merged):
                    if (cand["cx"] - m["cx"]) ** 2 + (cand["cy"] - m["cy"]) ** 2 <= (
                        TRIANGLE_DEDUPE_RADIUS**2
                    ):
                        if cand["area"] > m["area"]:
                            merged[i] = cand
                        break
                continue
            merged.append(cand)
    return merged


def _pair_hourglass_cameras(
    candidates: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], set[int]]:
    """Pair opposing tip-to-tip triangles into NETWORK CAMERA glyphs.

    Works at any zoom: similar area, aligned vertically, tips facing each
    other across a small gap (possibly with an X-box between them).
    """
    cameras: list[dict[str, Any]] = []
    used: set[int] = set()
    n = len(candidates)
    for i in range(n):
        if i in used:
            continue
        a = candidates[i]
        best_j: int | None = None
        best_score = 1e18
        for j in range(i + 1, n):
            if j in used:
                continue
            b = candidates[j]
            # Similar size (allow 2.5× for mixed half detections).
            if max(a["area"], b["area"]) / max(1.0, min(a["area"], b["area"])) > 2.5:
                continue
            # Roughly same row.
            y_tol = max(12.0, 0.6 * max(a["h"], b["h"]))
            if abs(a["cy"] - b["cy"]) > y_tol:
                continue
            # Side-by-side with a small gap (or slight overlap after padding).
            left, right = (a, b) if a["cx"] < b["cx"] else (b, a)
            gap = right["x"] - (left["x"] + left["w"])
            max_gap = max(18.0, 0.9 * max(left["w"], right["w"]))
            if gap > max_gap or gap < -max(left["w"], right["w"]) * 0.4:
                continue
            # Tips should point toward each other (opposing horizontal dirs).
            # left.dx > 0 (points right), right.dx < 0 (points left).
            if left["dx"] * right["dx"] > 0:
                # Same horizontal direction — not an hourglass.
                # Allow mostly-vertical tips only if tips are still near the gap.
                if abs(left["dx"]) < 0.35 and abs(right["dx"]) < 0.35:
                    pass
                else:
                    continue
            else:
                if left["dx"] < 0.15 or right["dx"] > -0.15:
                    # Not clearly facing inward.
                    if gap > 4:
                        continue
            tip_gap = right["tip_x"] - left["tip_x"]
            if tip_gap < -5:
                continue
            score = abs(gap) + abs(a["cy"] - b["cy"]) + abs(a["area"] - b["area"]) * 0.01
            if score < best_score:
                best_score = score
                best_j = j
        if best_j is None:
            continue
        b = candidates[best_j]
        x1 = min(a["x"], b["x"]) - 6.0
        y1 = min(a["y"], b["y"]) - 6.0
        x2 = max(a["x"] + a["w"], b["x"] + b["w"]) + 6.0
        y2 = max(a["y"] + a["h"], b["y"] + b["h"]) + 6.0
        cameras.append(
            {
                "x1": max(0.0, x1),
                "y1": max(0.0, y1),
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )
        used.update((i, best_j))
    return cameras, used


def _detect_xbox_cameras(binary: np.ndarray) -> list[dict[str, Any]]:
    """Detect the small square hourglass (X-box) that sits in a NETWORK CAMERA glyph."""
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cameras: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80 or area > 900:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 10 or height < 10:
            continue
        ratio = width / float(height)
        if ratio < 0.7 or ratio > 1.45:
            continue
        fill = area / float(width * height)
        # The X-box is a framed square; ink fill is mid-range, not a solid blob.
        if fill < 0.25 or fill > 0.70:
            continue
        # Interior should have a bright (empty) cross — sample the center.
        cx, cy = x + width // 2, y + height // 2
        patch = binary[y : y + height, x : x + width]
        if patch.size == 0:
            continue
        # Center 3x3 should have mixed ink (X crossing), edges denser.
        inner = patch[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3]
        if inner.size == 0:
            continue
        inner_fill = float(np.mean(inner > 0))
        if not (0.15 <= inner_fill <= 0.85):
            continue
        cameras.append(
            {
                "x1": float(max(0, x - 4)),
                "y1": float(max(0, y - 4)),
                "x2": float(x + width + 4),
                "y2": float(y + height + 4),
                "cx": float(cx),
                "cy": float(cy),
            }
        )
    return cameras


def _merge_cameras(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for group in groups:
        for cam in group:
            if any(
                abs(cam["cx"] - m["cx"]) < 30 and abs(cam["cy"] - m["cy"]) < 30
                for m in merged
            ):
                continue
            merged.append(cam)
    return merged


def _read_qty_digit(gray: np.ndarray, box: dict[str, Any]) -> int | None:
    """Read the quantity digit printed beside a drop triangle (``# = QTY``)."""
    try:
        import cv2
        import pytesseract
    except ImportError:
        return None

    h, w = gray.shape[:2]
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    y1 = int(max(0, box["cy"] - bh * 0.55))
    y2 = int(min(h, box["cy"] + bh * 0.55))
    if y2 - y1 < 8:
        return None

    tri = gray[
        int(max(0, box["y1"])) : int(min(h, box["y2"])),
        int(max(0, box["x1"])) : int(min(w, box["x2"])),
    ]
    left_strip = ((box["x1"] - bw * 2.2, box["x1"] + bw * 0.2), (y1, y2))
    right_strip = ((box["x2"] - bw * 0.2, box["x2"] + bw * 2.2), (y1, y2))
    x_band = (box["cx"] - bw * 1.9, box["cx"] + bw * 1.3)
    top_strip = (x_band, (box["y1"] - bh * 0.95, box["y1"] + bh * 0.2))
    bottom_strip = (x_band, (box["y2"] - bh * 0.2, box["y2"] + bh * 0.95))

    strips = [left_strip, right_strip]
    if tri.size:
        ink = (tri < 128).astype(np.uint8)
        half_x = max(1, ink.shape[1] // 2)
        half_y = max(1, ink.shape[0] // 2)
        total = max(1, int(ink.sum()))
        dx = (int(ink[:, half_x:].sum()) - int(ink[:, :half_x].sum())) / total
        dy = (int(ink[half_y:, :].sum()) - int(ink[:half_y, :].sum())) / total
        if abs(dy) > abs(dx) and abs(dy) > 0.1:
            strips = [top_strip, bottom_strip] if dy < 0 else [bottom_strip, top_strip]
        elif dx > 0.2:
            strips = [right_strip, left_strip]

    for (side_x1, side_x2), (side_y1, side_y2) in strips:
        sx1 = int(max(0, side_x1))
        sx2 = int(min(w, side_x2))
        sy1 = int(max(0, side_y1))
        sy2 = int(min(h, side_y2))
        if sx2 - sx1 < 8 or sy2 - sy1 < 8:
            continue
        raw = gray[sy1:sy2, sx1:sx2]
        dark = int((raw < 128).sum())
        if dark < 20 or dark / raw.size < 0.004:
            continue
        strip = cv2.resize(raw, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _t, strip = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates: list[tuple[float, float, int]] = []
        for psm in (7, 10, 6):
            data = pytesseract.image_to_data(
                strip,
                output_type=pytesseract.Output.DICT,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
            )
            for i, text in enumerate(data.get("text") or []):
                text = str(text or "").strip()
                if not text.isdigit():
                    continue
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    continue
                value = int(text)
                glyph_h = float(data["height"][i]) / 3.0
                if not (bh * 0.25 <= glyph_h <= bh * 1.6):
                    continue
                if conf < 0 or not (1 <= value <= 9):
                    continue
                word_cx = sx1 + (float(data["left"][i]) + float(data["width"][i]) / 2) / 3.0
                word_cy = sy1 + (float(data["top"][i]) + float(data["height"][i]) / 2) / 3.0
                dist = ((word_cx - box["cx"]) ** 2 + (word_cy - box["cy"]) ** 2) ** 0.5
                candidates.append((dist, -conf, value))
            if candidates:
                break
        if candidates:
            candidates.sort()
            return candidates[0][2]
    return None


def _detect_glyph_circles(binary: np.ndarray) -> list[tuple[float, float, float]]:
    """Circular device outlines (AP/WP: triangle inside a ring)."""
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles: list[tuple[float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 350 or area > 12000:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.62:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 14 or height < 14:
            continue
        aspect = width / float(height)
        if aspect < 0.65 or aspect > 1.55:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 8.0:
            continue
        circles.append((float(cx), float(cy), float(radius)))
    return circles


def _triangle_inside_circle(
    cand: dict[str, float],
    circles: list[tuple[float, float, float]],
    *,
    margin: float = 1.05,
) -> bool:
    cx, cy = cand["cx"], cand["cy"]
    for ccx, ccy, radius in circles:
        dist = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
        if dist <= radius * margin:
            return True
    return False


def device_glyph_skip_centers(
    image: Image.Image | Path,
    *,
    exclude_top_pct: float = 0.12,
) -> list[tuple[float, float]]:
    """Centers where generic triangle CV should not fire (AP circle glyphs).

    Only circles that contain a solid triangle (true AP/WP-style glyph) are
    returned. Bare circularity on raceway/door arcs must not wipe nearby ``#``
    drops within ``TRIANGLE_SKIP_RADIUS``.
    """
    import cv2

    if isinstance(image, Path):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    else:
        gray = _pil_to_gray(image)
    if gray is None:
        return []

    h, w = gray.shape[:2]
    y_min = int(h * max(0.0, exclude_top_pct))
    dense_zoom = max(h, w) <= DENSE_ZOOM_MAX_EDGE
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    if y_min > 0:
        binary[:y_min, :] = 0
    circles = _detect_glyph_circles(binary)
    triangles = _triangle_candidates(binary, dense_zoom=dense_zoom)
    out: list[tuple[float, float]] = []
    for cx, cy, radius in circles:
        if cy < y_min:
            continue
        # Require a triangle tip near the circle center (AP ring + tip).
        if any(
            ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5 <= radius * 1.15
            for t in triangles
        ):
            out.append((cx, cy))
    return out


def detect_drop_marks(
    image: Image.Image | Path,
    *,
    skip_centers: list[tuple[float, float]] | None = None,
    exclude_top_pct: float = 0.12,
    read_quantities: bool = True,
    protect_centers: list[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect solid triangle data drops and boxed-hourglass camera symbols.

    Returns (drop_boxes, camera_boxes).
    ``protect_centers`` (e.g. callout boxes) softens the tiny-upward-tip discard
    so wall-adjacent drops near keynotes survive on dense zoom tiles.
    """
    import cv2

    if isinstance(image, Path):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    else:
        gray = _pil_to_gray(image)
    if gray is None:
        return [], []

    h, w = gray.shape[:2]
    y_min = int(h * max(0.0, exclude_top_pct))
    dense_zoom = max(h, w) <= DENSE_ZOOM_MAX_EDGE
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    if y_min > 0:
        binary[:y_min, :] = 0
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Raw keeps tiny outer drop triangles that opening erases on zoom tiles;
    # opened separates triangles that touch the camera X-box.
    candidates = _merge_candidates(
        [
            _triangle_candidates(binary, dense_zoom=dense_zoom),
            _triangle_candidates(opened, dense_zoom=dense_zoom),
        ]
    )
    pair_cams, used = _pair_hourglass_cameras(candidates)
    xbox_cams = _detect_xbox_cameras(binary) + _detect_xbox_cameras(opened)
    cameras = _merge_cameras([pair_cams, xbox_cams])
    glyph_circles = _detect_glyph_circles(binary) + _detect_glyph_circles(opened)
    # Dedupe circle observations.
    merged_circles: list[tuple[float, float, float]] = []
    for cx, cy, radius in glyph_circles:
        if any(
            (cx - mc[0]) ** 2 + (cy - mc[1]) ** 2 <= max(radius, mc[2]) ** 2 for mc in merged_circles
        ):
            continue
        merged_circles.append((cx, cy, radius))
    glyph_circles = merged_circles

    # Expand camera boxes that came from X-box alone so neighbouring tip
    # triangles of the same glyph are excluded from the drop list.
    expanded_cams: list[dict[str, Any]] = []
    for cam in cameras:
        cw = cam["x2"] - cam["x1"]
        ch = cam["y2"] - cam["y1"]
        expanded_cams.append(
            {
                **cam,
                "x1": cam["x1"] - max(8.0, cw * 0.9),
                "y1": cam["y1"] - max(4.0, ch * 0.3),
                "x2": cam["x2"] + max(8.0, cw * 0.9),
                "y2": cam["y2"] + max(4.0, ch * 0.3),
            }
        )

    skip = list(skip_centers or [])
    protect = list(protect_centers or [])
    protect_r2 = float(config.CALLOUT_DROP_RADIUS_PX) ** 2
    kept: list[tuple[float, float]] = []
    drop_boxes: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        if idx in used:
            continue  # hourglass half — counts as camera, not a # drop
        cx, cy = cand["cx"], cand["cy"]
        # AP/WP glyphs are a solid triangle inside a circular ring — not # drops.
        if _triangle_inside_circle(cand, glyph_circles):
            continue
        # X-box corners often approx as tiny upward triangles — skip them.
        # Soften when a callout keynote is nearby (wall data drops).
        near_callout = any(
            (cx - px) ** 2 + (cy - py) ** 2 <= protect_r2 for px, py in protect
        )
        tiny_floor = 90 if dense_zoom else 160
        if near_callout:
            tiny_floor = 40 if dense_zoom else 80
        if cand["area"] < tiny_floor and abs(cand["dy"]) > 0.75:
            continue
        if any(
            cam["x1"] <= cx <= cam["x2"] and cam["y1"] <= cy <= cam["y2"]
            for cam in expanded_cams
        ):
            # Triangle tip that is part of the camera glyph assembly.
            # Keep it only if it sits clearly outside and looks like a
            # separate flanking drop (handled via qty marks instead).
            continue
        if any(
            (cx - sx) ** 2 + (cy - sy) ** 2
            <= (TRIANGLE_SKIP_RADIUS_DENSE if dense_zoom else TRIANGLE_SKIP_RADIUS) ** 2
            for sx, sy in skip
        ):
            continue
        if any((cx - kx) ** 2 + (cy - ky) ** 2 <= TRIANGLE_DEDUPE_RADIUS**2 for kx, ky in kept):
            continue
        kept.append((cx, cy))
        pad = max(4.0, min(cand["w"], cand["h"]) * 0.25)
        drop_boxes.append(
            {
                "x1": max(0.0, cand["x"] - pad),
                "y1": max(0.0, cand["y"] - pad),
                "x2": cand["x"] + cand["w"] + pad,
                "y2": cand["y"] + cand["h"] + pad,
                "cx": cx,
                "cy": cy,
            }
        )

    if read_quantities:
        for box in drop_boxes:
            box["qty"] = _read_qty_digit(gray, box) or 1
        # Camera glyphs carry quantity digits outside each tip (4 ▶[X]◀ 4).
        drop_boxes.extend(
            _qty_marks_beside_cameras(gray, cameras, existing=drop_boxes)
        )
        # Edge-clipped drops sometimes miss the digit on the first pass —
        # retry with a wider left/right band before defaulting to 1.
        for box in drop_boxes:
            if box.get("qty", 1) != 1:
                continue
            wider = dict(box)
            pad_x = max(18.0, (box["x2"] - box["x1"]) * 1.2)
            wider["x1"] = max(0.0, box["cx"] - pad_x)
            wider["x2"] = min(float(gray.shape[1]), box["cx"] + pad_x)
            wider["y1"] = max(0.0, box["cy"] - (box["y2"] - box["y1"]))
            wider["y2"] = min(float(gray.shape[0]), box["cy"] + (box["y2"] - box["y1"]))
            retry = _read_qty_digit(gray, wider)
            if retry and retry != 1:
                box["qty"] = retry

    return drop_boxes, cameras


def _qty_marks_beside_cameras(
    gray: np.ndarray,
    cameras: list[dict[str, Any]],
    *,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create synthetic drop marks from digits flanking a camera glyph."""
    import cv2
    import pytesseract

    h, w = gray.shape[:2]
    out: list[dict[str, Any]] = []
    for cam in cameras:
        cw = max(20.0, cam["x2"] - cam["x1"])
        ch = max(16.0, cam["y2"] - cam["y1"])
        # Digits sit immediately outside each tip — overlap slightly into the
        # camera box so we do not miss digits glued to the outer edge.
        strips = [
            (
                int(max(0, cam["x1"] - cw * 1.1)),
                int(max(0, cam["cy"] - ch * 0.75)),
                int(min(w, cam["x1"] + cw * 0.35)),
                int(min(h, cam["cy"] + ch * 0.75)),
            ),
            (
                int(max(0, cam["x2"] - cw * 0.35)),
                int(max(0, cam["cy"] - ch * 0.75)),
                int(min(w, cam["x2"] + cw * 1.1)),
                int(min(h, cam["cy"] + ch * 0.75)),
            ),
        ]
        for sx1, sy1, sx2, sy2 in strips:
            if sx2 - sx1 < 8 or sy2 - sy1 < 8:
                continue
            scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
            if any(
                (scx - d["cx"]) ** 2 + (scy - d["cy"]) ** 2 <= 28**2 for d in existing + out
            ):
                continue
            crop = gray[sy1:sy2, sx1:sx2]
            # Ignore strips that are almost solid ink (triangle body).
            if float(np.mean(crop < 128)) > 0.55:
                continue
            view = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            _t, bw = cv2.threshold(view, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            qty = None
            for psm in (7, 10, 8):
                data = pytesseract.image_to_data(
                    bw,
                    output_type=pytesseract.Output.DICT,
                    config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
                )
                for i, text in enumerate(data.get("text") or []):
                    text = str(text or "").strip()
                    if not text.isdigit():
                        continue
                    # Tesseract sometimes doubles a clean digit ("44" for "4").
                    if len(text) == 1:
                        value = int(text)
                    elif len(text) <= 3 and len(set(text)) == 1:
                        value = int(text[0])
                    else:
                        continue
                    try:
                        conf = float(data["conf"][i])
                    except (TypeError, ValueError):
                        continue
                    if (conf >= 35 or conf == 0) and 1 <= value <= 9:
                        qty = value
                        break
                if qty is not None:
                    break
            if qty is None:
                continue
            out.append(
                {
                    "x1": float(sx1),
                    "y1": float(sy1),
                    "x2": float(sx2),
                    "y2": float(sy2),
                    "cx": scx,
                    "cy": scy,
                    "qty": qty,
                }
            )
    return out


def count_drop_triangles(
    image: Image.Image | Path,
    *,
    skip_centers: list[tuple[float, float]] | None = None,
    exclude_top_pct: float = 0.12,
) -> tuple[int, list[dict[str, Any]]]:
    """Count solid triangle data drops; kept for API compatibility."""
    drop_boxes, _cameras = detect_drop_marks(
        image,
        skip_centers=skip_centers,
        exclude_top_pct=exclude_top_pct,
    )
    return len(drop_boxes), drop_boxes


def find_triangle_tip_near(
    image: Image.Image | Path,
    cx: float,
    cy: float,
    *,
    radius_px: float | None = None,
) -> dict[str, float] | None:
    """Find the nearest solid triangle tip within ``radius_px`` of (cx, cy).

    Uses a lowered area floor so tips that failed global dense filters can
    still reinforce a callout keynote into a ``#`` mark.
    """
    import cv2

    radius = float(radius_px if radius_px is not None else config.CALLOUT_DROP_RADIUS_PX)
    if isinstance(image, Path):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    else:
        gray = _pil_to_gray(image)
    if gray is None:
        return None
    h, w = gray.shape[:2]
    x0 = max(0, int(cx - radius))
    y0 = max(0, int(cy - radius))
    x1 = min(w, int(cx + radius))
    y1 = min(h, int(cy + radius))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = gray[y0:y1, x0:x1]
    _, binary = cv2.threshold(crop, 80, 255, cv2.THRESH_BINARY_INV)
    # Local search: allow very small tips.
    cands = _triangle_candidates(binary, dense_zoom=True)
    # Also try with absolute floor override via second pass on opened mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cands.extend(_triangle_candidates(opened, dense_zoom=True))

    best: dict[str, float] | None = None
    best_dist = radius + 1.0
    for cand in cands:
        if cand["area"] < 18:
            continue
        tx = cand["cx"] + x0
        ty = cand["cy"] + y0
        dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
        if dist <= radius and dist < best_dist:
            best_dist = dist
            best = {
                "x": cand["x"] + x0,
                "y": cand["y"] + y0,
                "w": cand["w"],
                "h": cand["h"],
                "cx": tx,
                "cy": ty,
                "area": cand["area"],
            }
    return best
