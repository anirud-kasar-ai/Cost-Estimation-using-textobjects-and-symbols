"""Detect data-drop callout boxes (e.g. ``1|13``) and reinforce ``#`` marks.

Callout pipe text is NOT a separate legend key. It locates / confirms nearby
DATA PERMANENT LINK drop triangles (``# = QTY``). Elevations like ``+48``
and room codes are ignored.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.cv.nms import BoundingBox, SymbolDetection

# Sheet/keynote style: 1|13, 1 | 5, 1|8, also OCR forms without a literal pipe.
_PIPE_CALLOUT_RE = re.compile(r"^(\d{1,2})\s*[|丨]\s*(\d{1,3})$")
_SPACE_DASH_SLASH_RE = re.compile(r"^(\d{1,2})\s*[-/\s]\s*(\d{1,3})$")
_ELEVATION_RE = re.compile(r"^\+\d{2,4}$")
_ROOM_LIKE_RE = re.compile(r"^[A-Z]-?\d", re.IGNORECASE)
# Sheet detail refs like 1/D5 — letter in second token.
_SHEET_REF_RE = re.compile(r"^\d{1,2}/[A-Z]\d", re.IGNORECASE)


def normalize_callout_text(text: str) -> str | None:
    """Return normalized ``a|b`` if ``text`` is a keynote callout, else None.

    Accepts ``1|13``, ``1 13``, ``1-13``, ``1/13`` (when second part is digits
    only), and compact OCR ``113`` when left is 1–2 digits and right 1–3.
    """
    raw = (text or "").strip().upper().replace("丨", "|")
    if not raw:
        return None
    if _ELEVATION_RE.match(raw.replace(" ", "")) or _ROOM_LIKE_RE.match(raw):
        return None
    if _SHEET_REF_RE.match(raw.replace(" ", "")):
        return None

    # Preserve separators for space/dash/slash forms before stripping.
    spaced = re.sub(r"\s+", " ", raw).strip()
    for pattern in (_PIPE_CALLOUT_RE, _SPACE_DASH_SLASH_RE):
        m = pattern.match(spaced)
        if m:
            return _format_pair(m.group(1), m.group(2))

    compact = re.sub(r"\s+", "", spaced)
    if _ELEVATION_RE.match(compact):
        return None
    m = _PIPE_CALLOUT_RE.match(compact) or re.match(r"^(\d{1,2})\|(\d{1,3})$", compact)
    if m:
        return _format_pair(m.group(1), m.group(2))
    m = re.match(r"^(\d{1,2})[-/](\d{1,3})$", compact)
    if m:
        return _format_pair(m.group(1), m.group(2))
    # OCR often drops the separator: "113" from "1 13" / "1|13".
    # Only accept leading sheet digit 1 (typical keynote) to avoid raceway numbers.
    m = re.match(r"^1(\d{1,2})$", compact)
    if m:
        return _format_pair("1", m.group(1))
    return None


def normalize_stacked_callout(line_a: str, line_b: str) -> str | None:
    """Combine vertically stacked OCR lines (``1`` then ``13``) into ``1|13``."""
    a = re.sub(r"[^0-9]", "", (line_a or "").strip())
    b = re.sub(r"[^0-9]", "", (line_b or "").strip())
    if not a or not b:
        return None
    if len(a) > 2 or len(b) > 3:
        return None
    return normalize_callout_text(f"{a}|{b}")


def _format_pair(left: str, right: str) -> str | None:
    try:
        a, b = int(left), int(right)
    except ValueError:
        return None
    if a < 1 or a > 99 or b < 1 or b > 999:
        return None
    # Avoid treating plain raceway-ish numbers alone as callouts when left>9
    # and no separator was present — handled by callers via patterns.
    return f"{a}|{b}"


def _pil_to_gray(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def detect_pipe_callout_hints(
    image: Image.Image,
    *,
    exclude_top_pct: float = 0.0,
) -> list[dict[str, Any]]:
    """OCR small boxes for ``1|n`` style callouts; return centers + optional qty digit."""
    if image.size[0] < 8 or image.size[1] < 8:
        return []
    try:
        import cv2
        import pytesseract
    except ImportError:
        return []

    from pipeline.cv.ocr_tags import tesseract_available, _configure_tesseract

    if not tesseract_available():
        return []
    _configure_tesseract()

    gray = _pil_to_gray(image)
    h, w = gray.shape[:2]
    y_min = int(h * max(0.0, exclude_top_pct))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if y_min > 0:
        binary[:y_min, :] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hints: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80 or area > 3500:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 10 or bh < 8:
            continue
        aspect = bw / float(bh)
        if aspect < 0.55 or aspect > 4.5:
            continue
        fill = area / float(max(1, bw * bh))
        # Framed callout boxes are mid-fill, not solid blobs.
        if fill < 0.12 or fill > 0.75:
            continue
        cy = y + bh / 2.0
        if cy < y_min:
            continue
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = 3.0
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _t, big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        normalized: str | None = None
        try:
            text = pytesseract.image_to_string(
                big,
                config="--psm 7 -c tessedit_char_whitelist=0123456789|-/",
            )
            normalized = normalize_callout_text((text or "").strip())
            if not normalized:
                # Stacked keynote: two digit lines inside the box.
                data = pytesseract.image_to_data(
                    big,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 6 -c tessedit_char_whitelist=0123456789",
                )
                digits = [
                    str(t or "").strip()
                    for t in (data.get("text") or [])
                    if str(t or "").strip().isdigit()
                ]
                if len(digits) >= 2:
                    normalized = normalize_stacked_callout(digits[0], digits[1])
        except Exception:  # noqa: BLE001
            continue
        if not normalized:
            continue
        # Nearby free digit as QTY (often outside the pipe box).
        qty = 1
        strip = gray[
            max(0, y - bh) : min(h, y + 2 * bh),
            max(0, x - int(bw * 1.5)) : min(w, x + int(bw * 2.5)),
        ]
        if strip.size:
            strip_big = cv2.resize(strip, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            try:
                data = pytesseract.image_to_data(
                    strip_big,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 6 -c tessedit_char_whitelist=0123456789",
                )
                for i, tok in enumerate(data.get("text") or []):
                    tok = str(tok or "").strip()
                    if tok.isdigit() and 1 <= int(tok) <= 9:
                        qty = int(tok)
                        break
            except Exception:  # noqa: BLE001
                pass

        hints.append(
            {
                "text": normalized,
                "cx": x + bw / 2.0,
                "cy": cy,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "qty": qty,
            }
        )
    return hints


def reinforce_hash_from_callouts(
    *,
    image: Image.Image,
    hash_key: str,
    existing: list[SymbolDetection],
    callout_hints: list[dict[str, Any]] | None = None,
    exclude_top_pct: float = 0.0,
    radius_px: float | None = None,
) -> list[SymbolDetection]:
    """Confirm / seed ``#`` drops from pipe callouts without double-counting."""
    radius = float(radius_px if radius_px is not None else config.CALLOUT_DROP_RADIUS_PX)
    hints = callout_hints
    if hints is None:
        hints = detect_pipe_callout_hints(image, exclude_top_pct=exclude_top_pct)
    if not hints:
        return existing

    from pipeline.cv.triangle_drops import find_triangle_tip_near

    triangles = [d for d in existing if d.source in {"triangle", "callout_drop"}]
    others = [d for d in existing if d.source not in {"triangle", "callout_drop"}]
    updated: list[SymbolDetection] = list(triangles)

    def _center(d: SymbolDetection) -> tuple[float, float]:
        return ((d.box.x1 + d.box.x2) / 2.0, (d.box.y1 + d.box.y2) / 2.0)

    for hint in hints:
        hx, hy = float(hint["cx"]), float(hint["cy"])
        hq = max(1, int(hint.get("qty") or 1))
        matched_idx: int | None = None
        best_dist = radius + 1.0
        for i, det in enumerate(updated):
            cx, cy = _center(det)
            dist = ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5
            if dist <= radius and dist < best_dist:
                best_dist = dist
                matched_idx = i
        if matched_idx is not None:
            det = updated[matched_idx]
            # Prefer higher QTY when callout cluster implies a digit.
            new_qty = max(max(1, int(det.qty)), hq)
            if new_qty != det.qty:
                updated[matched_idx] = SymbolDetection(
                    symbol=det.symbol,
                    box=det.box,
                    score=det.score,
                    source=det.source,
                    qty=new_qty,
                    tile_file=det.tile_file,
                )
            continue

        tip = find_triangle_tip_near(image, hx, hy, radius_px=radius)
        if tip is not None:
            pad = max(4.0, min(tip["w"], tip["h"]) * 0.25)
            updated.append(
                SymbolDetection(
                    symbol=hash_key,
                    box=BoundingBox(
                        x1=max(0.0, tip["x"] - pad),
                        y1=max(0.0, tip["y"] - pad),
                        x2=tip["x"] + tip["w"] + pad,
                        y2=tip["y"] + tip["h"] + pad,
                    ),
                    score=0.78,
                    source="callout_drop",
                    qty=hq,
                )
            )
            continue

        # No nearby triangle tip — seed a drop at the callout (evidence of a data drop).
        pad = 12.0
        updated.append(
            SymbolDetection(
                symbol=hash_key,
                box=BoundingBox(
                    x1=max(0.0, hx - pad),
                    y1=max(0.0, hy - pad),
                    x2=hx + pad,
                    y2=hy + pad,
                ),
                score=0.72,
                source="callout_drop",
                qty=hq,
            )
        )

    return others + updated


__all__ = [
    "detect_pipe_callout_hints",
    "normalize_callout_text",
    "normalize_stacked_callout",
    "reinforce_hash_from_callouts",
]
