from __future__ import annotations

import logging
import shutil

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.cv.tag_match import match_span_to_text_key, text_legend_keys
from pipeline.extraction.symbol_table_extractor import SymbolTableInfo

logger = logging.getLogger(__name__)


def tesseract_available() -> bool:
    if config.TESSERACT_CMD:
        return True
    return shutil.which("tesseract") is not None


def _configure_tesseract() -> None:
    if not config.TESSERACT_CMD:
        return
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def preprocess_for_ocr(image: Image.Image, *, upscale: bool = True) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if upscale:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(binary)
    return binary


def _unrotate_box(
    rot: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    orig_w: float,
    orig_h: float,
) -> tuple[float, float, float, float]:
    """Map a box from a PIL-rotated (expand=True) canvas back to upright."""
    if rot == 0:
        return x1, y1, x2, y2
    # PIL Image.rotate is counter-clockwise.
    if rot == 90:
        # view size (H, W); (xv,yv) came from (W-1-yv, xv)
        ux1 = orig_w - y2
        uy1 = x1
        ux2 = orig_w - y1
        uy2 = x2
        return ux1, uy1, ux2, uy2
    if rot == 270:
        # view size (H, W); (xv,yv) came from (yv, H-1-xv)
        ux1 = y1
        uy1 = orig_h - x2
        ux2 = y2
        uy2 = orig_h - x1
        return ux1, uy1, ux2, uy2
    return x1, y1, x2, y2


def detect_tags_from_ocr(
    image: Image.Image,
    legend: SymbolTableInfo,
    *,
    exclude_top_pct: float | None = None,
    near_centers: list[tuple[float, float]] | None = None,
) -> list[SymbolDetection]:
    """Detect legend character tags and part-number labels via OCR.

    Runs upright plus 90° / 270° passes so vertical plan text (e.g. \"2300\"
    next to Wiremold raceway) is matched to legend part numbers like WM2300.
    When ``near_centers`` is provided (drop-triangle centers), also OCR a
    narrow strip beside each center with a digit whitelist — that recovers
    raceway labels Tesseract otherwise misreads as 3000/25008/etc.
    """
    if not tesseract_available():
        logger.warning("Tesseract not available — OCR tag detection skipped")
        return []

    import pytesseract

    _configure_tesseract()
    keys = text_legend_keys(legend)
    if not keys:
        return []

    exclude_top_pct = float(
        exclude_top_pct if exclude_top_pct is not None else config.CV_TITLE_BAND_PCT
    )
    orig_w, orig_h = image.size
    y_min = orig_h * exclude_top_pct
    min_conf = float(config.CV_OCR_MIN_CONF)
    min_h = float(config.CV_OCR_MIN_HEIGHT)
    max_h = float(config.CV_OCR_MAX_HEIGHT) * 1.5
    scale = 2.0

    out: list[SymbolDetection] = []
    for rot in (0, 90, 270):
        view = image if rot == 0 else image.rotate(rot, expand=True)
        preprocessed = preprocess_for_ocr(view, upscale=True)
        data = pytesseract.image_to_data(
            preprocessed,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",
        )
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
            is_part_alias = any(text_upper == alias for alias, _ in keys)
            conf_floor = 40.0 if is_part_alias else min_conf
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

            matched = match_span_to_text_key(text, keys, size=glyph_size)
            if not matched:
                continue

            x1, y1, x2, y2 = _unrotate_box(
                rot, x, y, x + w, y + h, orig_w=float(orig_w), orig_h=float(orig_h)
            )
            xa, xb = sorted((x1, x2))
            ya, yb = sorted((y1, y2))
            if ya < y_min:
                continue
            xa = max(0.0, min(float(orig_w), xa))
            ya = max(0.0, min(float(orig_h), ya))
            xb = max(0.0, min(float(orig_w), xb))
            yb = max(0.0, min(float(orig_h), yb))
            if xb - xa < 2 or yb - ya < 2:
                continue

            out.append(
                SymbolDetection(
                    symbol=matched,
                    box=BoundingBox(x1=xa, y1=ya, x2=xb, y2=yb),
                    score=min(1.0, conf / 100.0),
                    source="ocr",
                )
            )

    if near_centers:
        out.extend(_ocr_part_labels_near_centers(image, keys, near_centers, y_min))

    return out


def _ocr_part_labels_near_centers(
    image: Image.Image,
    keys: list[tuple[str, str]],
    centers: list[tuple[float, float]],
    y_min: float,
) -> list[SymbolDetection]:
    """Digit-whitelisted OCR near each drop triangle for raceway part labels.

    On these drawings Wiremold labels (2300/5400/5500) are printed vertically
    along the wall the triangle points at — often below/above the mark rather
    than beside its body.
    """
    import cv2
    import pytesseract
    import numpy as np

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    numeric_aliases = {a for a, _ in keys if a.isdigit()}
    if not numeric_aliases:
        return []

    out: list[SymbolDetection] = []
    seen_boxes: list[tuple[float, float, float, float]] = []

    def _overlaps(box: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = box
        for ax1, ay1, ax2, ay2 in seen_boxes:
            ix1, iy1 = max(x1, ax1), max(y1, ay1)
            ix2, iy2 = min(x2, ax2), min(y2, ay2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / max(1.0, (x2 - x1) * (y2 - y1)) > 0.3:
                    return True
        return False

    for cx, cy in centers:
        if cy < y_min:
            continue
        # Tall strips along the wall line (vertical text) and short side strips.
        strips = [
            (int(cx - 20), int(cy + 10), int(cx + 45), int(cy + 140)),  # below
            (int(cx - 20), int(cy - 140), int(cx + 45), int(cy - 10)),  # above
            (int(cx + 10), int(cy - 80), int(cx + 70), int(cy + 80)),  # right
            (int(cx - 70), int(cy - 80), int(cx - 10), int(cy + 80)),  # left
        ]
        for sx1, sy1, sx2, sy2 in strips:
            sx1 = max(0, min(w - 1, sx1))
            sx2 = max(0, min(w, sx2))
            sy1 = max(0, min(h - 1, sy1))
            sy2 = max(0, min(h, sy2))
            if sx2 - sx1 < 8 or sy2 - sy1 < 20:
                continue
            if _overlaps((sx1, sy1, sx2, sy2)):
                continue
            crop = gray[sy1:sy2, sx1:sx2]
            found = False
            for rot in (0, 90, 270):
                view = (
                    crop
                    if rot == 0
                    else np.array(Image.fromarray(crop).rotate(rot, expand=True))
                )
                view = cv2.resize(
                    view, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
                )
                _t, bw = cv2.threshold(
                    view, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                for psm in (7, 6, 8):
                    data = pytesseract.image_to_data(
                        bw,
                        output_type=pytesseract.Output.DICT,
                        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
                    )
                    for i, text in enumerate(data.get("text") or []):
                        text = str(text or "").strip()
                        if text not in numeric_aliases:
                            continue
                        try:
                            conf = float(data["conf"][i])
                        except (TypeError, ValueError):
                            conf = 0.0
                        if conf < 35 and conf != 0:
                            continue
                        # conf 0 is common for clean isolated digits on small crops
                        matched = match_span_to_text_key(text, keys, size=14)
                        if not matched:
                            continue
                        box = (float(sx1), float(sy1), float(sx2), float(sy2))
                        if _overlaps(box):
                            continue
                        seen_boxes.append(box)
                        out.append(
                            SymbolDetection(
                                symbol=matched,
                                box=BoundingBox(
                                    x1=box[0], y1=box[1], x2=box[2], y2=box[3]
                                ),
                                score=min(1.0, max(0.55, conf / 100.0 if conf else 0.7)),
                                source="ocr",
                            )
                        )
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
    return out


def ap_skip_centers_from_ocr(detections: list[SymbolDetection]) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for det in detections:
        sym = det.symbol.upper()
        if sym == "AP" or "ACCESS POINT" in sym or sym.startswith("TAP"):
            cx = (det.box.x1 + det.box.x2) / 2.0
            cy = (det.box.y1 + det.box.y2) / 2.0
            centers.append((cx, cy))
    return centers
