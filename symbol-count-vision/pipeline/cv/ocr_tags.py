from __future__ import annotations

import logging
import re
import shutil
import threading
from typing import Any

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


_ocr_pass_lock = threading.Lock()
_ocr_pass_cache: dict[tuple[int, tuple[int, ...]], list[tuple[int, dict[str, Any]]]] = {}


def clear_ocr_page_cache(image: Image.Image | None = None) -> None:
    with _ocr_pass_lock:
        if image is None:
            _ocr_pass_cache.clear()
            return
        drop = [key for key in _ocr_pass_cache if key[0] == id(image)]
        for key in drop:
            _ocr_pass_cache.pop(key, None)


def ocr_page_passes(image: Image.Image) -> list[tuple[int, dict[str, Any]]]:
    """Run Tesseract once per configured rotation; reuse within a tile."""
    if not tesseract_available():
        return []
    import pytesseract

    rotations = config.ocr_rotations()
    cache_key = (id(image), rotations)
    with _ocr_pass_lock:
        cached = _ocr_pass_cache.get(cache_key)
        if cached is not None:
            return cached
    _configure_tesseract()
    passes: list[tuple[int, dict[str, Any]]] = []
    for rot in rotations:
        view = image if rot == 0 else image.rotate(rot, expand=True)
        preprocessed = preprocess_for_ocr(view, upscale=True)
        data = pytesseract.image_to_data(
            preprocessed,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",
        )
        passes.append((rot, data))
    with _ocr_pass_lock:
        _ocr_pass_cache[cache_key] = passes
    return passes


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


def _neighbor_ocr_blob(
    data: dict[str, Any], index: int, *, window: int = 3
) -> str:
    n = len(data.get("text") or [])
    parts: list[str] = []
    for j in range(max(0, index - window), min(n, index + window + 1)):
        tok = str(data["text"][j] or "").strip()
        if tok:
            parts.append(tok)
    return " ".join(parts)


_TITLE_BOILERPLATE_RE = re.compile(
    r"(A-?WING|\bWING\b|KEY\s*PLAN|\bEAST\b|\bWEST\b|\bNORTH\b|\bSOUTH\b)",
    re.IGNORECASE,
)
_KEY_TOKEN_RE = re.compile(r"^KEY$", re.IGNORECASE)
_PLAN_TOKEN_RE = re.compile(r"^PLAN$", re.IGNORECASE)
_KEYPLAN_TOKEN_RE = re.compile(r"^KEYPLAN$", re.IGNORECASE)


def keyplan_regions_from_words(
    words: list[tuple[str, float, float, float, float]],
    *,
    pad_x: float = 48.0,
    pad_y: float = 36.0,
    plan_extra_left: float = 260.0,
    plan_extra_top: float = 220.0,
) -> list[BoundingBox]:
    """Build exclusion boxes from OCR tokens that spell KEY PLAN.

    The mini-plan drawing usually sits left/above the label, so the box is
    extended that way. ``words`` are ``(text, x1, y1, x2, y2)``.
    """
    if not words:
        return []
    used: set[int] = set()
    regions: list[BoundingBox] = []

    def _expand(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
        return BoundingBox(
            x1=x1 - pad_x - plan_extra_left,
            y1=y1 - pad_y - plan_extra_top,
            x2=x2 + pad_x,
            y2=y2 + pad_y,
        )

    for i, (text, x1, y1, x2, y2) in enumerate(words):
        if i in used:
            continue
        tok = (text or "").strip()
        if _KEYPLAN_TOKEN_RE.match(tok.replace(" ", "")):
            used.add(i)
            regions.append(_expand(x1, y1, x2, y2))
            continue
        if not _KEY_TOKEN_RE.match(tok):
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        partner: tuple[float, float, float, float] | None = None
        for j, (other, ox1, oy1, ox2, oy2) in enumerate(words):
            if j == i or j in used:
                continue
            if not _PLAN_TOKEN_RE.match((other or "").strip()):
                continue
            ocx = (ox1 + ox2) / 2.0
            ocy = (oy1 + oy2) / 2.0
            if abs(ocy - cy) <= 40.0 and abs(ocx - cx) <= 180.0:
                partner = (ox1, oy1, ox2, oy2)
                used.add(j)
                break
        used.add(i)
        if partner is None:
            regions.append(_expand(x1, y1, x2, y2))
        else:
            px1, py1, px2, py2 = partner
            regions.append(
                _expand(min(x1, px1), min(y1, py1), max(x2, px2), max(y2, py2))
            )
    return regions


_TITLE_TOKEN_RE = re.compile(r"^(A-?WING|WING|EAST|WEST|NORTH|SOUTH)$", re.IGNORECASE)


def title_regions_from_words(
    words: list[tuple[str, float, float, float, float]],
    *,
    pad_x: float = 40.0,
    pad_y: float = 24.0,
    extra_bottom: float = 90.0,
) -> list[BoundingBox]:
    """Exclusion box around sheet titles like ``A-WING (EAST)``."""
    hits = [
        (x1, y1, x2, y2)
        for text, x1, y1, x2, y2 in words
        if _TITLE_TOKEN_RE.match((text or "").strip().replace("(", "").replace(")", ""))
    ]
    if not hits:
        return []
    x1 = min(h[0] for h in hits) - pad_x
    y1 = min(h[1] for h in hits) - pad_y
    x2 = max(h[2] for h in hits) + pad_x
    y2 = max(h[3] for h in hits) + pad_y + extra_bottom
    return [BoundingBox(x1, y1, x2, y2)]


def _ocr_words_in_band(
    image: Image.Image, y0: int, y1: int
) -> list[tuple[str, float, float, float, float]]:
    try:
        import pytesseract
    except ImportError:
        return []
    w, h = image.size
    y0 = max(0, min(h, y0))
    y1 = max(y0 + 1, min(h, y1))
    crop = image.crop((0, y0, w, y1))
    try:
        data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001
        return []
    words: list[tuple[str, float, float, float, float]] = []
    n = len(data.get("text") or [])
    for i in range(n):
        tok = str(data["text"][i] or "").strip()
        if not tok:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0 and conf < 40:
            continue
        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        height = float(data["height"][i])
        words.append((tok, left, top + y0, left + width, top + y0 + height))
    return words


def find_keyplan_regions(image: Image.Image) -> list[BoundingBox]:
    """OCR title + KEY PLAN bands; return exclusion boxes for those insets."""
    if not tesseract_available():
        return []
    _configure_tesseract()
    w, h = image.size
    if w < 20 or h < 20:
        return []
    regions = title_regions_from_words(_ocr_words_in_band(image, 0, int(h * 0.32)))
    regions.extend(
        keyplan_regions_from_words(_ocr_words_in_band(image, int(h * 0.62), h))
    )
    return regions


def _is_title_or_keyplan_ocr_span(
    *,
    text_upper: str,
    height: float,
    neighbor_blob: str,
    max_tag_h: float,
) -> bool:
    """True when span is title-block / KEY PLAN boilerplate, not a plan tag."""
    blob = f"{text_upper} {neighbor_blob}".upper()
    if _TITLE_BOILERPLATE_RE.search(blob):
        return True
    # Lone "E" taken from EAST / A-WING (EAST) is not CONDUIT STUB or a tag.
    if text_upper == "E" and re.search(r"EAST|\bWING\b", blob):
        return True
    # Title letters (e.g. G in A-WING) are much taller than wall tags.
    if re.fullmatch(r"[A-Z]", text_upper) and height > max_tag_h * 0.85:
        return True
    if re.fullmatch(r"[A-Z]{1,3}", text_upper) and height > max_tag_h * 1.1:
        return True
    return False


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

    from pipeline.cv.tag_match import _normalize_numeric_part

    numeric_aliases = {a for a, _ in keys if a.isdigit()}

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
            normalized = _normalize_numeric_part(text, numeric_aliases)
            if normalized:
                text_upper = normalized
            is_part_alias = text_upper in numeric_aliases or bool(
                any(text_upper == alias for alias, _ in keys)
            )
            is_short_tag = bool(re.fullmatch(r"[A-Z]{1,3}", text_upper))
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


def collect_pipe_callout_ocr_hints(
    image: Image.Image,
    *,
    exclude_top_pct: float | None = None,
) -> list[dict[str, Any]]:
    """Collect ``1|n`` style spans from full-image OCR (hints for # reinforcement)."""
    if not tesseract_available():
        return []

    from pipeline.cv.callout_boxes import (
        normalize_callout_text,
        normalize_stacked_callout,
    )

    _configure_tesseract()
    exclude_top_pct = float(
        exclude_top_pct if exclude_top_pct is not None else config.CV_TITLE_BAND_PCT
    )
    orig_w, orig_h = image.size
    y_min = orig_h * exclude_top_pct
    scale = 2.0
    hints: list[dict[str, Any]] = []
    for rot, data in ocr_page_passes(image):
        n = len(data.get("text") or [])
        for i in range(n):
            text = str(data["text"][i] or "").strip()
            if not text:
                continue
            # Merge adjacent token with next if pipe/space split across OCR words.
            nxt = str(data["text"][i + 1] or "").strip() if i + 1 < n else ""
            nxt2 = str(data["text"][i + 2] or "").strip() if i + 2 < n else ""
            candidates = [text]
            if nxt:
                candidates.extend(
                    [
                        f"{text}|{nxt}",
                        f"{text} {nxt}",
                        f"{text}-{nxt}",
                        f"{text}/{nxt}",
                        f"{text}{nxt}",
                    ]
                )
            if nxt and nxt2:
                candidates.append(f"{text}|{nxt}{nxt2}")
                candidates.append(f"{text} {nxt}{nxt2}")

            normalized = None
            for cand in candidates:
                normalized = normalize_callout_text(cand)
                if normalized:
                    break
            if not normalized and text.isdigit() and nxt.isdigit():
                normalized = normalize_stacked_callout(text, nxt)
            if not normalized:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 20:
                continue
            x = float(data["left"][i]) / scale
            y = float(data["top"][i]) / scale
            w = float(data["width"][i]) / scale
            h = float(data["height"][i]) / scale
            x1, y1, x2, y2 = _unrotate_box(
                rot, x, y, x + w, y + h, orig_w=float(orig_w), orig_h=float(orig_h)
            )
            xa, xb = sorted((x1, x2))
            ya, yb = sorted((y1, y2))
            if ya < y_min:
                continue
            hints.append(
                {
                    "text": normalized,
                    "cx": (xa + xb) / 2.0,
                    "cy": (ya + yb) / 2.0,
                    "x1": xa,
                    "y1": ya,
                    "x2": xb,
                    "y2": yb,
                    "qty": 1,
                }
            )
    return hints


def collect_qty_digit_hints(
    image: Image.Image,
    *,
    exclude_top_pct: float | None = None,
) -> list[dict[str, Any]]:
    """Single digits 2–9 from the cached page OCR (QTY beside a '#' drop)."""
    if not tesseract_available():
        return []
    exclude_top_pct = float(
        exclude_top_pct if exclude_top_pct is not None else config.CV_TITLE_BAND_PCT
    )
    orig_w, orig_h = image.size
    y_min = orig_h * exclude_top_pct
    scale = 2.0
    out: list[dict[str, Any]] = []
    for rot, data in ocr_page_passes(image):
        n = len(data.get("text") or [])
        for i in range(n):
            text = str(data["text"][i] or "").strip()
            if not re.fullmatch(r"[2-9]", text):
                continue
            prev = str(data["text"][i - 1] or "").strip() if i else ""
            nxt = str(data["text"][i + 1] or "").strip() if i + 1 < n else ""
            if prev.isdigit() or nxt.isdigit() or prev in {"|", "I", "l"} or nxt in {"|"}:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = 0.0
            if conf >= 0 and conf < 25:
                continue
            x = float(data["left"][i]) / scale
            y = float(data["top"][i]) / scale
            w = float(data["width"][i]) / scale
            h = float(data["height"][i]) / scale
            if h < 6 or h > 28 or w > 22:
                continue
            x1, y1, x2, y2 = _unrotate_box(
                rot, x, y, x + w, y + h, orig_w=float(orig_w), orig_h=float(orig_h)
            )
            xa, xb = sorted((x1, x2))
            ya, yb = sorted((y1, y2))
            if ya < y_min:
                continue
            out.append(
                {
                    "qty": int(text),
                    "cx": (xa + xb) / 2.0,
                    "cy": (ya + yb) / 2.0,
                }
            )
    return out


def _ocr_part_labels_near_centers(
    image: Image.Image,
    keys: list[tuple[str, str]],
    centers: list[tuple[float, float]],
    y_min: float,
) -> list[SymbolDetection]:
    """Digit OCR near drop triangles for raceway part labels (2300/5400/…).

    Labels print horizontally above counters and vertically along walls.
    Uses multiple band offsets per drop and returns a tight box on the digits.
    """
    import cv2
    import pytesseract
    import numpy as np

    from pipeline.cv.tag_match import _normalize_numeric_part

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    numeric_aliases = {a for a, _ in keys if a.isdigit()}
    if not numeric_aliases:
        return []

    out: list[SymbolDetection] = []
    seen_centers: list[tuple[float, float]] = []

    def _seen(cx: float, cy: float, *, radius: float = 45.0) -> bool:
        return any((cx - sx) ** 2 + (cy - sy) ** 2 <= radius**2 for sx, sy in seen_centers)

    def _append_hit(
        matched: str,
        conf: float,
        sx1: int,
        sy1: int,
        sx2: int,
        sy2: int,
        word_box: tuple[float, float, float, float] | None,
    ) -> None:
        if word_box:
            wx1, wy1, wx2, wy2 = word_box
            xa = sx1 + wx1
            ya = sy1 + wy1
            xb = sx1 + wx2
            yb = sy1 + wy2
        else:
            xa, ya, xb, yb = float(sx1), float(sy1), float(sx2), float(sy2)
        cx = (xa + xb) / 2.0
        cy = (ya + yb) / 2.0
        if _seen(cx, cy):
            return
        seen_centers.append((cx, cy))
        out.append(
            SymbolDetection(
                symbol=matched,
                box=BoundingBox(x1=xa, y1=ya, x2=xb, y2=yb),
                score=min(1.0, max(0.55, conf / 100.0 if conf else 0.72)),
                source="ocr",
            )
        )

    for cx, cy in centers:
        if cy < y_min:
            continue
        # Horizontal + vertical bands around each drop (raceway labels sit
        # above/below/beside the triangle, not always in one fixed strip).
        bands: list[tuple[int, int, int, int]] = []
        for dy in (-75, -50, -25, 25, 50):
            bands.append(
                (
                    int(max(0, cx - 110)),
                    int(max(0, cy + dy - 22)),
                    int(min(w, cx + 110)),
                    int(min(h, cy + dy + 22)),
                )
            )
        for dx in (-55, 55):
            bands.append(
                (
                    int(max(0, cx + dx - 22)),
                    int(max(0, cy - 90)),
                    int(min(w, cx + dx + 22)),
                    int(min(h, cy + 90)),
                )
            )

        for sx1, sy1, sx2, sy2 in bands:
            if sx2 - sx1 < 12 or sy2 - sy1 < 10:
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
                    view, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC
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
                        if not text:
                            continue
                        normalized = _normalize_numeric_part(text, numeric_aliases)
                        if not normalized:
                            continue
                        try:
                            conf = float(data["conf"][i])
                        except (TypeError, ValueError):
                            conf = 0.0
                        if conf < 25 and conf != 0:
                            continue
                        matched = match_span_to_text_key(normalized, keys, size=14)
                        if not matched:
                            continue
                        scale = 4.0
                        wx1 = float(data["left"][i]) / scale
                        wy1 = float(data["top"][i]) / scale
                        wx2 = wx1 + float(data["width"][i]) / scale
                        wy2 = wy1 + float(data["height"][i]) / scale
                        _append_hit(
                            matched,
                            conf,
                            sx1,
                            sy1,
                            sx2,
                            sy2,
                            (wx1, wy1, wx2, wy2),
                        )
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
    return out


def ap_skip_centers_from_ocr(detections: list[SymbolDetection]) -> list[tuple[float, float]]:
    """Centers where triangle CV should not fire (AP/WP device glyphs)."""
    centers: list[tuple[float, float]] = []
    for det in detections:
        sym = (det.symbol or "").strip().upper()
        if not sym:
            continue
        is_ap = sym == "AP" or "ACCESS POINT" in sym or sym.startswith("TAP")
        is_wp = sym == "WP" or sym.startswith("WEATHERPROOF")
        if not is_ap and not is_wp:
            continue
        cx = (det.box.x1 + det.box.x2) / 2.0
        cy = (det.box.y1 + det.box.y2) / 2.0
        bh = max(1.0, det.box.y2 - det.box.y1)
        if is_ap:
            # Plan text "AP" sits below the circle+triangle glyph.
            for lift in (bh * 0.9, bh * 1.4, 48.0):
                centers.append((cx, cy - lift))
        else:
            centers.append((cx, cy - bh * 0.5))
        if det.source == "template":
            bw = max(1.0, det.box.x2 - det.box.x1)
            centers.append((cx, cy - bh * 0.35))
            centers.append((cx - bw * 0.15, cy - bh * 0.2))
            centers.append((cx + bw * 0.15, cy - bh * 0.2))
    return centers


def clock_combo_skip_centers(
    image: Image.Image,
    *,
    exclude_top_pct: float | None = None,
) -> list[tuple[float, float]]:
    """Centers near IP clock/speaker combo assemblies (triangle + 12:00 + horn)."""
    if not tesseract_available():
        return []

    import pytesseract

    _configure_tesseract()
    exclude_top_pct = float(
        exclude_top_pct if exclude_top_pct is not None else config.CV_TITLE_BAND_PCT
    )
    orig_w, orig_h = image.size
    y_min = orig_h * exclude_top_pct
    scale = 2.0
    centers: list[tuple[float, float]] = []

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
            text = str(data["text"][i] or "").strip().upper()
            if not text:
                continue
            compact = re.sub(r"[^0-9:]", "", text)
            if compact not in {"12:00", "1200", "12.00"}:
                continue
            x = float(data["left"][i]) / scale
            y = float(data["top"][i]) / scale
            w = float(data["width"][i]) / scale
            h = float(data["height"][i]) / scale
            x1, y1, x2, y2 = _unrotate_box(
                rot, x, y, x + w, y + h, orig_w=float(orig_w), orig_h=float(orig_h)
            )
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if cy < y_min:
                continue
            # Triangle/horn sit beside the clock label on these combos.
            for ox, oy in (
                (0.0, 0.0),
                (-55.0, 0.0),
                (55.0, 0.0),
                (-80.0, 0.0),
                (80.0, 0.0),
                (0.0, -35.0),
                (0.0, 35.0),
            ):
                centers.append((cx + ox, cy + oy))
    return centers


def _bbox_from_rel(
    bbox_rel: dict[str, Any] | None,
    *,
    img_w: float,
    img_h: float,
) -> BoundingBox | None:
    if not isinstance(bbox_rel, dict):
        return None
    try:
        x1 = float(bbox_rel["x1"]) * img_w
        y1 = float(bbox_rel["y1"]) * img_h
        x2 = float(bbox_rel["x2"]) * img_w
        y2 = float(bbox_rel["y2"]) * img_h
    except (KeyError, TypeError, ValueError):
        return None
    xa, xb = sorted((max(0.0, x1), max(0.0, x2)))
    ya, yb = sorted((max(0.0, y1), max(0.0, y2)))
    xb = min(img_w, xb)
    yb = min(img_h, yb)
    if xb - xa < 2 or yb - ya < 2:
        return None
    return BoundingBox(x1=xa, y1=ya, x2=xb, y2=yb)


def _vision_entity_texts(parsed: dict[str, Any]) -> list[tuple[str, float, BoundingBox | None]]:
    """Flatten vision OCR response into (text, confidence, optional box)."""
    img_w = float(parsed.get("_img_w") or 0)
    img_h = float(parsed.get("_img_h") or 0)
    out: list[tuple[str, float, BoundingBox | None]] = []

    entities = parsed.get("text_entities")
    if isinstance(entities, list):
        for item in entities:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = float(item.get("confidence", 0.75))
            except (TypeError, ValueError):
                conf = 0.75
            box = _bbox_from_rel(item.get("bbox_rel"), img_w=img_w, img_h=img_h)
            out.append((text, conf, box))

    grouped = parsed.get("parsed_entities")
    if isinstance(grouped, dict):
        for values in grouped.values():
            if not isinstance(values, list):
                continue
            for raw in values:
                text = str(raw or "").strip()
                if text:
                    out.append((text, 0.7, None))
    return out


def detect_tags_from_vision_ocr(
    image: Image.Image,
    legend: SymbolTableInfo,
    *,
    expected_context: str | None = None,
    exclude_top_pct: float | None = None,
) -> list[SymbolDetection]:
    """Detect legend tags via vision-model OCR (architect prompt)."""
    from pipeline import config
    from pipeline.llm_verify import parse_crop_text_with_llm

    if not config.vision_ocr_enabled():
        return []

    keys = text_legend_keys(legend)
    if not keys:
        return []

    exclude_top_pct = float(
        exclude_top_pct if exclude_top_pct is not None else config.CV_TITLE_BAND_PCT
    )
    orig_w, orig_h = image.size
    y_min = orig_h * exclude_top_pct
    min_h = float(config.CV_OCR_MIN_HEIGHT)
    max_h = float(config.CV_OCR_MAX_HEIGHT) * 1.5

    parsed = parse_crop_text_with_llm(
        image,
        expected_context=expected_context or config.VISION_OCR_EXPECTED_CONTEXT,
    )
    if not parsed.get("ok"):
        logger.info("Vision OCR skipped: %s", parsed.get("error"))
        return []

    parsed["_img_w"] = orig_w
    parsed["_img_h"] = orig_h
    from pipeline.cv.tag_match import _normalize_numeric_part

    numeric_aliases = {a for a, _ in keys if a.isdigit()}
    out: list[SymbolDetection] = []
    seen: set[tuple[str, int, int]] = set()

    for text, conf, box in _vision_entity_texts(parsed):
        text_upper = text.upper().strip("-–—_/ ")
        normalized = _normalize_numeric_part(text, numeric_aliases)
        if normalized:
            text_upper = normalized
        glyph_size = 12.0
        if box is not None:
            glyph_size = min(box.x2 - box.x1, box.y2 - box.y1)
        matched = match_span_to_text_key(text_upper, keys, size=glyph_size)
        if not matched:
            matched = match_span_to_text_key(text, keys, size=glyph_size)
        if not matched:
            continue
        if box is None:
            continue
        if box.y1 < y_min:
            continue
        if glyph_size < min_h or glyph_size > max_h:
            continue
        key = (
            matched,
            int((box.x1 + box.x2) / 2),
            int((box.y1 + box.y2) / 2),
        )
        if key in seen:
            continue
        seen.add(key)
        score = max(0.55, min(1.0, conf))
        out.append(
            SymbolDetection(
                symbol=matched,
                box=box,
                score=score,
                source="vision_ocr",
            )
        )
    return out
