"""Detect ONLY digits drawn inside callout circles (balloon tags).

Target pattern (from user examples):
  - thin black circular outline
  - a short integer (1–99) centered inside
  - often a leader line attached to the outside

Free-floating dimension text, sheet numbers, and other digits are ignored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.legend_ocr import configure_tesseract, tesseract_available

logger = logging.getLogger(__name__)

_DIGIT_RE = re.compile(r"^\d{1,2}$")  # callout balloons are 1–99 (1–2 digits)


@dataclass(frozen=True)
class CalloutHit:
    number: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str  # hough_ring | contour_ring
    image_name: str = ""

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cx"] = self.cx
        d["cy"] = self.cy
        return d


def _gray(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask where True = dark ink."""
    import cv2

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return inv > 0


def is_hollow_callout_circle(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """True only for a thin ring with mostly empty interior (digit may sit inside).

    Rejects filled blobs, north arrows, dimension bubbles that are not rings,
    and random circular noise without a clear outline.
    """
    h, w = gray.shape[:2]
    if radius < config.CALLOUT_MIN_RADIUS_PX or radius > config.CALLOUT_MAX_RADIUS_PX:
        return False

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Annulus covering the expected stroke
    stroke = max(1.2, radius * 0.18)
    ring = (dist >= radius - stroke) & (dist <= radius + stroke * 0.85)
    core = dist <= max(2.0, radius * 0.55)
    # Outside just beyond the ring — should be mostly empty (except leader)
    halo = (dist > radius + stroke * 0.9) & (dist < radius + stroke * 2.2)

    if int(ring.sum()) < 12 or int(core.sum()) < 8:
        return False

    ink = _ink_mask(gray)
    ring_fill = float(ink[ring].mean())
    core_fill = float(ink[core].mean())
    halo_fill = float(ink[halo].mean()) if int(halo.sum()) > 0 else 0.0

    # Ring must have substantial ink (the circle outline)
    if ring_fill < 0.18:
        return False
    # Interior must NOT be a solid fill (north arrow / filled glyph)
    if core_fill > 0.55:
        return False
    # Interior should have some ink for the digit, but not required at this stage
    # (OCR decides). Still reject nearly-empty rings that are just noise.
    # Outside halo should be sparse — one leader line is OK, dense text is not.
    if halo_fill > 0.35:
        return False
    # Ring denser than core (outline vs inside)
    if ring_fill < core_fill + 0.05:
        return False
    return True


def _looks_like_digit_one(crop_gray: np.ndarray) -> bool:
    """Very thin vertical stroke only — used to fix 4→1 OCR mistakes."""
    import cv2

    h, w = crop_gray.shape[:2]
    if h < 4 or w < 3:
        return False
    _, inv = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(inv > 0)
    if len(xs) < 4:
        return False
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    if bh < 1:
        return False
    aspect = bw / float(bh)
    fill = float(len(xs)) / float(max(1, bw * bh))
    return aspect < 0.38 and fill < 0.45


def _pick_best_digit_string(
    candidates: list[tuple[str, float]],
    *,
    min_conf: float | None = None,
) -> tuple[str, float]:
    """Prefer longer callout numbers (14 over 4/1) then higher confidence."""
    floor = config.CALLOUT_OCR_MIN_CONF if min_conf is None else min_conf
    plausible = [
        (t, c) for t, c in candidates if _DIGIT_RE.match(t) and c >= floor
    ]
    if not plausible:
        # Fall back without conf floor only if nothing qualified
        plausible = [(t, c) for t, c in candidates if _DIGIT_RE.match(t)]
    if not plausible:
        return "", 0.0
    # Longer digit strings win only when they meet the confidence floor;
    # otherwise fall back to highest-confidence short read.
    plausible.sort(
        key=lambda tc: (len(tc[0]) if tc[1] >= floor else 0, tc[1]),
        reverse=True,
    )
    return plausible[0][0], max(0.0, plausible[0][1])


def _left_half_has_digit_one(crop_gray: np.ndarray) -> bool:
    """True if the left half of a circle crop looks like a thin '1' stem.

    Used to recover '14' when OCR only returns '4' because the leading 1 is faint.
    """
    h, w = crop_gray.shape[:2]
    if w < 8 or h < 8:
        return False
    left = crop_gray[:, : max(3, w // 2)]
    return _looks_like_digit_one(left)


def _prepare_ocr_binary(crop_gray: np.ndarray) -> np.ndarray | None:
    """Upscale + Otsu binary (light ink on dark inverted to dark-on-light)."""
    import cv2

    h, w = crop_gray.shape[:2]
    if h < 3 or w < 3 or float(np.std(crop_gray)) < 4.0:
        return None
    scale = max(3.0, 56.0 / max(1, min(h, w)))
    scaled = cv2.resize(crop_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad = max(2, int(0.12 * min(scaled.shape[:2])))
    scaled = cv2.copyMakeBorder(
        scaled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.median(scaled))
    )
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(binary)
    return binary


def _tesseract_read_digits(view: np.ndarray, *, psm: int) -> str:
    """Fast digit read via image_to_string (much cheaper than image_to_data)."""
    import pytesseract

    raw = pytesseract.image_to_string(
        view,
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
    )
    return re.sub(r"\D", "", (raw or "").strip())[:2]


def _tesseract_digit_candidates(crop_gray: np.ndarray) -> list[tuple[str, float]]:
    """Fast digit OCR — usually a single PSM-8 call; escalate only if empty."""
    import cv2

    configure_tesseract()
    binary = _prepare_ocr_binary(crop_gray)
    if binary is None:
        return []

    candidates: list[tuple[str, float]] = []

    def _add(view: np.ndarray, psm: int) -> str:
        text = _tesseract_read_digits(view, psm=psm)
        if _DIGIT_RE.match(text):
            conf = 75.0 if len(text) >= 2 else 65.0
            candidates.append((text, conf))
        return text

    # Happy path: one Tesseract call (do NOT use PSM 10 — drops 14/15)
    text = _add(binary, 8)
    if _DIGIT_RE.match(text):
        return candidates

    text = _add(cv2.bitwise_not(binary), 8)
    if _DIGIT_RE.match(text):
        return candidates

    _add(binary, 7)
    return candidates


def _recover_two_digit_callout(
    best_text: str,
    best_conf: float,
    crop_gray: np.ndarray,
) -> tuple[str, float]:
    """Fix common two-digit failures without extra Tesseract calls."""
    if best_text in {"14", "15"}:
        return best_text, max(best_conf, 40.0)
    if best_text == "45":
        return "15", max(best_conf * 0.9, 50.0)
    if best_text == "44":
        return "14", max(best_conf, 55.0)
    if best_text == "4" and _left_half_has_digit_one(crop_gray):
        return "14", max(best_conf, 55.0)
    if best_text == "4" and _looks_like_digit_one(crop_gray):
        return "1", max(best_conf, 55.0)
    return best_text, best_conf


def _ocr_digits_in_crop(crop_gray: np.ndarray) -> tuple[str, float]:
    """OCR digits inside a callout circle. Supports 1–2 digit numbers (e.g. 14, 15)."""
    if crop_gray.size == 0:
        return "", 0.0
    if not tesseract_available():
        return "", 0.0

    candidates = _tesseract_digit_candidates(crop_gray)
    best_text, best_conf = _pick_best_digit_string(candidates)
    if not _DIGIT_RE.match(best_text):
        return "", 0.0

    best_text, best_conf = _recover_two_digit_callout(best_text, best_conf, crop_gray)
    if not _DIGIT_RE.match(best_text):
        return "", 0.0
    if best_conf < 1.0 and best_text:
        best_conf = 40.0
    return best_text, max(0.0, best_conf)


def _hit_from_circle(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    source: str,
) -> CalloutHit | None:
    if not is_hollow_callout_circle(gray, cx, cy, radius):
        return None

    h, w = gray.shape[:2]

    def _try_inset(inset_frac: float) -> tuple[str, float] | None:
        inset = max(0.8, radius * inset_frac)
        x1 = int(max(0, cx - radius + inset))
        y1 = int(max(0, cy - radius + inset))
        x2 = int(min(w, cx + radius - inset))
        y2 = int(min(h, cy + radius - inset))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        text, conf = _ocr_digits_in_crop(gray[y1:y2, x1:x2])
        if not _DIGIT_RE.match(text) or conf < config.CALLOUT_OCR_MIN_CONF:
            return None
        return text, conf

    # One crop inset — avoid a second full OCR pass unless we got nothing.
    result = _try_inset(0.16)
    if result is None:
        result = _try_inset(0.08)
    if result is None:
        return None
    text, conf = result
    number = int(text)
    if number < 1 or number > 99:
        return None

    return CalloutHit(
        number=number,
        x1=float(cx - radius),
        y1=float(cy - radius),
        x2=float(cx + radius),
        y2=float(cy + radius),
        confidence=float(conf),
        source=source,
    )


def detect_hough_callouts(image: Image.Image) -> list[CalloutHit]:
    """Hough circles → keep only hollow rings → OCR digit inside."""
    import cv2

    gray = _gray(image)
    h, w = gray.shape[:2]
    blur = cv2.medianBlur(gray, 5)

    min_r = max(5, config.CALLOUT_MIN_RADIUS_PX)
    # Allow slightly larger rings for two-digit callouts (14, 15)
    max_r = min(config.CALLOUT_MAX_RADIUS_PX, max(18, int(0.035 * min(h, w))))
    min_dist = max(12, int(min_r * 2.0))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=14,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=70,
            param2=11,
            minRadius=min_r,
            maxRadius=max_r,
        )
    if circles is None:
        return []

    raw = np.round(circles[0]).astype(int)
    # Prefer mid-size balloons (~10–12 px); two-digit rings trend a bit larger
    order = np.argsort(np.abs(raw[:, 2].astype(float) - 11.0))
    raw = raw[order][:40]

    # Filter hollow first, then OCR only the best mid-size rings
    hollow: list[tuple[int, int, int]] = []
    for x, y, r in raw:
        if is_hollow_callout_circle(gray, float(x), float(y), float(r)):
            hollow.append((int(x), int(y), int(r)))
    hollow.sort(key=lambda t: abs(t[2] - 11))
    hollow = hollow[:25]

    def _ocr_one(xyr: tuple[int, int, int]) -> CalloutHit | None:
        x, y, r = xyr
        return _hit_from_circle(gray, float(x), float(y), float(r), source="hough_ring")

    from concurrent.futures import ThreadPoolExecutor

    hits: list[CalloutHit] = []
    seen: list[tuple[float, float]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for hit in pool.map(_ocr_one, hollow):
            if hit is None:
                continue
            if any((hit.cx - sx) ** 2 + (hit.cy - sy) ** 2 < 100.0 for sx, sy in seen):
                continue
            hits.append(hit)
            seen.append((hit.cx, hit.cy))
    return hits


def _circularity(contour: np.ndarray) -> float:
    import cv2

    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def detect_contour_callouts(
    image: Image.Image,
    *,
    skip_centers: list[tuple[float, float]] | None = None,
) -> list[CalloutHit]:
    """Circular contours that look like hollow rings → OCR digit inside."""
    import cv2

    gray = _gray(image)
    ink = _ink_mask(gray).astype(np.uint8) * 255
    # Dilate slightly so thin rings form a closed contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_r = config.CALLOUT_MIN_RADIUS_PX
    max_r = min(config.CALLOUT_MAX_RADIUS_PX, 30)
    skip = list(skip_centers or [])
    min_dist2 = float(config.CALLOUT_DEDUP_DIST_PX**2)

    candidates: list[tuple[float, float, float, float]] = []  # circ, cx, cy, r
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (np.pi * min_r * min_r * 0.4) or area > (np.pi * max_r * max_r * 1.6):
            continue
        circ = _circularity(contour)
        if circ < 0.65:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_r or radius > max_r:
            continue
        circle_area = np.pi * radius * radius
        if area / max(1.0, circle_area) < 0.55:
            continue
        if any((float(cx) - sx) ** 2 + (float(cy) - sy) ** 2 < min_dist2 for sx, sy in skip):
            continue
        candidates.append((circ, float(cx), float(cy), float(radius)))

    # OCR only the roundest candidates (contour lists can be huge)
    candidates.sort(key=lambda t: (-t[0], abs(t[3] - 11.0)))
    candidates = candidates[:25]

    hits: list[CalloutHit] = []
    seen: list[tuple[float, float]] = list(skip)

    def _ocr_one(item: tuple[float, float, float, float]) -> CalloutHit | None:
        _circ, cx, cy, radius = item
        return _hit_from_circle(gray, cx, cy, radius, source="contour_ring")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        for hit in pool.map(_ocr_one, candidates):
            if hit is None:
                continue
            if any((hit.cx - sx) ** 2 + (hit.cy - sy) ** 2 < 100.0 for sx, sy in seen):
                continue
            hits.append(hit)
            seen.append((hit.cx, hit.cy))
    return hits


def _iou(a: CalloutHit, b: CalloutHit) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / (area_a + area_b - inter)


def nms_callouts(hits: list[CalloutHit]) -> list[CalloutHit]:
    if not hits:
        return []
    # Prefer two-digit recoveries (14/15) over orphan single digits (4/1/5)
    ranked = sorted(
        hits,
        key=lambda h: (
            -len(str(h.number)),
            -h.confidence,
            -(h.x2 - h.x1) * (h.y2 - h.y1),
        ),
    )
    kept: list[CalloutHit] = []
    for hit in ranked:
        drop = False
        for other in kept:
            dist = ((hit.cx - other.cx) ** 2 + (hit.cy - other.cy) ** 2) ** 0.5
            if _iou(hit, other) >= config.CALLOUT_NMS_IOU:
                drop = True
                break
            if dist < config.CALLOUT_DEDUP_DIST_PX:
                drop = True
                break
        if not drop:
            kept.append(hit)
    return kept


def detect_callouts(image: Image.Image, *, image_name: str = "") -> list[CalloutHit]:
    """Detect circled callout numbers only — never free-floating digits."""
    if not tesseract_available():
        raise RuntimeError(
            "Tesseract OCR is not available. Install Tesseract and/or set TESSERACT_CMD."
        )
    hough = detect_hough_callouts(image)
    contour = detect_contour_callouts(
        image,
        skip_centers=[(h.cx, h.cy) for h in hough],
    )
    merged = nms_callouts([*hough, *contour])
    if image_name:
        merged = [
            CalloutHit(
                number=h.number,
                x1=h.x1,
                y1=h.y1,
                x2=h.x2,
                y2=h.y2,
                confidence=h.confidence,
                source=h.source,
                image_name=image_name,
            )
            for h in merged
        ]
    logger.info(
        "Circled callouts on %s: hough=%d contour=%d → nms=%d",
        image_name or "image",
        len(hough),
        len(contour),
        len(merged),
    )
    return merged
