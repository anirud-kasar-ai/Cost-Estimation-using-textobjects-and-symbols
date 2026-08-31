from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class SymbolDetection:
    symbol: str
    box: BoundingBox
    score: float
    source: str = "cv"
    qty: int = 1  # per-mark quantity multiplier (drawing convention: # = QTY)
    tile_file: str | None = None  # set for folder jobs (source zoom tile)
    classify_source: str | None = None  # context_rule | vision_classify | cv_fallback


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_cross_label(
    detections: list[SymbolDetection],
    iou_threshold: float = 0.6,
    source: str = "template",
) -> list[SymbolDetection]:
    """Drop lower-scored detections of *different* labels on the same spot.

    Near-identical legend glyphs (camera family) match the same plan symbol
    under several labels; without this, one symbol counts once per label.
    """
    targeted = [d for d in detections if d.source == source]
    others = [d for d in detections if d.source != source]
    ordered = sorted(targeted, key=lambda d: d.score, reverse=True)
    kept: list[SymbolDetection] = []
    for det in ordered:
        if any(box_iou(det.box, k.box) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    return others + kept


def suppress_nearby_same_label(
    detections: list[SymbolDetection],
    *,
    source: str = "ocr",
    max_center_dist: float = 90.0,
) -> list[SymbolDetection]:
    """Keep one OCR hit per label when several boxes sit on the same text run."""
    targeted = [d for d in detections if d.source == source]
    others = [d for d in detections if d.source != source]
    ordered = sorted(targeted, key=lambda d: d.score, reverse=True)
    kept: list[SymbolDetection] = []
    for det in ordered:
        dcx = (det.box.x1 + det.box.x2) / 2.0
        dcy = (det.box.y1 + det.box.y2) / 2.0
        duplicate = False
        for other in kept:
            if other.symbol != det.symbol:
                continue
            ocx = (other.box.x1 + other.box.x2) / 2.0
            ocy = (other.box.y1 + other.box.y2) / 2.0
            if ((dcx - ocx) ** 2 + (dcy - ocy) ** 2) ** 0.5 <= max_center_dist:
                # Same text run / double-read — not two separate plan labels.
                if box_iou(det.box, other.box) >= 0.12:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)
    return others + kept


def suppress_cross_source_duplicates(
    detections: list[SymbolDetection],
    *,
    sources: set[str] | None = None,
    max_center_dist: float = 36.0,
    iou_threshold: float = 0.15,
) -> list[SymbolDetection]:
    """One physical symbol → one count when CV OCR / vision OCR / template overlap.

    Dense zoom tiles often fire the same AP/J/2300 from Tesseract and Gemini
    (or template). Keep the highest-scoring hit per label near the same center.
    """
    sources = sources or {"ocr", "vision_ocr", "template"}
    targeted = [d for d in detections if d.source in sources]
    others = [d for d in detections if d.source not in sources]
    ordered = sorted(
        targeted,
        key=lambda d: (float(d.score), max(1, int(d.qty))),
        reverse=True,
    )
    kept: list[SymbolDetection] = []
    for det in ordered:
        dcx = (det.box.x1 + det.box.x2) / 2.0
        dcy = (det.box.y1 + det.box.y2) / 2.0
        duplicate = False
        for other in kept:
            if other.symbol != det.symbol:
                continue
            ocx = (other.box.x1 + other.box.x2) / 2.0
            ocy = (other.box.y1 + other.box.y2) / 2.0
            dist = ((dcx - ocx) ** 2 + (dcy - ocy) ** 2) ** 0.5
            if dist <= max_center_dist or box_iou(det.box, other.box) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return others + kept


def suppress_hash_on_devices(
    detections: list[SymbolDetection],
    *,
    max_center_dist: float = 32.0,
    iou_threshold: float = 0.12,
) -> list[SymbolDetection]:
    """Drop '#' / triangle hits that sit on a camera hourglass or data-pole bowtie.

    Flanking jack triangles beside a DATA POLE stay; only marks whose center
    is inside the device box (or that overlap it) are the glyph itself.
    """
    devices = [d for d in detections if d.source in {"hourglass", "bowtie"}]
    if not devices:
        return detections
    _ = max_center_dist
    kept: list[SymbolDetection] = []
    for det in detections:
        is_hash = det.source == "triangle" or det.symbol == "#"
        if not is_hash:
            kept.append(det)
            continue
        dcx, dcy = _center(det.box)
        covered = False
        for other in devices:
            pad = 6.0
            inside = (
                other.box.x1 - pad <= dcx <= other.box.x2 + pad
                and other.box.y1 - pad <= dcy <= other.box.y2 + pad
            )
            if inside or box_iou(det.box, other.box) >= iou_threshold:
                covered = True
                break
        if not covered:
            kept.append(det)
    return kept


def suppress_overlapping_devices(
    detections: list[SymbolDetection],
    *,
    max_center_dist: float = 28.0,
    iou_threshold: float = 0.12,
) -> list[SymbolDetection]:
    """One glyph cannot be both a camera (hourglass) and a data pole (bowtie).

    A square with an X is a DATA POLE. Prefer bowtie when the two overlap.
    """
    devices = [d for d in detections if d.source in {"hourglass", "bowtie"}]
    others = [d for d in detections if d.source not in {"hourglass", "bowtie"}]
    if len(devices) < 2:
        return detections
    ordered = sorted(
        devices,
        key=lambda d: (0 if d.source == "bowtie" else 1, -float(d.score)),
    )
    kept: list[SymbolDetection] = []
    for det in ordered:
        dcx, dcy = _center(det.box)
        duplicate = False
        for other in kept:
            ocx, ocy = _center(other.box)
            dist = ((dcx - ocx) ** 2 + (dcy - ocy) ** 2) ** 0.5
            if dist <= max_center_dist or box_iou(det.box, other.box) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return others + kept


def suppress_stub_on_devices(
    detections: list[SymbolDetection],
    *,
    max_center_dist: float = 28.0,
    iou_threshold: float = 0.12,
) -> list[SymbolDetection]:
    """Drop conduit-stub hits that sit on a drop triangle, camera, or pole."""
    devices = [
        d
        for d in detections
        if d.source in {"triangle", "hourglass", "bowtie", "callout_drop"}
    ]
    if not devices:
        return detections
    kept: list[SymbolDetection] = []
    for det in detections:
        if det.source != "stub":
            kept.append(det)
            continue
        dcx, dcy = _center(det.box)
        covered = False
        for other in devices:
            ocx, ocy = _center(other.box)
            dist = ((dcx - ocx) ** 2 + (dcy - ocy) ** 2) ** 0.5
            if dist <= max_center_dist or box_iou(det.box, other.box) >= iou_threshold:
                covered = True
                break
        if not covered:
            kept.append(det)
    return kept


def suppress_in_boxes(
    detections: list[SymbolDetection],
    regions: list[BoundingBox],
) -> list[SymbolDetection]:
    """Drop detections whose center sits inside a boilerplate region (KEY PLAN)."""
    if not regions:
        return detections
    kept: list[SymbolDetection] = []
    for det in detections:
        cx, cy = _center(det.box)
        inside = False
        for box in regions:
            if box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2:
                inside = True
                break
        if not inside:
            kept.append(det)
    return kept


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def nms_by_class(
    detections: list[SymbolDetection],
    iou_threshold: float,
    *,
    center_dist: float | None = None,
) -> list[SymbolDetection]:
    """Greedy class-wise NMS. On score ties prefer the higher qty (``# = QTY``).

    ``center_dist`` also suppresses same-label marks whose centers sit within
    that many ROI pixels — needed for small drop triangles that overlap across
    zoom tiles but have IoU well below 0.5.
    """
    iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
    by_label: dict[str, list[SymbolDetection]] = {}
    for det in detections:
        by_label.setdefault(det.symbol, []).append(det)

    kept: list[SymbolDetection] = []
    for group in by_label.values():
        ordered = sorted(
            group,
            key=lambda d: (float(d.score), max(1, int(d.qty))),
            reverse=True,
        )
        suppressed = [False] * len(ordered)
        for i, det in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(det)
            dcx, dcy = _center(det.box)
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                other = ordered[j]
                if box_iou(det.box, other.box) >= iou_threshold:
                    suppressed[j] = True
                    continue
                if center_dist is not None and center_dist > 0:
                    ocx, ocy = _center(other.box)
                    if ((dcx - ocx) ** 2 + (dcy - ocy) ** 2) ** 0.5 <= center_dist:
                        suppressed[j] = True
    return kept
