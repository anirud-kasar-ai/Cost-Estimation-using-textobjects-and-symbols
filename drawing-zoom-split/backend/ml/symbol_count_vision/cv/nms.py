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
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return others + kept


def nms_by_class(detections: list[SymbolDetection], iou_threshold: float) -> list[SymbolDetection]:
    """Greedy class-wise NMS. On score ties prefer the higher qty (``# = QTY``)."""
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
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                if box_iou(det.box, ordered[j].box) >= iou_threshold:
                    suppressed[j] = True
    return kept
