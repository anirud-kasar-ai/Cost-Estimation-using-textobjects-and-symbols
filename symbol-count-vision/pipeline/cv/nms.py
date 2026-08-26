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


def _expand_box(box: BoundingBox, *, pad_ratio: float = 0.75) -> BoundingBox:
    bw = max(1.0, box.x2 - box.x1)
    bh = max(1.0, box.y2 - box.y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    return BoundingBox(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y)


def _center_in_box(cx: float, cy: float, box: BoundingBox) -> bool:
    return box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2


def suppress_hourglass_on_triangles(
    detections: list[SymbolDetection],
    *,
    iou_threshold: float = 0.12,
    max_center_dist: float = 48.0,
    triangle_min_score: float = 0.75,
) -> list[SymbolDetection]:
    """Drop hourglass/camera hits on or beside a high-confidence drop triangle."""
    triangles = [
        d for d in detections if d.source == "triangle" and float(d.score) >= triangle_min_score
    ]
    if not triangles:
        return detections
    kept: list[SymbolDetection] = []
    for det in detections:
        if det.source != "hourglass":
            kept.append(det)
            continue
        ccx = (det.box.x1 + det.box.x2) / 2.0
        ccy = (det.box.y1 + det.box.y2) / 2.0
        drop = False
        for tri in triangles:
            if box_iou(det.box, tri.box) >= iou_threshold:
                drop = True
                break
            expanded = _expand_box(tri.box, pad_ratio=0.85)
            if _center_in_box(ccx, ccy, expanded):
                drop = True
                break
            tcx = (tri.box.x1 + tri.box.x2) / 2.0
            tcy = (tri.box.y1 + tri.box.y2) / 2.0
            if ((ccx - tcx) ** 2 + (ccy - tcy) ** 2) ** 0.5 <= max_center_dist:
                drop = True
                break
        if not drop:
            kept.append(det)
    return kept


def suppress_triangles_near_tags(
    detections: list[SymbolDetection],
    *,
    tag_symbols: set[str] | None = None,
    max_center_dist: float = 70.0,
    template_pad: float = 40.0,
) -> list[SymbolDetection]:
    """Drop drop-triangle hits sitting on AP/WP and other tagged device glyphs."""
    tag_symbols = tag_symbols or {"AP", "WP"}
    tag_symbols = {s.upper() for s in tag_symbols}
    anchors: list[tuple[float, float, float]] = []
    for det in detections:
        sym = (det.symbol or "").strip().upper()
        if sym not in tag_symbols:
            continue
        cx = (det.box.x1 + det.box.x2) / 2.0
        cy = (det.box.y1 + det.box.y2) / 2.0
        radius = max_center_dist
        bw = max(1.0, det.box.x2 - det.box.x1)
        bh = max(1.0, det.box.y2 - det.box.y1)
        if det.source == "template":
            radius = max(max_center_dist, max(bw, bh) * 0.85 + template_pad)
            if sym == "AP":
                cy -= bh * 0.45
        elif sym == "AP":
            cy -= max(45.0, bh * 1.1)
        anchors.append((cx, cy, radius))

    if not anchors:
        return detections
    kept: list[SymbolDetection] = []
    for det in detections:
        if det.source != "triangle":
            kept.append(det)
            continue
        tcx = (det.box.x1 + det.box.x2) / 2.0
        tcy = (det.box.y1 + det.box.y2) / 2.0
        if any((tcx - ax) ** 2 + (tcy - ay) ** 2 <= r * r for ax, ay, r in anchors):
            continue
        kept.append(det)
    return kept


def suppress_triangles_near_templates(
    detections: list[SymbolDetection],
    *,
    min_center_dist: float = 75.0,
    pad_ratio: float = 1.15,
) -> list[SymbolDetection]:
    """Drop # triangles sitting on any template-matched device glyph."""
    anchors: list[tuple[float, float, float]] = []
    for det in detections:
        if det.source != "template":
            continue
        cx = (det.box.x1 + det.box.x2) / 2.0
        cy = (det.box.y1 + det.box.y2) / 2.0
        bw = max(1.0, det.box.x2 - det.box.x1)
        bh = max(1.0, det.box.y2 - det.box.y1)
        radius = max(min_center_dist, max(bw, bh) * pad_ratio + 20.0)
        anchors.append((cx, cy, radius))

    if not anchors:
        return detections
    kept: list[SymbolDetection] = []
    for det in detections:
        if det.source != "triangle":
            kept.append(det)
            continue
        tcx = (det.box.x1 + det.box.x2) / 2.0
        tcy = (det.box.y1 + det.box.y2) / 2.0
        if any((tcx - ax) ** 2 + (tcy - ay) ** 2 <= r * r for ax, ay, r in anchors):
            continue
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
