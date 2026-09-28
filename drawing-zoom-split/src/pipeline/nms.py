"""NMS / overlap de-dup for YOLO boxes (same tile and across overlapping tiles)."""

from __future__ import annotations

from pipeline.yolo_detect import YoloHit


def box_iou(a: YoloHit, b: YoloHit) -> float:
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


def _center(hit: YoloHit) -> tuple[float, float]:
    return ((hit.x1 + hit.x2) / 2.0, (hit.y1 + hit.y2) / 2.0)


def _is_compact(hit: YoloHit) -> bool:
    w = max(1.0, hit.x2 - hit.x1)
    h = max(1.0, hit.y2 - hit.y1)
    return (max(w, h) / min(w, h)) < 2.2


def _same_label(a: YoloHit, b: YoloHit) -> bool:
    if a.class_id == b.class_id:
        return True
    return (a.class_name or "").strip().upper() == (b.class_name or "").strip().upper()


def nms_yolo_hits(
    hits: list[YoloHit],
    *,
    same_class_iou: float = 0.35,
    cross_class_iou: float = 0.50,
    compact_center_px: float = 8.0,
) -> list[YoloHit]:
    """Keep one box per physical mark.

    - Same class: drop if IoU is high, or compact symbols share a near center
      (two YOLO boxes on one camera / stub).
    - Different class: drop the lower-confidence box when they sit on the same
      glyph (e.g. WM5400 vs WM5500 on one raceway stroke).
    Elongated raceway segments are *not* merged by center distance, only IoU.

    compact_center_px must stay below the smallest real symbol size: two
    TOUCHING small symbols have centers one symbol-width apart, and a floor
    above that width silently merges connected clusters into one count
    (measured on eval: floor 16 erased 12 cluster members, floor 8 saves 7
    of them with no false-positive increase).
    """
    if not hits:
        return []
    ranked = sorted(hits, key=lambda h: float(h.confidence), reverse=True)
    kept: list[YoloHit] = []
    for hit in ranked:
        drop = False
        hcx, hcy = _center(hit)
        compact = _is_compact(hit)
        for other in kept:
            iou = box_iou(hit, other)
            same = _same_label(hit, other)
            if same and iou >= same_class_iou:
                drop = True
                break
            if not same and iou >= cross_class_iou:
                drop = True
                break
            if same and compact and _is_compact(other):
                ocx, ocy = _center(other)
                dist = ((hcx - ocx) ** 2 + (hcy - ocy) ** 2) ** 0.5
                span = min(
                    max(hit.x2 - hit.x1, hit.y2 - hit.y1),
                    max(other.x2 - other.x1, other.y2 - other.y1),
                )
                if dist <= max(compact_center_px, 0.45 * span):
                    drop = True
                    break
        if not drop:
            kept.append(hit)
    return kept
