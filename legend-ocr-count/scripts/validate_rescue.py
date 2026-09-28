"""Before/after recall check for the two-pass rescue detection.

Runs the real production path (detect_symbols -> nms_yolo_hits) over
eval_data/test_small_9c/test with rescue disabled vs enabled and reports
recall / precision against ground-truth labels (class-aware IoU >= 0.5).

Usage: python scripts/validate_rescue.py   (from legend-ocr-count/)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from pipeline import config  # noqa: E402
from pipeline.nms import nms_yolo_hits  # noqa: E402
from pipeline.yolo_detect import detect_symbols  # noqa: E402

EVAL = ROOT.parent / "eval_data" / "test_small_9c" / "test"
MATCH_IOU = 0.50


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(p[0])), *(float(v) for v in p[1:5])
        out.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(rescue: bool) -> dict:
    config.YOLO_RESCUE_ENABLED = rescue
    tp = n_gt = n_pred = rescued_used = 0
    images = sorted((EVAL / "images").glob("*.jpg")) + sorted((EVAL / "images").glob("*.png"))
    for img_path in images:
        img = Image.open(img_path)
        gts = load_gt(EVAL / "labels" / (img_path.stem + ".txt"), *img.size)
        n_gt += len(gts)
        hits = detect_symbols(img, image_name=img_path.name)
        hits = nms_yolo_hits(
            hits,
            same_class_iou=config.YOLO_NMS_IOU,
            cross_class_iou=config.YOLO_CROSS_NMS_IOU,
            compact_center_px=config.YOLO_NMS_CENTER_PX,
        )
        n_pred += len(hits)
        taken = [False] * len(gts)
        for h in sorted(hits, key=lambda x: x.confidence, reverse=True):
            best_i, best_v = -1, MATCH_IOU
            for i, g in enumerate(gts):
                if taken[i] or g[0] != h.class_id:
                    continue
                v = iou(g[1:], (h.x1, h.y1, h.x2, h.y2))
                if v >= best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                taken[best_i] = True
                tp += 1
                if h.rescued:
                    rescued_used += 1
    return {
        "gt": n_gt,
        "pred": n_pred,
        "tp": tp,
        "recall": tp / n_gt if n_gt else 0.0,
        "precision": tp / n_pred if n_pred else 0.0,
        "rescued_matched": rescued_used,
    }


def main() -> None:
    print(f"Eval: {EVAL} | prod conf={config.YOLO_CONF} rescue conf={config.YOLO_RESCUE_CONF} "
          f"tta={config.YOLO_RESCUE_TTA}")
    base = evaluate(rescue=False)
    resc = evaluate(rescue=True)
    print(f"\n{'':22}{'single-pass':>12}{'with rescue':>12}")
    for k in ("gt", "pred", "tp", "recall", "precision", "rescued_matched"):
        b, r = base[k], resc[k]
        fmt = (lambda v: f"{v:.3f}") if isinstance(b, float) else str
        print(f"{k:22}{fmt(b):>12}{fmt(r):>12}")


if __name__ == "__main__":
    main()
