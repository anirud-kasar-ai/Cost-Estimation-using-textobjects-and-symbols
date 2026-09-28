"""Before/after recall check for the two-pass rescue in drawing-zoom-split.

Runs this project's production path (detect_symbols -> nms_yolo_hits) over a
labeled eval split with rescue disabled vs enabled and reports recall /
precision against ground truth (class-aware IoU >= 0.5). Also prints the
check-4 scale facts (tile px vs training/inference imgsz).

Usage (from drawing-zoom-split/):
  python scripts/validate_rescue.py --model model/symbol_detector_best.pt \
      --dataset ../eval_data/test_small_9c/test
  python scripts/validate_rescue.py --model model/symbol_detector_tech.pt \
      --dataset ../eval_data/test_project/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from pipeline import config  # noqa: E402
from pipeline.nms import nms_yolo_hits  # noqa: E402
from pipeline.yolo_detect import detect_symbols  # noqa: E402

MATCH_IOU = 0.50


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not label_path.is_file():
        return out
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


def evaluate(dataset: Path, weights: Path, rescue: bool) -> dict:
    config.YOLO_RESCUE_ENABLED = rescue
    tp = n_gt = n_pred = rescued_used = 0
    img_dir = dataset / "images"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for img_path in images:
        img = Image.open(img_path)
        gts = load_gt(dataset / "labels" / (img_path.stem + ".txt"), *img.size)
        n_gt += len(gts)
        hits = detect_symbols(img, image_name=img_path.name, weights=weights)
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
        "images": len(images),
        "gt": n_gt,
        "pred": n_pred,
        "tp": tp,
        "recall": tp / n_gt if n_gt else 0.0,
        "precision": tp / n_pred if n_pred else 0.0,
        "rescued_matched": rescued_used,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="split folder with images/ + labels/")
    args = ap.parse_args()
    weights = Path(args.model)
    dataset = Path(args.dataset)

    overlap_px = config.ROI_ZOOM_TILE_SIZE * config.ROI_ZOOM_OVERLAP_PCT
    print(f"Model: {weights.name} | dataset: {dataset}")
    print(f"Check 4 (scale): tiles {config.ROI_ZOOM_TILE_SIZE}px letterboxed to "
          f"imgsz {config.YOLO_IMGSZ} at inference (training convention: 1280).")
    print(f"Check 3 (overlap): {config.ROI_ZOOM_OVERLAP_PCT:.2f} => {overlap_px:.0f}px.")
    print(f"Rescue: conf {config.YOLO_RESCUE_CONF}, tta {config.YOLO_RESCUE_TTA}, "
          f"prod conf {config.YOLO_CONF}\n")

    base = evaluate(dataset, weights, rescue=False)
    resc = evaluate(dataset, weights, rescue=True)
    print(f"{'':22}{'single-pass':>12}{'with rescue':>12}")
    for k in ("images", "gt", "pred", "tp", "recall", "precision", "rescued_matched"):
        b, r = base[k], resc[k]
        fmt = (lambda v: f"{v:.3f}") if isinstance(b, float) else str
        print(f"{k:22}{fmt(b):>12}{fmt(r):>12}")


if __name__ == "__main__":
    main()
