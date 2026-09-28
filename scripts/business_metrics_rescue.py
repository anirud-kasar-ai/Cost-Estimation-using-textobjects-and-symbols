"""Business metrics BEFORE vs AFTER the two-pass rescue (double detection).

Runs the drawing-zoom-split production detection path (detect_symbols ->
nms_yolo_hits) over the labeled test sets with rescue disabled (before) and
enabled (after), and writes docs/business_metrics_rescue.csv with the same
object-level business metrics as docs/overall_business_metrics.csv:

  actual objects, correctly identified, object recall, wrong class, missed,
  total detections, false detections, object precision, object F1.

Matching is the standard class-aware IoU >= 0.5; detections that land on a
ground-truth box of a different class count as "wrong class"; leftover
detections are false detections.

Usage (from repo root):
    python scripts/business_metrics_rescue.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DZS = ROOT / "drawing-zoom-split"
sys.path.insert(0, str(DZS / "src"))

from PIL import Image  # noqa: E402

from pipeline import config  # noqa: E402
from pipeline.nms import nms_yolo_hits  # noqa: E402
from pipeline.yolo_detect import detect_symbols  # noqa: E402

MATCH_IOU = 0.50

MODELS = [
    {
        "name": "Main symbol detector (52 classes)",
        "classes": 52,
        "weights": DZS / "model" / "symbol_detector_tech.pt",
        "dataset": ROOT / "eval_data" / "test_project" / "test",
    },
    {
        "name": "Roof symbol detector (9 classes)",
        "classes": 9,
        "weights": DZS / "model" / "symbol_detector_best.pt",
        "dataset": ROOT / "eval_data" / "test_small_9c" / "test",
    },
]


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(p[0])), *(float(v) for v in p[1:5])
        out.append(
            (cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        )
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Tally:
    images: int = 0
    gt: int = 0
    pred: int = 0
    correct: int = 0
    wrong_class: int = 0
    false_det: int = 0
    rescued_matched: int = 0

    @property
    def missed(self) -> int:
        return self.gt - self.correct - self.wrong_class

    @property
    def recall(self) -> float:
        return self.correct / self.gt if self.gt else 0.0

    @property
    def precision(self) -> float:
        return self.correct / self.pred if self.pred else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate(dataset: Path, weights: Path, rescue: bool) -> Tally:
    config.YOLO_RESCUE_ENABLED = rescue
    t = Tally()
    img_dir = dataset / "images"
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    t.images = len(images)
    for img_path in images:
        img = Image.open(img_path)
        gts = load_gt(dataset / "labels" / (img_path.stem + ".txt"), *img.size)
        t.gt += len(gts)
        hits = detect_symbols(img, image_name=img_path.name, weights=weights)
        hits = nms_yolo_hits(
            hits,
            same_class_iou=config.YOLO_NMS_IOU,
            cross_class_iou=config.YOLO_CROSS_NMS_IOU,
            compact_center_px=config.YOLO_NMS_CENTER_PX,
        )
        t.pred += len(hits)

        taken = [False] * len(gts)
        unmatched_hits = []
        # Pass A: class-aware matching (correctly identified).
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
                t.correct += 1
                if h.rescued:
                    t.rescued_matched += 1
            else:
                unmatched_hits.append(h)
        # Pass B: remaining detections on a remaining GT box = wrong class.
        for h in unmatched_hits:
            best_i, best_v = -1, MATCH_IOU
            for i, g in enumerate(gts):
                if taken[i]:
                    continue
                v = iou(g[1:], (h.x1, h.y1, h.x2, h.y2))
                if v >= best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                taken[best_i] = True
                t.wrong_class += 1
            else:
                t.false_det += 1
    return t


def main() -> None:
    rows = []
    totals: dict[str, Tally] = {"single-pass (before)": Tally(), "two-pass rescue (after)": Tally()}
    for m in MODELS:
        if not m["weights"].is_file() or not m["dataset"].is_dir():
            print(f"SKIP {m['name']}: missing weights or dataset")
            continue
        for mode, rescue in (("single-pass (before)", False), ("two-pass rescue (after)", True)):
            print(f"Evaluating {m['name']} — {mode} ...", flush=True)
            t = evaluate(m["dataset"], m["weights"], rescue)
            rows.append((m["name"], m["classes"], mode, t))
            agg = totals[mode]
            agg.images += t.images
            agg.gt += t.gt
            agg.pred += t.pred
            agg.correct += t.correct
            agg.wrong_class += t.wrong_class
            agg.false_det += t.false_det
            agg.rescued_matched += t.rescued_matched
            print(
                f"  gt={t.gt} correct={t.correct} recall={t.recall:.4f} "
                f"precision={t.precision:.4f} rescued_matched={t.rescued_matched}",
                flush=True,
            )
    for mode, t in totals.items():
        rows.append(("Overall (both models)", "52+9", mode, t))

    out = ROOT / "docs" / "business_metrics_rescue.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Model",
                "Classes",
                "Mode",
                "Images",
                "Actual objects",
                "Correctly identified",
                "Object recall",
                "Wrong class",
                "Missed",
                "Total detections",
                "False detections",
                "Object precision",
                "Object F1",
                "Rescued symbols matched",
            ]
        )
        for name, classes, mode, t in rows:
            w.writerow(
                [
                    name,
                    classes,
                    mode,
                    t.images,
                    t.gt,
                    t.correct,
                    f"{t.recall:.4f}",
                    t.wrong_class,
                    t.missed,
                    t.pred,
                    t.false_det,
                    f"{t.precision:.4f}",
                    f"{t.f1:.4f}",
                    t.rescued_matched,
                ]
            )
    print(f"\nSaved: {out}")
    print(
        f"Settings: prod conf {config.YOLO_CONF}, rescue conf {config.YOLO_RESCUE_CONF}, "
        f"tta {config.YOLO_RESCUE_TTA}, match IoU {MATCH_IOU}"
    )


if __name__ == "__main__":
    main()
