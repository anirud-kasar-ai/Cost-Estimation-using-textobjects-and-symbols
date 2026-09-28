"""Build a true/false crop dataset for the detection verification classifier.

Reads the cached detections from scripts/tune_thresholds.py (all YOLO hits at
the lowest rescue floor), labels each hit by matching against ground truth
(any-class IoU >= 0.5: the question is "is this a real symbol at all?"), crops
it from the eval image with context margin, and writes an Ultralytics
classification folder:

    eval_data/verifier/
        train/true_symbol/*.jpg   train/false_symbol/*.jpg
        val/true_symbol/*.jpg     val/false_symbol/*.jpg

The split is by IMAGE (hash of the stem), never by crop, so the validation
numbers are honest — no drawing appears in both splits.

Usage (from repo root):
    python scripts/build_verifier_dataset.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_data" / "verifier"
MATCH_IOU = 0.50
MARGIN = 0.40  # context margin around the box on each side
MIN_CROP = 32  # px
VAL_FRACTION = 0.30

SOURCES = [
    {
        "cache": ROOT / "eval_data" / "report_misses" / "cache_main_52c.json",
        "images": ROOT / "eval_data" / "test_project" / "test" / "images",
        "tag": "main",
    },
    {
        "cache": ROOT / "eval_data" / "report_misses" / "cache_roof_9c.json",
        "images": ROOT / "eval_data" / "test_small_9c" / "test" / "images",
        "tag": "roof",
    },
]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def split_of(stem: str) -> str:
    h = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16)
    return "val" if (h % 100) < VAL_FRACTION * 100 else "train"


def main() -> None:
    counts = {"train": {"true_symbol": 0, "false_symbol": 0},
              "val": {"true_symbol": 0, "false_symbol": 0}}

    for src in SOURCES:
        if not src["cache"].is_file():
            print(f"SKIP {src['tag']}: no cache at {src['cache']}")
            continue
        cache = json.loads(src["cache"].read_text(encoding="utf-8"))
        for image_name, entry in cache["images"].items():
            img_path = src["images"] / image_name
            if not img_path.is_file():
                continue
            gts = [tuple(g) for g in entry["gt"]]
            hits = entry["hits"]
            if not hits:
                continue
            split = split_of(f"{src['tag']}:{img_path.stem}")
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                for k, hit in enumerate(hits):
                    box = hit["box"]
                    matched = any(iou(g[1:], box) >= MATCH_IOU for g in gts)
                    label = "true_symbol" if matched else "false_symbol"

                    bw, bh = box[2] - box[0], box[3] - box[1]
                    mx, my = MARGIN * bw, MARGIN * bh
                    x1, y1 = box[0] - mx, box[1] - my
                    x2, y2 = box[2] + mx, box[3] + my
                    # enforce a minimum crop so tiny boxes keep context
                    if x2 - x1 < MIN_CROP:
                        cx = (x1 + x2) / 2
                        x1, x2 = cx - MIN_CROP / 2, cx + MIN_CROP / 2
                    if y2 - y1 < MIN_CROP:
                        cy = (y1 + y2) / 2
                        y1, y2 = cy - MIN_CROP / 2, cy + MIN_CROP / 2
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue

                    crop = im.crop((x1, y1, x2, y2))
                    dest = OUT / split / label
                    dest.mkdir(parents=True, exist_ok=True)
                    crop.save(dest / f"{src['tag']}_{img_path.stem}_{k}.jpg", quality=92)
                    counts[split][label] += 1

    print(json.dumps(counts, indent=2))
    print(f"Dataset at: {OUT}")


if __name__ == "__main__":
    main()
