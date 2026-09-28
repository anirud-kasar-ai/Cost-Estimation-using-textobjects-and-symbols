"""Does more zoom (higher inference resolution) fix touching-symbol clusters?

Runs each detector on its eval set at several imgsz values and reports, per
zoom level: recall on cluster members (touching same-class symbols), recall on
isolated symbols, and false positives at the production threshold (0.35).

Usage (from repo root):  python scripts/experiment_zoom.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DZS = ROOT / "drawing-zoom-split"
sys.path.insert(0, str(DZS / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_clusters import greedy_match, iou, touching  # noqa: E402
from pipeline.yolo_detect import YoloHit  # noqa: E402

IMGSZ_LIST = [640, 1280, 1920]
RAW_CONF = 0.10
PROD_CONF = 0.35
MATCH_IOU = 0.50

MODELS = [
    ("main 52c", DZS / "model" / "symbol_detector_tech.pt", ROOT / "eval_data" / "test_project" / "test"),
    ("roof 9c", DZS / "model" / "symbol_detector_best.pt", ROOT / "eval_data" / "test_small_9c" / "test"),
]


def load_gt(label_path: Path, w: int, h: int) -> list[tuple]:
    gts = []
    if not label_path.is_file():
        return gts
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        gts.append(
            (cid, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        )
    return gts


def main() -> None:
    for name, weights, dataset in MODELS:
        images = sorted((dataset / "images").glob("*"))
        model = YOLO(str(weights))
        print(f"\n=== {name} ({len(images)} images) ===")
        for imgsz in IMGSZ_LIST:
            t0 = time.time()
            stats = {"cm": 0, "cm_hit": 0, "sg": 0, "sg_hit": 0, "fp": 0}
            for img_path in images:
                img = Image.open(img_path)
                gts = load_gt(
                    dataset / "labels" / (img_path.stem + ".txt"), img.width, img.height
                )
                res = model.predict(
                    source=img, conf=RAW_CONF, imgsz=imgsz, verbose=False
                )[0]
                hits = [
                    YoloHit(
                        class_id=int(b.cls),
                        class_name=res.names[int(b.cls)],
                        confidence=float(b.conf),
                        x1=float(b.xyxy[0][0]),
                        y1=float(b.xyxy[0][1]),
                        x2=float(b.xyxy[0][2]),
                        y2=float(b.xyxy[0][3]),
                    )
                    for b in res.boxes
                ]
                if not gts and not hits:
                    continue

                in_cluster = [False] * len(gts)
                for i in range(len(gts)):
                    for j in range(i + 1, len(gts)):
                        if touching(gts[i], gts[j]):
                            in_cluster[i] = in_cluster[j] = True

                matched = greedy_match(gts, hits)
                for i in range(len(gts)):
                    key = "cm" if in_cluster[i] else "sg"
                    stats[key] += 1
                    if matched[i]:
                        stats[f"{key}_hit"] += 1

                # false positives at production threshold
                prod = [h for h in hits if h.confidence >= PROD_CONF]
                taken = [False] * len(gts)
                for h in sorted(prod, key=lambda x: x.confidence, reverse=True):
                    best_i, best_v = -1, MATCH_IOU
                    for i, g in enumerate(gts):
                        if taken[i] or g[0] != h.class_id:
                            continue
                        v = iou(g[1:], (h.x1, h.y1, h.x2, h.y2))
                        if v >= best_v:
                            best_i, best_v = i, v
                    if best_i >= 0:
                        taken[best_i] = True
                    else:
                        stats["fp"] += 1

            cm, sg = stats["cm"], stats["sg"]
            print(
                f"imgsz {imgsz:4d}: cluster recall {stats['cm_hit'] / cm:.1%} ({stats['cm_hit']}/{cm})"
                f"  isolated recall {stats['sg_hit'] / sg:.1%} ({stats['sg_hit']}/{sg})"
                f"  FPs@0.35 {stats['fp']}"
                f"  [{time.time() - t0:.0f}s]"
            )


if __name__ == "__main__":
    main()
