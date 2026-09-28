"""Trace every raw YOLO candidate on one wing through the pipeline.

Re-runs the model on each zoom tile at very low conf (0.02, with and
without TTA), maps candidates to ROI space, clusters duplicates across
overlapping tiles, then compares against the final kept detections in
06_counts/result.json. Every cluster that has no matching final box is
categorized:

  DROPPED-BY-MERGE   best conf >= primary conf (0.35)  -> the model found it
                     confidently but NMS / cross-tile merge deleted it
  RESCUE-DROPPED     0.15 <= best conf < 0.35          -> rescue pass saw it
                     but it was merged/deduped away
  BELOW-FLOOR        best conf < 0.15                  -> under the rescue floor

Writes an overlay (kept = green, dropped-by-merge = red, rescue-dropped =
orange, below-floor = gray dashed) next to the job's 06_counts folder.

Usage (from drawing-zoom-split/):
  python scripts/diagnose_wing.py --job 1974BIDDrawings-GlenbrookMS \
      --wing ROOF-PLAN --page page_003 --model model/symbol_detector_best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--wing", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--model", default="model/symbol_detector_best.pt")
    ap.add_argument("--raw-conf", type=float, default=0.02)
    ap.add_argument("--primary-conf", type=float, default=0.35)
    ap.add_argument("--rescue-conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    job_dir = Path("storage/jobs") / args.job
    zoom_dir = job_dir / "04_wings" / args.wing / "zooms" / args.page
    manifest = json.loads((zoom_dir / "zooms_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((job_dir / "06_counts" / "result.json").read_text(encoding="utf-8"))

    wing_key = f"zooms/{args.page}"
    finals = [
        d
        for d in result.get("yolo_detections", [])
        if str(d.get("wing_folder", "")).endswith(wing_key)
        and args.wing in str(d.get("wing_folder", ""))
    ]
    print(f"final kept detections on this wing: {len(finals)}")
    print(Counter(d["class_name"] for d in finals))

    from ultralytics import YOLO

    model = YOLO(args.model)

    raw: list[dict] = []
    for t in manifest["tiles"]:
        tile_path = zoom_dir / t["file"]
        img = Image.open(tile_path).convert("RGB")
        bb = t["bbox_roi"]
        sx = (bb["x2"] - bb["x1"]) / img.width
        sy = (bb["y2"] - bb["y1"]) / img.height
        for augment in (False, True):
            results = model.predict(
                img, imgsz=args.imgsz, conf=args.raw_conf, iou=0.45,
                augment=augment, verbose=False,
            )
            for r in results:
                names = r.names
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    raw.append(
                        {
                            "class": names[int(b.cls[0])],
                            "conf": float(b.conf[0]),
                            "box": (
                                bb["x1"] + x1 * sx,
                                bb["y1"] + y1 * sy,
                                bb["x1"] + x2 * sx,
                                bb["y1"] + y2 * sy,
                            ),
                            "tile": t["file"],
                            "aug": augment,
                        }
                    )
        print(f"  {t['file']}: raw so far {len(raw)}")

    # Cluster raw candidates (same class, IoU >= 0.4) keeping best conf.
    raw.sort(key=lambda d: -d["conf"])
    clusters: list[dict] = []
    for d in raw:
        for c in clusters:
            if c["class"] == d["class"] and iou(c["box"], d["box"]) >= 0.4:
                c["hits"] += 1
                break
        else:
            clusters.append({**d, "hits": 1})
    print(f"\nraw candidates: {len(raw)}  ->  {len(clusters)} unique clusters")

    final_boxes = [
        (d["class_name"], (d["x1"], d["y1"], d["x2"], d["y2"])) for d in finals
    ]

    def matched(c: dict) -> bool:
        return any(
            cls == c["class"] and iou(box, c["box"]) >= 0.3 for cls, box in final_boxes
        )

    missed = [c for c in clusters if not matched(c)]
    bands = {"DROPPED-BY-MERGE": [], "RESCUE-DROPPED": [], "BELOW-FLOOR": []}
    for c in missed:
        if c["conf"] >= args.primary_conf:
            bands["DROPPED-BY-MERGE"].append(c)
        elif c["conf"] >= args.rescue_conf:
            bands["RESCUE-DROPPED"].append(c)
        else:
            bands["BELOW-FLOOR"].append(c)

    print(f"\nclusters with NO matching final detection: {len(missed)}")
    for band, items in bands.items():
        print(f"\n== {band}: {len(items)} ==")
        for c in sorted(items, key=lambda d: -d["conf"]):
            x1, y1, x2, y2 = c["box"]
            print(
                f"  {c['class']:<16} conf={c['conf']:.3f} "
                f"roi=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                f"first_tile={c['tile']} aug={c['aug']} dup_hits={c['hits']}"
            )

    # Overlay
    wing_img_path = zoom_dir / "full_wing.jpg"
    wing = Image.open(wing_img_path).convert("RGB")
    roi = manifest["roi_box_px"]
    ox = wing.width / (roi["x2"] - roi["x1"])
    oy = wing.height / (roi["y2"] - roi["y1"])
    draw = ImageDraw.Draw(wing)

    def rect(box, color, width=3):
        x1, y1, x2, y2 = box
        draw.rectangle(
            (x1 * ox, y1 * oy, x2 * ox, y2 * oy), outline=color, width=width
        )

    for _, box in final_boxes:
        rect(box, (0, 170, 0), 2)
    for c in bands["BELOW-FLOOR"]:
        rect(c["box"], (140, 140, 140), 2)
    for c in bands["RESCUE-DROPPED"]:
        rect(c["box"], (255, 140, 0), 3)
    for c in bands["DROPPED-BY-MERGE"]:
        rect(c["box"], (220, 0, 0), 4)

    out = job_dir / "06_counts" / f"diagnose_{args.wing}_{args.page}.png"
    wing.save(out)
    print(f"\noverlay written: {out}")
    print("legend: green=kept, red=dropped-by-merge, orange=rescue-dropped, gray=below-floor")


if __name__ == "__main__":
    main()
