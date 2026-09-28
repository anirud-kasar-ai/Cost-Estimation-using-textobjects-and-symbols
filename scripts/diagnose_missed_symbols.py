"""Missed-symbol diagnostics on the labeled eval set.

Sweeps eval_data/test_project/<split> with the production YOLO weights and
classifies every ground-truth symbol into:
  - detected        (conf >= production 0.35)
  - threshold_miss  (found at 0.05 <= conf < 0.35 -> silently dropped in prod)
  - tta_only        (recovered only with augment=True test-time augmentation)
  - undetected      (not found even at conf 0.05)

Also reports per-class training support (instances in train/labels), symbol
geometry (size at inference scale, distance to image edge, fit inside the
production tile overlap), and a per-class diagnosis + recommended fix.

Outputs (eval_data/report_misses/):
  miss_details.csv       one row per missed GT instance
  per_class_summary.csv  recall at 0.35 / 0.05 / TTA + geometry + train count
  REPORT.md              ranked findings and recommendations
  overlays/*.jpg         representative missed instances drawn on the image

Two model/dataset pairings exist in this repo (class lists must match):
  - legend-ocr-count/model/symbol_detector_best.pt (9 roof classes)
        -> eval_data/test_small_9c, train labels eval_data/train_small/train/labels
  - drawing-zoom-split/model/symbol_detector_tech.pt (52 tech-symbol classes)
        -> eval_data/test_project, train labels eval_data/train_symbo/train/labels

Usage:
  python scripts/diagnose_missed_symbols.py --model <weights.pt> --dataset <folder> \
      --train-labels <labels dir> --out <report dir> [--split test]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Production settings (legend-ocr-count/pipeline/config.py)
PROD_CONF = 0.35
LOW_CONF = 0.05
IMGSZ = 1280
MATCH_IOU = 0.50
# Production tiling (drawing-zoom-split/config/defaults.yaml)
TILE_SIZE = 653
TILE_OVERLAP_PX = int(round(TILE_SIZE * 0.10))  # ~65 px
# P3 (stride 8) practical floor at inference resolution
TINY_PX_AT_IMGSZ = 16.0
FEW_SHOT_TRAIN = 20

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class GtBox:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class Pred:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    matched: bool = field(default=False)


def iou(a, b) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0


def load_yaml_names(path: Path) -> list[str]:
    """Minimal parse of the names: [...] line in data.yaml (no yaml dep needed)."""
    import ast

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("names:"):
            return [str(n) for n in ast.literal_eval(s[len("names:"):].strip())]
    raise ValueError(f"names: not found in {path}")


def load_gt(label_path: Path, img_w: int, img_h: int) -> list[GtBox]:
    boxes: list[GtBox] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            GtBox(
                cls=cls,
                x1=(cx - w / 2) * img_w,
                y1=(cy - h / 2) * img_h,
                x2=(cx + w / 2) * img_w,
                y2=(cy + h / 2) * img_h,
            )
        )
    return boxes


def preds_from_result(result) -> list[Pred]:
    out: list[Pred] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    for b in boxes:
        xyxy = b.xyxy[0].tolist()
        out.append(
            Pred(
                cls=int(b.cls.item()),
                conf=float(b.conf.item()),
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )
        )
    return out


def match_gt(gts: list[GtBox], preds: list[Pred]) -> list[Pred | None]:
    """Greedy same-class matching: highest-conf pred claims the best free GT.

    Returns, for each GT (same order), the matched Pred or None.
    """
    assigned: list[Pred | None] = [None] * len(gts)
    for p in sorted(preds, key=lambda x: x.conf, reverse=True):
        best_i, best_iou = -1, MATCH_IOU
        for i, g in enumerate(gts):
            if assigned[i] is not None or g.cls != p.cls:
                continue
            v = iou(g, p)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            assigned[best_i] = p
            p.matched = True
    return assigned


def best_cross_class(g: GtBox, preds: list[Pred]) -> tuple[Pred | None, float]:
    """Highest-IoU pred of a DIFFERENT class overlapping this GT (>=0.5)."""
    best, best_v = None, MATCH_IOU
    for p in preds:
        if p.cls == g.cls:
            continue
        v = iou(g, p)
        if v >= best_v:
            best, best_v = p, v
    return best, (best_v if best else 0.0)


def count_train_instances(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for f in labels_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                counts[int(float(parts[0]))] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .pt weights")
    ap.add_argument("--dataset", required=True, help="dataset root with <split>/images+labels")
    ap.add_argument("--train-labels", required=True, help="dir of train label .txt files")
    ap.add_argument("--out", required=True, help="report output directory")
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--max-overlays", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    dataset = Path(args.dataset)
    model_path = Path(args.model)
    train_labels_dir = Path(args.train_labels)
    out_dir = Path(args.out)

    img_dir = dataset / args.split / "images"
    lbl_dir = dataset / args.split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"No images in {img_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)

    yaml_path = next(p for p in (dataset / "data_fixed.yaml", dataset / "data.yaml") if p.is_file())
    names = load_yaml_names(yaml_path)
    model = YOLO(str(model_path))
    model_names = {int(k): str(v) for k, v in (model.names or {}).items()}

    # ---- Class-name alignment check (index mismatch masquerades as misses) ----
    mismatches = [
        (i, names[i], model_names.get(i, "<missing>"))
        for i in range(len(names))
        if model_names.get(i, "").strip().lower() != names[i].strip().lower()
    ]
    if len(mismatches) > len(names) // 2:
        sys.exit(
            f"ABORT: {len(mismatches)}/{len(names)} class names differ between "
            f"{model_path.name} and {yaml_path} - wrong model/dataset pairing.\n"
            "First mismatches: "
            + "; ".join(f"id {i}: yaml={y!r} model={m!r}" for i, y, m in mismatches[:3])
        )
    if mismatches:
        print("WARNING: model.names vs data.yaml mismatches:")
        for i, y, m in mismatches:
            print(f"  id {i}: yaml={y!r}  model={m!r}")
    else:
        print(f"Class-name alignment OK ({len(names)} classes match model.names).")

    # ---- Pass 1: base sweep at conf=0.05 ----
    print(f"\nPass 1: base predict on {len(images)} '{args.split}' images "
          f"(conf={LOW_CONF}, imgsz={IMGSZ}) ...")
    per_image: dict[str, dict] = {}
    img_sizes: dict[str, tuple[int, int]] = {}
    for n, path in enumerate(images, 1):
        with Image.open(path) as im:
            w, h = im.size
        img_sizes[path.name] = (w, h)
        gts = load_gt(lbl_dir / (path.stem + ".txt"), w, h)
        res = model.predict(source=str(path), conf=LOW_CONF, imgsz=IMGSZ, verbose=False)
        preds = preds_from_result(res[0]) if res else []
        per_image[path.name] = {"path": path, "gts": gts, "preds": preds,
                                "assigned": match_gt(gts, preds)}
        if n % 25 == 0 or n == len(images):
            print(f"  {n}/{len(images)}")

    # Images that still have unmatched GT -> TTA pass only on those
    miss_images = [k for k, v in per_image.items()
                   if any(a is None for a in v["assigned"])]
    print(f"\nPass 1 done. {len(miss_images)}/{len(images)} images have >=1 GT "
          f"unmatched at conf {LOW_CONF}. Running TTA on those ...")

    for n, key in enumerate(miss_images, 1):
        entry = per_image[key]
        res = model.predict(source=str(entry["path"]), conf=LOW_CONF, imgsz=IMGSZ,
                            augment=True, verbose=False)
        tta_preds = preds_from_result(res[0]) if res else []
        # Match TTA preds only against the still-unmatched GTs
        open_idx = [i for i, a in enumerate(entry["assigned"]) if a is None]
        open_gts = [entry["gts"][i] for i in open_idx]
        tta_assigned = match_gt(open_gts, tta_preds)
        entry["tta_assigned"] = dict(zip(open_idx, tta_assigned))
        entry["tta_preds"] = tta_preds
        if n % 10 == 0 or n == len(miss_images):
            print(f"  TTA {n}/{len(miss_images)}")

    # ---- Pass 2: training support ----
    train_counts = count_train_instances(train_labels_dir)

    # ---- Classify every GT instance ----
    rows: list[dict] = []          # all GT instances
    miss_rows: list[dict] = []     # misses only (for CSV + overlays)
    per_class: dict[int, dict] = defaultdict(lambda: {
        "gt": 0, "detected": 0, "threshold_miss": 0, "tta_only": 0,
        "undetected": 0, "sizes": [], "sizes_scaled": [], "miss_edge_near": 0,
        "miss_confs": [], "misclassified": Counter(),
    })

    for key, entry in per_image.items():
        w, h = img_sizes[key]
        scale = IMGSZ / max(w, h)  # letterbox scale to inference resolution
        tta_assigned = entry.get("tta_assigned", {})
        for i, (g, a) in enumerate(zip(entry["gts"], entry["assigned"])):
            c = per_class[g.cls]
            c["gt"] += 1
            size_px = max(g.w, g.h)
            c["sizes"].append(size_px)
            c["sizes_scaled"].append(size_px * scale)
            edge_dist = min(g.x1, g.y1, w - g.x2, h - g.y2)

            if a is not None and a.conf >= PROD_CONF:
                status, conf = "detected", a.conf
            elif a is not None:
                status, conf = "threshold_miss", a.conf
            else:
                ta = tta_assigned.get(i)
                if ta is not None:
                    status, conf = "tta_only", ta.conf
                else:
                    status, conf = "undetected", 0.0

            row = {
                "image": key, "class_id": g.cls, "class_name": names[g.cls],
                "status": status, "conf": round(conf, 3),
                "x1": round(g.x1, 1), "y1": round(g.y1, 1),
                "x2": round(g.x2, 1), "y2": round(g.y2, 1),
                "box_px": round(size_px, 1),
                "box_px_at_1280": round(size_px * scale, 1),
                "edge_dist_px": round(edge_dist, 1),
                "fits_tile_overlap": size_px <= TILE_OVERLAP_PX,
                "misclassified_as": "", "misclass_conf": "",
            }
            if status == "detected":
                c["detected"] += 1
            else:
                c[status if status != "tta_only" else "tta_only"] += 1
                if status == "threshold_miss":
                    c["miss_confs"].append(conf)
                if edge_dist <= 0.05 * min(w, h):
                    c["miss_edge_near"] += 1
                if status == "undetected":
                    xc, xiou = best_cross_class(g, entry["preds"])
                    if xc is not None:
                        row["misclassified_as"] = names[xc.cls] if xc.cls < len(names) else str(xc.cls)
                        row["misclass_conf"] = round(xc.conf, 3)
                        c["misclassified"][xc.cls] += 1
                miss_rows.append(row)
            rows.append(row)

    # ---- Pass 3 rollup + diagnosis per class ----
    summary_rows: list[dict] = []
    for cls_id in range(len(names)):
        c = per_class[cls_id]
        gt = c["gt"]
        det, thr, tta, und = c["detected"], c["threshold_miss"], c["tta_only"], c["undetected"]
        med_px = statistics.median(c["sizes"]) if c["sizes"] else 0.0
        med_1280 = statistics.median(c["sizes_scaled"]) if c["sizes_scaled"] else 0.0
        tr = train_counts.get(cls_id, 0)
        misses = gt - det

        tags: list[str] = []
        recs: list[str] = []
        if misses:
            if thr >= max(1, misses // 2):
                lo = min(c["miss_confs"]) if c["miss_confs"] else 0.0
                hi = max(c["miss_confs"]) if c["miss_confs"] else 0.0
                tags.append("threshold")
                recs.append(f"found at conf {lo:.2f}-{hi:.2f} but dropped at 0.35; "
                            "more training examples to push conf up (or lower conf for this class)")
            if tr < FEW_SHOT_TRAIN:
                tags.append("data_scarcity")
                recs.append(f"only {tr} train instances (<{FEW_SHOT_TRAIN}); label more examples")
            if med_1280 < TINY_PX_AT_IMGSZ:
                tags.append("tiny_p2_candidate")
                recs.append(f"median {med_1280:.0f}px at imgsz {IMGSZ} (<{TINY_PX_AT_IMGSZ:.0f}px, "
                            "below P3 stride-8 floor); retrain with yolov8-p2.yaml")
            if c["miss_edge_near"] >= max(1, misses // 2):
                tags.append("tile_edge")
                recs.append(f"{c['miss_edge_near']}/{misses} misses within 5% of an image edge; "
                            f"symbol {med_px:.0f}px vs {TILE_OVERLAP_PX}px tile overlap - increase overlap")
            if c["misclassified"]:
                top = c["misclassified"].most_common(1)[0]
                tags.append("confused_class")
                recs.append(f"often predicted as '{names[top[0]]}' instead ({top[1]}x); "
                            "check label consistency between the two classes")
            if und and not tags:
                tags.append("hard_examples")
                recs.append("not found even at conf 0.05 and not tiny/rare; "
                            "inspect overlays - likely appearance drift vs training data")

        summary_rows.append({
            "class_id": cls_id,
            "class_name": names[cls_id] if cls_id < len(names) else str(cls_id),
            "gt_instances": gt,
            "train_instances": tr,
            "recall_at_0.35": round(det / gt, 3) if gt else "",
            "recall_at_0.05": round((det + thr) / gt, 3) if gt else "",
            "recall_with_tta": round((det + thr + tta) / gt, 3) if gt else "",
            "threshold_miss": thr,
            "tta_only": tta,
            "undetected": und,
            "median_box_px": round(med_px, 1),
            "median_box_px_at_1280": round(med_1280, 1),
            "diagnosis": "+".join(tags),
            "recommendation": " | ".join(recs),
        })

    # ---- Write CSVs ----
    with (out_dir / "miss_details.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(miss_rows[0].keys()) if miss_rows else
                             ["image", "class_id", "class_name", "status"])
        wtr.writeheader()
        wtr.writerows(miss_rows)
    with (out_dir / "per_class_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(summary_rows)

    # ---- Overlays for representative misses (worst classes first) ----
    order = {"undetected": 0, "tta_only": 1, "threshold_miss": 2}
    picks: list[dict] = []
    seen_cls: Counter = Counter()
    for r in sorted(miss_rows, key=lambda r: (order[r["status"]], -r["box_px"])):
        if seen_cls[r["class_id"]] >= 2:
            continue
        picks.append(r)
        seen_cls[r["class_id"]] += 1
        if len(picks) >= args.max_overlays:
            break
    for j, r in enumerate(picks):
        entry = per_image[r["image"]]
        with Image.open(entry["path"]) as im:
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            # all GT of this class: red; low-conf preds of this class: orange
            for g in entry["gts"]:
                if g.cls == r["class_id"]:
                    d.rectangle([g.x1, g.y1, g.x2, g.y2], outline=(255, 0, 0), width=3)
            for p in entry["preds"]:
                if p.cls == r["class_id"]:
                    d.rectangle([p.x1, p.y1, p.x2, p.y2], outline=(255, 165, 0), width=2)
                    d.text((p.x1, max(0, p.y1 - 12)), f"{p.conf:.2f}", fill=(255, 165, 0))
            safe = r["class_name"].replace(" ", "_").replace("/", "-")[:40]
            im.save(out_dir / "overlays" / f"{j:02d}_{r['status']}_{safe}.jpg", quality=90)

    # ---- REPORT.md ----
    tot_gt = sum(c["gt"] for c in per_class.values())
    tot_det = sum(c["detected"] for c in per_class.values())
    tot_thr = sum(c["threshold_miss"] for c in per_class.values())
    tot_tta = sum(c["tta_only"] for c in per_class.values())
    tot_und = sum(c["undetected"] for c in per_class.values())

    lines = [
        "# Missed-Symbol Diagnostic Report",
        "",
        f"Model: `{model_path.name}` | dataset: `{dataset.name}` | split: `{args.split}` "
        f"({len(images)} images, {tot_gt} GT instances)",
        f"Production settings: conf={PROD_CONF}, imgsz={IMGSZ}; tiles {TILE_SIZE}px with {TILE_OVERLAP_PX}px overlap.",
        "",
        "## Overall",
        "",
        "| Bucket | Count | % of GT |",
        "|---|---|---|",
        f"| Detected at prod conf {PROD_CONF} | {tot_det} | {100*tot_det/tot_gt:.1f}% |",
        f"| Threshold miss (0.05-0.35, silently dropped) | {tot_thr} | {100*tot_thr/tot_gt:.1f}% |",
        f"| Recovered only by TTA | {tot_tta} | {100*tot_tta/tot_gt:.1f}% |",
        f"| Undetected even at 0.05 | {tot_und} | {100*tot_und/tot_gt:.1f}% |",
        "",
    ]
    if mismatches:
        lines += ["## WARNING: class-name mismatches (model vs data.yaml)", ""]
        lines += [f"- id {i}: yaml=`{y}` model=`{m}`" for i, y, m in mismatches]
        lines.append("")
    lines += [
        "## Per-class findings (classes with misses, worst recall first)",
        "",
        "| Class | GT | Train | R@0.35 | R@0.05 | +TTA | Diagnosis | Recommendation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted((s for s in summary_rows if s["diagnosis"]),
                    key=lambda s: (s["recall_at_0.35"] if s["recall_at_0.35"] != "" else 1.0)):
        lines.append(
            f"| {s['class_name']} | {s['gt_instances']} | {s['train_instances']} | "
            f"{s['recall_at_0.35']} | {s['recall_at_0.05']} | {s['recall_with_tta']} | "
            f"{s['diagnosis']} | {s['recommendation']} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `miss_details.csv` - every missed GT instance with status, conf, size, edge distance",
        "- `per_class_summary.csv` - full per-class table (including clean classes)",
        "- `overlays/` - GT (red) vs low-conf predictions (orange) for representative misses",
        "",
        "## Legend for diagnosis tags",
        "",
        "- **threshold**: model finds it below prod conf 0.35 - not a detection failure",
        "- **data_scarcity**: under 20 training instances - label more, don't retune",
        "- **tiny_p2_candidate**: below the P3 stride-8 floor at inference scale - yolov8-p2.yaml retrain",
        "- **tile_edge**: misses cluster near image edges - increase tile overlap above symbol size",
        "- **confused_class**: detected but as a different class - check labels for the confused pair",
        "- **hard_examples**: invisible even at conf 0.05 - inspect overlays for appearance drift",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # Console recap
    print(f"\nDone. GT={tot_gt}  detected@{PROD_CONF}={tot_det}  threshold={tot_thr}  "
          f"tta_only={tot_tta}  undetected={tot_und}")
    print(f"Report: {out_dir / 'REPORT.md'}")
    summary_json = {
        "model": str(model_path), "dataset": str(dataset), "split": args.split,
        "gt": tot_gt, "detected": tot_det, "threshold_miss": tot_thr,
        "tta_only": tot_tta, "undetected": tot_und,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
