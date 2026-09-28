"""Generate overall model metrics + business metrics for the YOLOv8 symbol detector.

Runs validation on the trained model and produces:
  1. Overall model metrics (the official YOLO "all" row): precision, recall,
     mAP50, mAP50-95 -- macro-averaged over classes at a single global threshold.
  2. Business metrics from the confusion matrix (exact object counts):
     total ground-truth objects, objects correctly identified, misclassified,
     missed, false detections, and object-level precision / recall / F1.
  3. Per-class CSV (per_class_metrics.csv) using the SAME global threshold as
     the overall row, so per-class numbers aggregate consistently.
  4. business_metrics.json with everything above.

Usage (Colab):
    !python generate_business_metrics.py \
        --model /content/runs/detect/symbol_detector/weights/best.pt \
        --data  /content/dataset/data.yaml

Usage (local):
    python scripts/generate_business_metrics.py --model best.pt --data data.yaml
"""

import argparse
import json
from pathlib import Path

# NOTE: ultralytics (torch) must be imported BEFORE pandas on this machine,
# otherwise torch's c10.dll fails to load (WinError 1114).
from ultralytics import YOLO

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280, help="Val image size (match training)")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold for business object counts (confusion matrix)")
    ap.add_argument("--outdir", default=".", help="Where to write the CSV/JSON outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Pass 1 -- official metrics: low conf so the PR curve / mAP are correct.
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                        conf=0.001, iou=0.7, plots=True, verbose=False)

    names = model.names  # {idx: class_name}
    nc = len(names)

    # ------------------------------------------------------------------
    # 1. Overall model metrics (identical to YOLO's printed "all" row)
    # ------------------------------------------------------------------
    mp, mr = metrics.box.mp, metrics.box.mr          # mean precision / recall
    map50, map5095 = metrics.box.map50, metrics.box.map
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. Business metrics from the confusion matrix (exact object counts).
    #    Pass 2 -- re-validate at the production confidence threshold so the
    #    confusion matrix reflects real deployment behaviour (Ultralytics
    #    builds the matrix at whatever conf the validator ran with).
    #    matrix[pred_class, true_class]; last index = background.
    # ------------------------------------------------------------------
    # plots=True is required: ultralytics only fills the confusion matrix
    # when plotting is enabled.
    metrics_biz = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                            conf=args.conf, iou=0.7, plots=True, verbose=False)
    cm = metrics_biz.confusion_matrix.matrix  # shape (nc+1, nc+1)
    core = cm[:nc, :nc]

    correctly_identified = int(np.trace(core))            # right box, right class
    misclassified = int(core.sum() - np.trace(core))       # found, but wrong class
    missed = int(cm[nc, :nc].sum())                        # GT object, no detection
    false_detections = int(cm[:nc, nc].sum())              # detection, no GT object

    total_gt = correctly_identified + misclassified + missed
    total_pred = correctly_identified + misclassified + false_detections

    biz_precision = correctly_identified / total_pred if total_pred else 0.0
    biz_recall = correctly_identified / total_gt if total_gt else 0.0
    biz_f1 = (2 * biz_precision * biz_recall / (biz_precision + biz_recall)
              if (biz_precision + biz_recall) > 0 else 0.0)
    # "Localization" view: object found at all, even if label was wrong
    found_any = correctly_identified + misclassified
    detection_rate = found_any / total_gt if total_gt else 0.0

    # ------------------------------------------------------------------
    # 3. Per-class table at the global threshold (consistent aggregation)
    # ------------------------------------------------------------------
    idx = metrics.box.ap_class_index  # classes that appear in this split
    rows = []
    for i, ci in enumerate(idx):
        gt_count = int(cm[:, ci].sum())            # GT instances of this class seen by CM
        tp_c = int(cm[ci, ci])
        rows.append({
            "class": names[int(ci)],
            "gt_instances": gt_count,
            "objects_identified": tp_c,
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "f1": float(metrics.box.f1[i]),
            "AP50": float(metrics.box.ap50[i]),
            "AP50-95": float(metrics.box.ap[i]),
        })
    df = pd.DataFrame(rows).sort_values("AP50").reset_index(drop=True)
    csv_path = outdir / "per_class_metrics.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # 4. Print + save the report
    # ------------------------------------------------------------------
    report = {
        "overall_model_metrics": {
            "precision": round(float(mp), 4),
            "recall": round(float(mr), 4),
            "f1": round(float(f1), 4),
            "mAP50": round(float(map50), 4),
            "mAP50-95": round(float(map5095), 4),
            "classes": nc,
        },
        "business_metrics": {
            "total_actual_objects": total_gt,
            "objects_correctly_identified": correctly_identified,
            "objects_found_but_misclassified": misclassified,
            "objects_missed": missed,
            "total_detections_made": total_pred,
            "false_detections": false_detections,
            "object_precision": round(biz_precision, 4),
            "object_recall": round(biz_recall, 4),
            "object_f1": round(biz_f1, 4),
            "detection_rate_any_class": round(detection_rate, 4),
            "confidence_threshold": args.conf,
        },
    }
    json_path = outdir / "business_metrics.json"
    json_path.write_text(json.dumps(report, indent=2))

    o, b = report["overall_model_metrics"], report["business_metrics"]
    print("\n" + "=" * 62)
    print("OVERALL MODEL METRICS (YOLO 'all' row, macro over classes)")
    print("=" * 62)
    print(f"  Precision : {o['precision']:.1%}")
    print(f"  Recall    : {o['recall']:.1%}")
    print(f"  F1 score  : {o['f1']:.1%}")
    print(f"  mAP@50    : {o['mAP50']:.1%}")
    print(f"  mAP@50-95 : {o['mAP50-95']:.1%}")

    print("\n" + "=" * 62)
    print(f"BUSINESS METRICS (object counts @ conf={args.conf})")
    print("=" * 62)
    print(f"  Actual objects on drawings      : {b['total_actual_objects']}")
    print(f"  Correctly identified            : {b['objects_correctly_identified']}"
          f"  ({b['object_recall']:.1%} recall)")
    print(f"  Found but wrong class           : {b['objects_found_but_misclassified']}")
    print(f"  Missed completely               : {b['objects_missed']}")
    print(f"  Total detections made           : {b['total_detections_made']}")
    print(f"  False detections                : {b['false_detections']}"
          f"  (precision {b['object_precision']:.1%})")
    print(f"  Object-level F1                 : {b['object_f1']:.1%}")
    print(f"  Found at all (any class)        : {b['detection_rate_any_class']:.1%}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
