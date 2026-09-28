"""Build docs/model_metrics_report.xlsx (+ CSV) from the saved evaluation outputs.

Reads business_metrics.json and per_class_metrics.csv from
eval_data/report_main and eval_data/report_small, computes the combined
(overall) business metrics, and writes:
  - docs/model_metrics_report.xlsx  (4 sheets)
  - docs/overall_business_metrics.csv
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Cost Estimation Using Text and Object")
MAIN = ROOT / "eval_data" / "report_main"
SMALL = ROOT / "eval_data" / "report_small"
DOCS = ROOT / "docs"

main = json.loads((MAIN / "business_metrics.json").read_text())
small = json.loads((SMALL / "business_metrics.json").read_text())

mb, sb = main["business_metrics"], small["business_metrics"]
mo, so = main["overall_model_metrics"], small["overall_model_metrics"]


def combine(key):
    return mb[key] + sb[key]


tot_gt = combine("total_actual_objects")
tot_tp = combine("objects_correctly_identified")
tot_wrong = combine("objects_found_but_misclassified")
tot_miss = combine("objects_missed")
tot_pred = combine("total_detections_made")
tot_fp = combine("false_detections")
prec = tot_tp / tot_pred
rec = tot_tp / tot_gt
f1 = 2 * prec * rec / (prec + rec)
found_any = (tot_tp + tot_wrong) / tot_gt

# instance-weighted overall model metrics
w_m, w_s = mb["total_actual_objects"], sb["total_actual_objects"]


def wavg(key):
    return (mo[key] * w_m + so[key] * w_s) / (w_m + w_s)


# ---- Sheet 1: Overall business metrics ----
overall = pd.DataFrame([
    ["Total actual objects on test drawings", tot_gt],
    ["Objects correctly identified", tot_tp],
    ["Overall recall (objects found)", f"{rec:.1%}"],
    ["Objects found but wrong symbol type", tot_wrong],
    ["Objects missed completely", tot_miss],
    ["Total detections made by the system", tot_pred],
    ["False detections", tot_fp],
    ["Overall precision (correct detections)", f"{prec:.1%}"],
    ["Overall F1 score", f"{f1:.1%}"],
    ["Objects located at all (even if wrong class)", f"{found_any:.1%}"],
    ["", ""],
    ["Weighted model precision", f"{wavg('precision'):.1%}"],
    ["Weighted model recall", f"{wavg('recall'):.1%}"],
    ["Weighted mAP@50", f"{wavg('mAP50'):.1%}"],
    ["Weighted mAP@50-95", f"{wavg('mAP50-95'):.1%}"],
], columns=["Business Metric", "Value"])

# ---- Sheet 2: Per-model summary ----
def model_row(label, o, b):
    return {
        "Model": label,
        "Classes": o["classes"],
        "Actual objects": b["total_actual_objects"],
        "Correctly identified": b["objects_correctly_identified"],
        "Object recall": round(b["object_recall"], 4),
        "Wrong class": b["objects_found_but_misclassified"],
        "Missed": b["objects_missed"],
        "Total detections": b["total_detections_made"],
        "False detections": b["false_detections"],
        "Object precision": round(b["object_precision"], 4),
        "Model precision (best-F1)": round(o["precision"], 4),
        "Model recall (best-F1)": round(o["recall"], 4),
        "mAP@50": round(o["mAP50"], 4),
        "mAP@50-95": round(o["mAP50-95"], 4),
    }


per_model = pd.DataFrame([
    model_row("Main symbol detector (52 classes)", mo, mb),
    model_row("Roof detail detector (9 classes)", so, sb),
    {
        "Model": "COMBINED (overall)",
        "Classes": mo["classes"] + so["classes"],
        "Actual objects": tot_gt,
        "Correctly identified": tot_tp,
        "Object recall": round(rec, 4),
        "Wrong class": tot_wrong,
        "Missed": tot_miss,
        "Total detections": tot_pred,
        "False detections": tot_fp,
        "Object precision": round(prec, 4),
        "Model precision (best-F1)": round(wavg("precision"), 4),
        "Model recall (best-F1)": round(wavg("recall"), 4),
        "mAP@50": round(wavg("mAP50"), 4),
        "mAP@50-95": round(wavg("mAP50-95"), 4),
    },
])

# ---- Sheets 3 & 4: per-class tables ----
pc_main = pd.read_csv(MAIN / "per_class_metrics.csv")
pc_small = pd.read_csv(SMALL / "per_class_metrics.csv")

xlsx = DOCS / "model_metrics_report.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    overall.to_excel(xw, sheet_name="Overall Business Metrics", index=False)
    per_model.to_excel(xw, sheet_name="Per-Model Summary", index=False)
    pc_main.to_excel(xw, sheet_name="Per-Class Main (52c)", index=False)
    pc_small.to_excel(xw, sheet_name="Per-Class Roof (9c)", index=False)
    # widen columns for readability
    for ws in xw.book.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) for c in col if c.value is not None)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 48)

csv_path = DOCS / "overall_business_metrics.csv"
per_model.to_csv(csv_path, index=False)

print("Saved:", xlsx)
print("Saved:", csv_path)
