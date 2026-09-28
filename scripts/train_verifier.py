"""Train the detection verification classifier and pick its drop threshold.

Trains yolov8n-cls on eval_data/verifier (true_symbol vs false_symbol crops),
then sweeps P(false_symbol) thresholds on the held-out val split and reports,
for each, how many false positives are caught vs true symbols lost. Copies
the best weights to drawing-zoom-split/model/verifier/detection_verifier.pt
(a subfolder, so the detector's *.pt auto-discovery never picks it up).

Usage (from repo root):
    python scripts/train_verifier.py [--epochs 30]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval_data" / "verifier"
DEST = ROOT / "drawing-zoom-split" / "model" / "verifier" / "detection_verifier.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=64)
    args = ap.parse_args()

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=64,
        project=str(ROOT / "eval_data" / "verifier_runs"),
        name="verifier",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, DEST)
    print(f"weights: {DEST}")

    # ------------------------------------------------------------------
    # Threshold sweep on the held-out val split
    # ------------------------------------------------------------------
    clf = YOLO(str(DEST))
    names = clf.names  # {0: 'false_symbol', 1: 'true_symbol'} (alphabetical)
    false_idx = next(i for i, n in names.items() if n == "false_symbol")

    def p_false(img: Path) -> float:
        r = clf.predict(source=str(img), imgsz=args.imgsz, verbose=False)
        return float(r[0].probs.data[false_idx])

    scores = {"true_symbol": [], "false_symbol": []}
    for label in scores:
        for img in sorted((DATA / "val" / label).glob("*.jpg")):
            scores[label].append(p_false(img))

    n_tp, n_fp = len(scores["true_symbol"]), len(scores["false_symbol"])
    print(f"\nval crops: {n_tp} true, {n_fp} false")
    print(f"{'drop if P(false)>=':>20}{'FPs caught':>12}{'TPs lost':>10}")
    sweep = {}
    for th in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        fp_caught = sum(1 for s in scores["false_symbol"] if s >= th)
        tp_lost = sum(1 for s in scores["true_symbol"] if s >= th)
        sweep[th] = {
            "fp_caught": fp_caught,
            "fp_caught_pct": round(fp_caught / n_fp * 100, 1) if n_fp else 0,
            "tp_lost": tp_lost,
            "tp_lost_pct": round(tp_lost / n_tp * 100, 1) if n_tp else 0,
        }
        print(
            f"{th:>20.2f}{fp_caught:>8} ({sweep[th]['fp_caught_pct']:>4.1f}%)"
            f"{tp_lost:>6} ({sweep[th]['tp_lost_pct']:>4.1f}%)"
        )

    out = ROOT / "eval_data" / "report_misses" / "verifier_sweep.json"
    out.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(f"\nsweep saved: {out}")


if __name__ == "__main__":
    main()
