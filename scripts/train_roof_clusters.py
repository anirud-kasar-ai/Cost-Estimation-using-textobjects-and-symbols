"""Fine-tune the roof detector on the cluster-augmented dataset.

Starts from the production weights so all prior learning is kept; the new
synthetic touching-symbol clusters teach it to separate connected symbols.

Usage (from repo root):  python scripts/train_roof_clusters.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model = YOLO(str(ROOT / "drawing-zoom-split" / "model" / "symbol_detector_best.pt"))
model.train(
    data=str(ROOT / "eval_data" / "train_small_clusters" / "data.yaml"),
    epochs=60,
    patience=20,
    imgsz=640,
    batch=8,
    device="cpu",
    seed=20260923,
    project=str(ROOT / "eval_data" / "cluster_runs"),
    name="roof_clusters",
    exist_ok=True,
)
print("BEST:", ROOT / "eval_data" / "cluster_runs" / "roof_clusters" / "weights" / "best.pt")
