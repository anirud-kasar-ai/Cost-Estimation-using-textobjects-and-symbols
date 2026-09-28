"""Fine-tune the roof symbol detector on Google Colab (GPU).

In Colab:
  1. Runtime > Change runtime type > T4 GPU
  2. Upload roof_cluster_training.zip to the Files panel (or Drive)
  3. Run:
       !unzip -q roof_cluster_training.zip
       !pip -q install ultralytics
       !python roof_cluster_training/train_colab.py
  4. Download the result:
       runs/detect/roof_clusters/weights/best.pt
     and copy it to this project's drawing-zoom-split/model/ folder.
"""

from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

model = YOLO(str(HERE / "symbol_detector_best.pt"))
model.train(
    data=str(HERE / "data" / "data.yaml"),
    epochs=120,
    patience=30,
    imgsz=640,
    batch=16,
    device=0,          # Colab GPU
    seed=20260923,
    name="roof_clusters",
    exist_ok=True,
)

# Optional: if you also uploaded/unzipped roof_testing_data.zip, evaluate on it
test_yaml = HERE.parent / "roof_testing" / "data.yaml"
if test_yaml.is_file():
    metrics = model.val(data=str(test_yaml), split="test", device=0)
    print("test mAP50:", metrics.box.map50)

print("Trained weights: runs/detect/roof_clusters/weights/best.pt")
