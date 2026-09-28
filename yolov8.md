# YOLOv8 Symbol Detector Training — Cell-by-Cell Guide

Short explanations for each cell in `YOLO_Symbol_Training_Colab (3).ipynb`.  
Run in Google Colab with **Runtime → Change runtime type → T4 GPU**.

---

## Cell 0 — Title & overview

Markdown intro. Explains that this notebook trains YOLOv8 on a legend-symbol dataset (images + `.txt` labels + `data.yaml`) uploaded as a zip.

**Expected zip layout:**
```
your_dataset.zip
├── data.yaml
├── train/images + train/labels
├── valid/ (or val/) images + labels
└── test/ (optional)
```

If only `train/` exists, the notebook later creates a validation split automatically.

---

## Cell 1 — Section header: Environment check

Markdown heading only. Starts Section 1.

---

## Cell 2 — Check GPU

```python
!nvidia-smi
```

Shows whether a GPU (e.g. T4) is available. If this fails or shows no GPU, switch runtime type before continuing.

---

## Cell 3 — Install Ultralytics

Installs the `ultralytics` package (YOLOv8) and runs a health check (`ultralytics.checks()`). Confirms PyTorch, CUDA, and YOLO are ready.

---

## Cell 4 — Section header: Upload dataset

Markdown heading only. Starts Section 2.

---

## Cell 5 — Upload zip

Opens Colab’s file picker. You select your dataset zip. Stores the uploaded filename in `zip_path`.

---

## Cell 6 — Unzip & show folder tree

Extracts the zip into `/content/dataset` and prints a short folder tree so you can see images, labels, and `data.yaml` landed correctly.

---

## Cell 7 — Section header: Locate data.yaml

Markdown. Explains that the next cells find `data.yaml` even if the zip has an extra nested folder, and handle both `val` and `valid` naming.

---

## Cell 8 — Find and read data.yaml

Searches recursively for `data.yaml` (or any `.yaml`). Saves path as `DATA_YAML_PATH` and dataset folder as `DATASET_ROOT`. Prints the original config so you can verify class names and paths.

---

## Cell 9 — Resolve train / val / test paths

Looks up real image folders from `data.yaml` (`train`, `val`/`valid`, `test`). Falls back to common folder names if paths in yaml are wrong. Prints resolved paths. Errors if train images cannot be found.

---

## Cell 10 — Section header: Auto validation split

Markdown. Explains Cell 11: if no val folder exists, create one.

---

## Cell 11 — Create val split if missing

If `val_path` is missing:
- Takes **15%** of train images (seeded random)
- Moves those images **and** matching label `.txt` files into a new `val/` folder

If val already exists, does nothing.

---

## Cell 12 — Write final training config

Builds a clean `dataset_config.yaml` with absolute paths for train/val/(test) and class names. Saves to `/content/dataset_config.yaml`. This is what `model.train()` will use.

---

## Cell 13 — Section header: Sanity checks

Markdown. Reminds you to check labels **before** a long training run.

---

## Cell 14 — Class distribution table + chart

Counts how many times each symbol class appears in train and val labels. Prints a table, draws a bar chart, and warns if any class has fewer than **20** training instances (those will usually perform poorly).

---

## Cell 15 — Visual label check

Draws ground-truth boxes on **4 random train images**. Use this to confirm boxes and class names look correct (not shifted, not wrong class).

---

## Cell 16 — Section header: Training configuration

Markdown notes for small symbols on drawings:
- Use high `imgsz` (1280) so tiny marks stay visible
- `yolov8s` fits well on T4 at that size
- Turn off vertical flip; keep horizontal flip
- On CUDA OOM: lower `BATCH` first, then `IMG_SIZE`

---

## Cell 17 — Set training hyperparameters

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL` | `yolov8s.pt` | Starting weights (n=fast, s=balanced, m=heavier) |
| `IMG_SIZE` | `1280` | Input resolution |
| `EPOCHS` | `150` | Max training epochs |
| `BATCH` | `16` | Images per step (lower if OOM) |
| `PATIENCE` | `30` | Early stop if val mAP stalls |
| `RUN_NAME` | `symbol_detector` | Output folder name under `/content/runs` |

---

## Cell 18 — Section header: Train

Markdown heading only. Starts Section 6.

---

## Cell 19 — Run training

Loads YOLO, then trains with settings tuned for small symbols:
- Cosine LR, auto optimizer
- Mosaic on, then off for last 10 epochs (`close_mosaic=10`)
- Mild rotation (`degrees=5`), horizontal flip only
- Light brightness/saturation jitter; no hue shift

Results go to `/content/runs/symbol_detector/`. Best weights: `weights/best.pt`.

---

## Cell 20 — Section header: Review curves

Markdown heading only. Starts Section 7.

---

## Cell 21 — Show training plots

Displays Ultralytics plots if present: `results.png`, confusion matrices, PR curve, F1 curve. Quick visual of how training progressed.

---

## Cell 22 — Section header: Evaluation metrics

Markdown. Explains Section 8: overall mAP is not enough — you also need per-class scores, confusion, and speed.

---

## Cell 23 — Overall val metrics

Loads `best.pt` and runs validation. Prints mean **precision**, **recall**, **mAP50**, and **mAP50-95**.

---

## Cell 24 — Section header: Per-class table

Markdown. Notes that precision/recall/F1 are at each class’s best F1 threshold, and the table is sorted worst → best by AP50.

---

## Cell 25 — Per-class metrics DataFrame

Builds a table: class name, val instance count, precision, recall, F1, AP50, AP50-95. Saves CSV to `/content/per_class_metrics.csv`. Warns about classes with very few val examples.

---

## Cell 26 — Per-class AP50 bar chart

Horizontal bar chart of AP50 per class (worst at top). Red line = overall mAP50.

---

## Cell 27 — Precision / recall / F1 chart

Grouped horizontal bars comparing precision, recall, and F1 for each class.

---

## Cell 28 — Section header: Confusion matrix

Markdown. Rows = predicted, columns = true. Extra **background** row/column = misses and false positives.

---

## Cell 29 — Plot confusion matrix

Draws the raw-count confusion matrix with labels. Use it to see which symbols get confused with which.

---

## Cell 30 — Section header: Inference speed

Markdown heading only.

---

## Cell 31 — Print speed (ms / FPS)

Prints preprocess / inference / postprocess time per image and approximate FPS on the validation set.

---

## Cell 32 — Section header: Optional test split

Markdown. If you have a held-out `test/` set, compare it to val to spot overfitting.

---

## Cell 33 — Evaluate on test (if present)

Runs validation on `test` when available and prints mAP vs val. Warns if the gap is large (>0.1 mAP50). Skips if no test split.

---

## Cell 34 — Download per-class CSV

Downloads `per_class_metrics.csv` to your computer.

---

## Cell 35 — Section header: Visual predictions

Markdown heading only. Starts Section 9.

---

## Cell 36 — Predict on 4 val images

Runs the best model on random val images (`conf=0.25`) and shows boxes. Quick sanity check that detections look right.

---

## Cell 37 — Section header: Save & download model

Markdown. Two downloads: weights only, and optional full run folder.

---

## Cell 38 — Download best weights

Copies `best.pt` (and `last.pt`) to `/content/exported_model/` and downloads **`symbol_detector_best.pt`**. This is the file you reuse for inference elsewhere.

---

## Cell 39 — Download full run zip (optional)

Zips the entire run folder (curves, args, both checkpoints) and downloads it for records.

---

## Cell 40 — How to use weights elsewhere

Markdown example:
```python
from ultralytics import YOLO
model = YOLO("symbol_detector_best.pt")
results = model.predict("some_image.jpg", imgsz=1280, conf=0.25)
```
No retraining needed — `best.pt` has architecture + weights.

---

## Cell 41 — Section header: Optional ONNX export

Markdown heading only.

---

## Cell 42 — Export ONNX

Exports the best model to ONNX for non-Python runtimes. Optional.

---

## Cell 43 — Section header: Test on real zoom tiles

Markdown. Explains uploading a **zoom-tile folder zip** (from your drawing-split stage):

```
your_zoom_folder.zip
├── plan_zoom_r00_c00.jpg
├── ...
├── full_wing.jpg          (optional — overlay canvas)
└── zooms_manifest.json    (optional — for cross-tile merge)
```

- **With manifest:** map boxes into shared ROI space, then class-aware NMS → count each symbol once across overlaps.
- **Without manifest:** sum per-tile counts (may double-count overlaps).

---

## Cell 44 — Upload & extract zoom zip

Uploads the zoom folder zip, extracts to `/content/zoom_test`, finds tile images, optional `full_wing`, and optional manifest. Prints what was found.

---

## Cell 45 — Detection / merge settings + load manifest

Sets:
- `CONF_THRESHOLD = 0.25`
- `MERGE_IOU_THRESHOLD = 0.3`

Loads `zooms_manifest.json` (flexible key names for filename and bbox). If no usable manifest, later cells skip merge.

---

## Cell 46 — Run detection on every tile

Predicts on each tile with the best model. Stores boxes (class, score, xyxy) per tile. Prints raw detection count.

---

## Cell 47 — Merge across tiles (NMS)

**If manifest exists:** maps each tile box into full-wing ROI coordinates, then runs **class-aware NMS** so overlap duplicates become one detection.

**If no manifest:** concatenates all tile detections (no merge).

---

## Cell 48 — Print & save symbol counts

Counts detections per class, prints a table, saves `/content/symbol_counts.json`.

---

## Cell 49 — Section header: Overlay

Markdown. Overlay on `full_wing.jpg` if available; otherwise show a grid of tiles.

---

## Cell 50 — Draw overlay or tile grid

- With manifest + `full_wing`: draws merged boxes on the full wing image and saves `full_wing_overlay.jpg`.
- Otherwise: shows up to 9 tiles with local predictions.

---

## Cell 51 — Download counts (+ overlay)

Downloads `symbol_counts.json`, and the overlay image if it was created.

---

## Cell 52 — Dataset-specific notes

Markdown tips:
1. Rare classes (<~20 examples) stay weak — add more labels, don’t just train longer.
2. Don’t use YOLO to read digits/text — detect symbols only; use OCR/PDF text for numbers.
3. Train on tiles; at full-page inference, tile first then merge (don’t run YOLO on a tiny full sheet).
4. Change `RUN_NAME` when trying new configs and compare `results.csv`.

---

## Quick run order

| Step | Cells | What you do |
|------|-------|-------------|
| 1 | 2–3 | Confirm GPU + install YOLO |
| 2 | 5–12 | Upload dataset, fix paths, create val if needed |
| 3 | 14–15 | Check class counts and label boxes |
| 4 | 17, 19 | Set params and train |
| 5 | 21–36 | Review metrics and sample predictions |
| 6 | 38–39 | Download `best.pt` (and optional full run) |
| 7 | 44–51 | Optional: test on real zoom tiles and get counts |

---

## Files you care about after training

| File | Purpose |
|------|---------|
| `symbol_detector_best.pt` | Trained model for local / pipeline use |
| `per_class_metrics.csv` | Per-symbol precision / recall / AP |
| `symbol_counts.json` | Counts from a zoom-folder test |
| `full_wing_overlay.jpg` | Visual of merged detections (if tested with manifest) |
