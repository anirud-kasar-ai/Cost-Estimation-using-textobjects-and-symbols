# Setup — legend-ocr-count

OCR + YOLO: numbered symbol legend, circled callouts, and `model/symbol_detector_best.pt`.

## Prerequisites

- Python **3.10+**
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) on `PATH`, or set `TESSERACT_CMD` in `.env`
- YOLO weights at `model/symbol_detector_best.pt`

## 1. Create venv and install

```powershell
cd legend-ocr-count
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

macOS / Linux:

```bash
cd legend-ocr-count
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If Tesseract is not on PATH (Windows), edit `.env`:

```text
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 2. Run the UI

```powershell
uvicorn app:app --reload --reload-dir pipeline --reload-dir static --host 127.0.0.1 --port 8768
```

Open http://127.0.0.1:8768 — health should show **YOLO ready**.

## 3. What to upload

1. **Symbol legend** — image/PDF (descriptions should match YOLO class names for YOLO counts)
2. **Plan image** or **folder** of plans

**Count rule:** YOLO wins when class name matches a legend description; otherwise circled callout OCR is used.

## 4. YOLO tuning (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `YOLO_ENABLED` | true | Turn detector on/off |
| `YOLO_CONF` | 0.35 | Min confidence |
| `YOLO_IMGSZ` | 1280 | Inference size |
| `YOLO_PRIMARY_COUNTS` | true | Prefer YOLO over callout OCR when matched |

## 5. Smoke test

```powershell
$env:PYTHONPATH="."
python scripts\test_yolo_smoke.py
```

Jobs are written under `storage/jobs/` (gitignored). **Do not commit `.env`.**
