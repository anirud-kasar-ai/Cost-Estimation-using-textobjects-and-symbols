# Legend OCR Callout Count

Standalone FastAPI tool that:

1. OCRs a numbered **SYMBOL LEGEND**
2. Finds **circled callout numbers** on the plan (OCR + circle detection)
3. Runs **YOLO** (`model/symbol_detector_best.pt`) for symbol detections
4. Matches YOLO class names to legend descriptions and shows **counts in the UI**

**Count rule:** if a legend row matches a YOLO class → use YOLO instance count; otherwise fall back to circled callout OCR.

## Setup

See [SETUP.md](SETUP.md).

```powershell
cd legend-ocr-count
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app:app --reload --host 127.0.0.1 --port 8768
```

Open http://127.0.0.1:8768

## Upload

| Input | What |
|-------|------|
| Symbol legend | Screenshot/PDF of the numbered legend |
| Plan image | Single roof plan / sheet |
| Folder | Multiple plan images (counts summed) |

Sample screenshots used during build: `samples/symbol_legend.png`, `samples/roof_plan.png`.

## Output

Per job under `storage/jobs/<job_id>/`:

- `result.json` — legend rows + counts + detection boxes
- `overlay.png` — callouts highlighted
- `plan_image.png` — working copy of the plan
