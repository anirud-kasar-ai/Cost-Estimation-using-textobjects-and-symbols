# Cost Estimation using Text, Objects, and Symbols

Local proof-of-concept that turns HVAC / construction bid drawing PDFs into:

1. **Requirement summary PDF** — provider, project details, and scope of work extracted from drawing text (and logo/stamp fallbacks when the firm name is graphic-only).
2. **Title-block metadata** — client, architect, engineer, address, dates (ROI + OCR + fuzzy mapping).
3. **Device symbol costing** — detect / classify HVAC symbols, count them, and produce an editable cost report.

The full HVAC dashboard lives in [`hvac-cost-estimator/`](hvac-cost-estimator/). Two standalone drawing tools (zoom export + symbol count) are also in this repo:

| Folder | Role | Setup |
|--------|------|--------|
| [`dual-pathway-drawing-split/`](dual-pathway-drawing-split/) | PDF → wing crops + 653px zoom tiles | [SETUP](dual-pathway-drawing-split/SETUP.md) |
| [`symbol-count-vision/`](symbol-count-vision/) | Count legend symbols on tiles / zoom folders | [SETUP](symbol-count-vision/SETUP.md) |

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI + SQLite |
| ML pipeline | Mock models by default; optional Detectron2 / **PaddleOCR** / ONNX |
| Requirement extract | PyMuPDF text + heuristics (+ logo stamp map / PaddleOCR when mock is off) |
| Drawing split / zooms | OpenCV + Groq (or HF) vision |
| Symbol count | Tesseract + OpenCV + optional Gemini/Groq vision |
| UI | React + Vite + Tailwind (HVAC app); lightweight FastAPI UIs for the two tools above |

## Quick start

See **[SETUP.md](SETUP.md)** for full Windows / macOS / Linux steps (HVAC app + the two drawing folders).

### HVAC cost estimator

```bash
cd hvac-cost-estimator
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux

cd frontend && npm install && cd ..

# Terminal 1 — API
cd backend && uvicorn main:app --reload

# Terminal 2 — UI
cd frontend && npm run dev
```

- Dashboard: http://localhost:5173  
- API docs: http://localhost:8000/docs  

### Drawing split + symbol count

```bash
# 1) Export zooms from a bid PDF
cd dual-pathway-drawing-split
copy .env.example .env   # set GROQ_API_KEY
pip install -r requirements.txt
python scripts/run_batch.py --input "path\to\pdfs"

# 2) Count symbols on a zoom folder or single tile
cd ../symbol-count-vision
copy .env.example .env   # set GEMINI_API_KEY or GROQ_API_KEY
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8767
```

**Never commit `.env`.** Use each folder’s `.env.example` as the template. API keys are gitignored at the repo root and in both tool folders.

## Repository layout

```
├── README.md
├── SETUP.md
├── .gitignore
├── dual-pathway-drawing-split/   # PDF → wings + zoom tiles
├── symbol-count-vision/          # Legend symbol counting UI
└── hvac-cost-estimator/
    ├── backend/          # FastAPI, ML pipeline, requirement extractor
    ├── frontend/         # React dashboard
    ├── docs/             # Architecture notes
    ├── requirements.txt
    ├── requirements-ml.txt
    └── .env.example
```

## License / notes

POC for local single-user use.

Sample bid drawings live in [`Real data/`](Real%20data/) (stored with **Git LFS**). Generated requirement PDFs are under `Real data/requirements/` and `hvac-cost-estimator/backend/storage/requirements/`.

```bash
git lfs install
git clone https://github.com/anirud-kasar-ai/Cost-Estimation-using-textobjects-and-symbols.git
```
