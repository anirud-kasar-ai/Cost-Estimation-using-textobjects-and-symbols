# Cost Estimation using Text, Objects, and Symbols

Local proof-of-concept that turns HVAC / construction bid drawing PDFs into:

1. **Requirement summary PDF** — provider, project details, and scope of work extracted from drawing text (and logo/stamp fallbacks when the firm name is graphic-only).
2. **Title-block metadata** — client, architect, engineer, address, dates (ROI + OCR + fuzzy mapping).
3. **Device symbol costing** — detect / classify HVAC symbols, count them, and produce an editable cost report.

The runnable application lives in [`hvac-cost-estimator/`](hvac-cost-estimator/).

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI + SQLite |
| ML pipeline | Mock models by default; optional Detectron2 / **PaddleOCR** / ONNX |
| Requirement extract | PyMuPDF text + heuristics (+ logo stamp map / PaddleOCR when mock is off) |
| UI | React + Vite + Tailwind |

## Quick start

See **[SETUP.md](SETUP.md)** for full Windows / macOS / Linux steps.

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

**Never commit `.env`.** Use `.env.example` as the template.

## Repository layout

```
├── README.md
├── SETUP.md
├── .gitignore
└── hvac-cost-estimator/
    ├── backend/          # FastAPI, ML pipeline, requirement extractor
    ├── frontend/         # React dashboard
    ├── docs/             # Architecture notes
    ├── requirements.txt
    ├── requirements-ml.txt
    └── .env.example
```

## License / notes

POC for local single-user use. Bid PDFs under `Real data/` are gitignored and not published with this repo.
