# Mechanical Device Cost Estimator (i-TAB HVAC POC)

Turn a static HVAC layout PDF into an itemized, editable costing report:

1. **Title-block metadata** (client, architect, engineer, address, due date) is
   extracted via ROI detection + OCR + fuzzy field mapping.
2. **HVAC device symbols** (diffusers, grilles, sensors, thermostats, VAV boxes)
   are detected, classified, and counted.
3. Counts are joined against a unit-cost rate table to produce a **costed
   line-item report** with a grand total — reviewable and editable in a web
   dashboard, exportable as CSV.

This is a **local proof of concept**: no containers, no cloud services, single
user, single machine. By default it runs with **deterministic mock model
outputs** (`HVAC_USE_MOCK_MODELS=true`), so the entire pipeline, API, and
dashboard work end-to-end without trained weights or heavy ML dependencies.
Real Detectron2 / PaddleOCR / ONNX backends are implemented behind the same
interfaces and activate once weights exist (see "Using real models" below).

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 20+
- (Optional) [poppler](https://github.com/oschwartz10612/poppler-windows/releases)
  for `pdf2image`. **Not required** — if poppler is missing, PDF rendering
  automatically falls back to PyMuPDF, which installs via pip.

## Setup

From the repository root (`hvac-cost-estimator/`):

```bash
# 1. Backend: create the venv and install base dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows (use `source .venv/bin/activate` elsewhere)
pip install -r requirements.txt

# 2. Frontend
cd frontend
npm install
cd ..

# 3. (Optional) configuration — defaults work out of the box
copy .env.example .env          # then edit as needed
```

## Run

Two terminals:

```bash
# Terminal 1 — backend API on http://localhost:8000
cd backend
..\.venv\Scripts\activate
uvicorn main:app --reload

# Terminal 2 — dashboard on http://localhost:5173
cd frontend
npm run dev
```

Open http://localhost:5173, drop in a layout PDF, and watch the report appear.
Interactive API docs are at http://localhost:8000/docs.

No HVAC drawing at hand? Generate a synthetic one:

```bash
cd backend
python scripts\generate_sample_pdf.py storage\sample_layout.pdf
```

### Dashboard features

- Drag-and-drop PDF upload with validation (type, corruption) and live
  processing status.
- **Requirement extract**: on upload, provider/contact details and scope of work
  are parsed from the PDF text and saved as
  `storage/requirements/<filename> requirement.pdf` (downloadable from the UI).
- Extracted project metadata panel.
- Editable costing grid: override any count or unit cost — totals recalculate
  server-side, and a "reset" link restores the detected/default value.
- CSV export of the full report (metadata + line items + grand total).

### Batch-extract requirements from a folder of bid PDFs

```bash
cd backend
python scripts\extract_requirements.py "..\..\Real data"
# writes Real data\requirements\<name> requirement.pdf for each input
```

> **Schema note (POC):** if you already ran an older build, new columns are
> added automatically on startup via a best-effort SQLite `ALTER TABLE`. If
> anything looks off, delete `backend/storage/hvac.db` and restart.

## Tests

```bash
# Backend (52 tests, no model weights needed)
cd backend
..\.venv\Scripts\python.exe -m pytest

# Frontend (Vitest + Testing Library)
cd frontend
npm test
```

## Configuration

All settings load from `.env` (prefix `HVAC_`) via pydantic-settings — see
[.env.example](.env.example) for the full list. Highlights:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HVAC_USE_MOCK_MODELS` | `true` | Mock vs real model backends |
| `HVAC_PDF_DPI` | `300` | PDF page render resolution |
| `HVAC_MAX_UPLOAD_MB` | unset (no limit) | Optional upload size cap in MB |
| `HVAC_MAX_PDF_PAGES` | `40` | Max pages rendered for CV; requirement text always uses the full PDF |
| `HVAC_POPPLER_PATH` | unset | Poppler `bin` dir (Windows, optional) |
| `HVAC_COST_RATES_PATH` | `backend/data/cost_rates.json` | Device rate table |

Unit costs live in [backend/data/cost_rates.json](backend/data/cost_rates.json).
Device types missing from the table get `default_unit_cost` and are flagged
"needs review" in the dashboard.

## Using real models (when trained weights exist)

1. Install the heavy ML dependencies: `pip install -r requirements-ml.txt`
   (detectron2 must be built from source — see notes inside that file).
2. Drop weights into `backend/models/`:
   - `roi_detector.pth` + `roi_detector.yaml` (Detectron2 title-block model)
   - `symbol_detector.pth` + `symbol_detector.yaml` (Detectron2 symbol model)
   - `device_classifier.onnx` (exported via `scripts/export_model.py`)
3. Set `HVAC_USE_MOCK_MODELS=false` in `.env` and restart the backend.

PaddleOCR needs no custom weights — it downloads its defaults on first use.

Training entrypoints (`backend/scripts/train_roi_detector.py`,
`train_symbol_detector.py`, `export_model.py`) are documented stubs: they
describe the required annotated data (Label Studio → COCO export into
`ml-training/data/`) and exit with instructions until that data exists.

## Repository layout

```
backend/          FastAPI app, ML pipeline modules, SQLite, tests
  ml/             pipeline stages (each with mock + real backends)
  api/routes/     upload, projects CRUD + overrides, report/CSV endpoints
  data/           cost_rates.json seed rate table
  models/         trained weights go here (gitignored)
  storage/        uploaded PDFs, page images, SQLite DB (gitignored)
  scripts/        sample PDF generator + training/export stubs
frontend/         React 18 + TypeScript + Vite + Tailwind dashboard
ml-training/      annotated data, notebooks, Detectron2 configs (future)
docs/             ARCHITECTURE.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline design and
the mock/real model swap mechanism.
