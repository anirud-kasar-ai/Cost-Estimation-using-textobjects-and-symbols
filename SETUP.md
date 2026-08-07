# Setup guide

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Node.js 20+** and npm
- Git
- (Optional) [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) — not required; PyMuPDF is used as a fallback

## 1. Clone

```bash
git clone https://github.com/anirud-kasar-ai/Cost-Estimation-using-textobjects-and-symbols.git
cd Cost-Estimation-using-textobjects-and-symbols
```

## 2. Backend

```bash
cd hvac-cost-estimator
python -m venv .venv
```

Activate the virtualenv:

```bash
# Windows (PowerShell / cmd)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional heavy ML stack (real Detectron2 / PaddleOCR / ONNX — only if `HVAC_USE_MOCK_MODELS=false`):

```bash
pip install -r requirements-ml.txt
```

## 3. Environment file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` if needed. Defaults run the app in **mock model mode** (no trained weights required).

| Variable | Default | Purpose |
|----------|---------|---------|
| `HVAC_USE_MOCK_MODELS` | `true` | Mock detectors/OCR vs real models |
| `HVAC_PDF_DPI` | `120` | Render DPI for CV pages |
| `HVAC_MAX_PDF_PAGES` | `8` | Max pages rendered for device costing |

**Do not commit `.env`.** It is listed in `.gitignore`.

## 4. Frontend

```bash
cd frontend
npm install
cd ..
```

## 5. Run

**Terminal 1 — API** (http://localhost:8000):

```bash
cd hvac-cost-estimator\backend
..\..\hvac-cost-estimator\.venv\Scripts\activate
# if already at hvac-cost-estimator with venv active:
cd backend
uvicorn main:app --reload
```

From `hvac-cost-estimator/` with venv active:

```bash
cd backend
uvicorn main:app --reload
```

**Terminal 2 — UI** (http://localhost:5173):

```bash
cd hvac-cost-estimator/frontend
npm run dev
```

Open http://localhost:5173 and upload a drawing PDF. On upload, a requirement PDF is written under `backend/storage/requirements/`.

## 6. Sample PDF (optional)

```bash
cd hvac-cost-estimator/backend
python scripts/generate_sample_pdf.py storage/sample_layout.pdf
```

## 7. Tests (optional)

```bash
# Backend
cd hvac-cost-estimator/backend
pytest -q

# Frontend
cd hvac-cost-estimator/frontend
npm test
```

## One-shot Windows setup script

From the repo root you can also run:

```powershell
.\setup.ps1
```

This creates the venv, installs Python/Node deps, and copies `.env.example` → `.env` if missing.
