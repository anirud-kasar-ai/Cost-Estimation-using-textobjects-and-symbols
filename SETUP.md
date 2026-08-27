# Setup guide

Tracked tools: **drawing-zoom-split** and **dual-pathway-drawing-split** (PDF → zoom tiles) plus **symbol-count-vision** (symbol counts).

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- Git
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for symbol-count-vision)
- API keys: `GROQ_API_KEY` (drawing split); `GEMINI_API_KEY` and/or `GROQ_API_KEY` (symbol count)

## 1. Clone

```bash
git lfs install
git clone https://github.com/anirud-kasar-ai/Cost-Estimation-using-textobjects-and-symbols.git
cd Cost-Estimation-using-textobjects-and-symbols
```

Bid drawing PDFs live in `Real data/` (stored with **Git LFS**). Use them as `--input` for drawing-zoom-split.

## 2. Drawing zoom split

Full steps: [`drawing-zoom-split/SETUP.md`](drawing-zoom-split/SETUP.md)

```bash
cd drawing-zoom-split
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# Edit .env — set GROQ_API_KEY
python scripts/run_batch.py --input "../Real data"
```

Optional UI: `python app.py` → http://127.0.0.1:8766

The same drawing-split pipeline also lives in [`dual-pathway-drawing-split/`](dual-pathway-drawing-split/SETUP.md) (`python app.py` → http://127.0.0.1:8766). Copy `.env.example` to `.env` there as well.

## 3. Symbol count vision

Full steps: [`symbol-count-vision/SETUP.md`](symbol-count-vision/SETUP.md)

```bash
cd symbol-count-vision
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# Edit .env — set GEMINI_API_KEY or GROQ_API_KEY
uvicorn app:app --reload --reload-dir pipeline --reload-dir static --host 127.0.0.1 --port 8767
```

Open http://127.0.0.1:8767

Upload a technical-symbol PDF plus either one plan tile or a zoom folder from:

```text
drawing-zoom-split/storage/jobs/<job>/04_wings/<WING>/zooms/page_XXX/
```

**Do not commit `.env`.** Keys stay local; only `.env.example` is tracked.
