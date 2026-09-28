# Setup guide

Tracked tools: **dual-pathway-drawing-split** (PDF → zoom tiles), **symbol-count-vision** (CV + Gemini symbol counts), and **legend-ocr-count** (numbered legend + OCR callout counts).

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- Git
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for symbol-count-vision and legend-ocr-count)
- API keys: `GROQ_API_KEY` (drawing split); `GEMINI_API_KEY` (symbol count). Legend OCR needs no API key.

## 1. Clone

```bash
git lfs install
git clone https://github.com/anirud-kasar-ai/Cost-Estimation-using-textobjects-and-symbols.git
cd Cost-Estimation-using-textobjects-and-symbols
```

Bid drawing PDFs live in `Real data/` (stored with **Git LFS**). Use them as `--input` for dual-pathway-drawing-split.

## 2. Dual-pathway drawing split

Full steps: [`dual-pathway-drawing-split/SETUP.md`](dual-pathway-drawing-split/SETUP.md)

```bash
cd dual-pathway-drawing-split
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# Edit .env — set GROQ_API_KEY
python scripts/run_batch.py --input "../Real data"
```

Optional UI: `python app.py` → http://127.0.0.1:8766

## 3. Symbol count vision

Full steps: [`symbol-count-vision/SETUP.md`](symbol-count-vision/SETUP.md)

```bash
cd symbol-count-vision
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# Edit .env — set GEMINI_API_KEY
uvicorn app:app --reload --reload-dir pipeline --reload-dir static --host 127.0.0.1 --port 8767
```

Open http://127.0.0.1:8767

Upload a technical-symbol PDF plus either one plan tile or a zoom folder from:

```text
dual-pathway-drawing-split/storage/jobs/<job>/04_wings/<WING>/zooms/page_XXX/
```

**Do not commit `.env`.** Keys stay local; only `.env.example` is tracked.

## 4. Legend OCR callout count

Full steps: [`legend-ocr-count/SETUP.md`](legend-ocr-count/SETUP.md)

```bash
cd legend-ocr-count
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# Optional: set TESSERACT_CMD if tesseract is not on PATH
uvicorn app:app --reload --reload-dir pipeline --reload-dir static --host 127.0.0.1 --port 8768
```

Open http://127.0.0.1:8768

Upload a numbered SYMBOL LEGEND image/PDF plus a single plan image or a folder of plans. Counts are OCR callout detections matched to the legend (no YOLO).
