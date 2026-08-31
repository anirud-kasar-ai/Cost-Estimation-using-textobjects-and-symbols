# Setup — drawing-zoom-split

Standalone pipeline that turns bid drawing PDFs into cropped wing images and **588px zoom tiles** (10% overlap). Symbol counting is a separate tool (`../symbol-count-vision`).

## Prerequisites

- Python **3.10+** (3.11 recommended)
- **GROQ_API_KEY** (default vision runtime) — or Hugging Face token if using `VISION_LLM_RUNTIME=hf`
- Enough disk space under `storage/jobs/` for rendered pages and zooms

## 1. Create venv and install

```powershell
cd "drawing-zoom-split"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd drawing-zoom-split
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` includes optional local torch / transformers stacks. For Groq-only cloud vision you still need the package set listed there; GPU is not required when `VISION_LLM_RUNTIME=groq`.

## 2. Environment file

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Required for default Groq vision |
| `GROQ_MODEL_ID` | Default `qwen/qwen3.6-27b` |
| `ROI_ZOOM_TILE_SIZE` | Default `588` (10% closer than 653px; matches symbol-count-vision) |
| `ROI_ZOOM_OVERLAP_PCT` | Default `0.10` |

**Never commit `.env`.** It is listed in `.gitignore`.

## 3. Run batch (CLI)

```powershell
python scripts/run_batch.py --input "path\to\pdfs"
```

Optional:

```powershell
python scripts/run_batch.py --input "path\to\pdfs" --output storage/jobs
python scripts/run_batch.py --input "path\to\pdfs" --verify
```

## 4. Run UI (optional)

```powershell
python app.py
```

Open http://127.0.0.1:8766

## 5. Hand off to symbol counting

After a job finishes, use a zoom folder as input to **symbol-count-vision**:

```text
storage/jobs/<pdf_stem>/04_wings/<WING>/zooms/page_NNN/
```

Include `plan_zoom_*.jpg`, `full_wing.jpg`, and `zooms_manifest.json`.

## 6. Verify (optional)

```powershell
python scripts/verify_output.py
python scripts/verify_realdata_roi_zoom.py
```
