# Setup — symbol-count-vision

Standalone FastAPI UI that counts technology-legend symbols on plan tiles or zoom folders (CV + Gemini vision).

## Prerequisites

- Python **3.10+** (3.11 recommended)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) on `PATH` (or set `TESSERACT_CMD` in `.env`)
- API key for **Gemini** if vision LLM features are enabled

## 1. Create venv and install

```powershell
cd "symbol-count-vision"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd symbol-count-vision
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Environment file

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

Edit `.env` and set:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google AI Studio / Gemini API key (required for classify + count-verify) |
| `GEMINI_MODEL` | Primary vision model. Default `gemini-3.7-flash`. |
| `GEMINI_FALLBACK_MODEL` | Tried only on HTTP 404. Default `gemini-3.5-flash`. |
| `VISION_CLASSIFY_MAX_PER_TILE` | Gemini Pro labels for ambiguous crops (camera vs pole vs `#`). Default `4`. `0` = CV only. |
| `MERGE_EVAL_MAX_JUDGE_CALLS` | Post-merge Pro checks on cameras/poles. Default `16`. |
| `VISION_DETECT_MAX_REFS` | Max legend glyph crops attached to classify and count-verify |
| `TESSERACT_CMD` | Full path to `tesseract.exe` if not on PATH |

**Never commit `.env`.** It is listed in `.gitignore`.

## 3. Run the UI

```powershell
uvicorn app:app --reload --reload-dir pipeline --reload-dir static --host 127.0.0.1 --port 8767
```

Open http://127.0.0.1:8767

## 4. What to upload

1. **Symbol file** — technical-symbol PDF (or `symbol_table.json`)
2. Either:
   - **Single plan image** — one `plan_zoom_*.jpg` / PNG, or
   - **Zoom folder** — from dual-pathway: all `plan_zoom_*.jpg`, `full_wing.jpg`, and `zooms_manifest.json`

Typical zoom folder path after dual-pathway processing:

```text
dual-pathway-drawing-split/storage/jobs/<job>/04_wings/<WING>/zooms/page_XXX/
```

Jobs are written under `storage/jobs/<job_id>/` (ignored by git).

## 5. Tests (optional)

```powershell
pytest -q
```
