# Setup — symbol-count-vision

Standalone FastAPI UI that counts technology-legend symbols on plan tiles or zoom folders (CV + optional Gemini/Groq vision).

## Prerequisites

- Python **3.10+** (3.11 recommended)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) on `PATH` (or set `TESSERACT_CMD` in `.env`)
- API key for **Gemini** and/or **Groq** if vision LLM features are enabled

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

Edit `.env` and set **only** the keys you use:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google AI Studio / Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `LLM_PROVIDER` | `gemini` or `groq` |
| `GEMINI_MODEL` | Detection/count model. `gemini-2.5-flash` is requested first; this API often returns 404 (retired for new users). There is no Gemini 2.6. Set `GEMINI_FALLBACK_MODEL=gemini-3.5-flash,gemini-3.6-flash`. |
| `GEMINI_FALLBACK_MODEL` | Comma-separated models tried on 404/429 |
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
drawing-zoom-split/storage/jobs/<job>/04_wings/<WING>/zooms/page_XXX/
```

Jobs are written under `storage/jobs/<job_id>/` (ignored by git).

## 5. Tests (optional)

```powershell
pytest -q
```
