# Symbol Count Vision

Standalone FastAPI app that counts **technology-legend symbols** on CAD plan tiles.

Pipeline: detect (OCR + template glyphs + CV + optional Gemini/Groq vision) → map → NMS → evaluate → count. Prefer **per-tile / folder zoom** workflows — do not stitch the full wing first.

## Docs

- **[SETUP.md](SETUP.md)** — install, `.env`, run UI
- Companion zoom exporter: [`../dual-pathway-drawing-split`](../dual-pathway-drawing-split)

## Quick start

```powershell
cd symbol-count-vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env — set GEMINI_API_KEY or GROQ_API_KEY (never commit .env)
uvicorn app:app --reload --reload-dir pipeline --reload-dir static --host 127.0.0.1 --port 8767
```

Open http://127.0.0.1:8767

## Uploads

| Input | Description |
|-------|-------------|
| Symbol file | Technical-symbol PDF (preferred) or `symbol_table.json` |
| Plan image | Single zoom tile (`.jpg` / `.png` / `.webp`) |
| Zoom folder | All `plan_zoom_*.jpg` + `full_wing.jpg` + `zooms_manifest.json` from dual-pathway |

Outputs per job under `storage/jobs/<job_id>/`: `result.json`, `overlay.png`, extracted `07_glyphs/`.

## System requirement

Install **Tesseract OCR** and ensure `tesseract` is on PATH ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)). Optional: set `TESSERACT_CMD` in `.env`.

## Secrets

Copy `.env.example` → `.env` and add API keys locally. `.env` is gitignored — **do not commit keys**.
