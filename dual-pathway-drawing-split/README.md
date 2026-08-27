# Dual-Pathway Drawing Split

Batch-process construction/architecture PDFs: extract plan pages, crop diagrams, split into per-wing images, and export **653px zoom tiles** (10% overlap) for symbol counting.

Standalone project — does not modify `vision-extraction` or `hvac-cost-estimator`.

## Docs

- **[SETUP.md](SETUP.md)** — install, `.env`, CLI + UI
- Downstream counter: [`../symbol-count-vision`](../symbol-count-vision)
- [PLAN.md](PLAN.md) — implementation plan
- [docs/architecture.md](docs/architecture.md) — dual-pathway design
- [docs/requirements.md](docs/requirements.md) — functional requirements
- [docs/output-layout.md](docs/output-layout.md) — full output contract

## What it does

1. Render PDF pages and keep sheets with plan/diagram content.
2. Crop the drawing area (remove title block, notes, legend).
3. Split each diagram by wing/building name (e.g. `B-WING-EAST`).
4. Store wing JPEGs and zoom tiles under `04_wings/{WING-NAME}/`.

Uses a **dual-pathway** design: PDF text layer + vision LLM for wing anchors; OpenCV for crop boundaries.

## Quick start

```powershell
cd dual-pathway-drawing-split
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env — set GROQ_API_KEY (never commit .env)
python scripts/run_batch.py --input "path\to\pdfs"
```

Optional UI: `python app.py` → http://127.0.0.1:8766

## Output layout

```text
storage/jobs/{pdf_stem}/
  01_source/{original.pdf}
  02_pages/page_004.jpg
  03_drawings/page_004_diagram.jpg
  04_wings/B-WING-EAST/page_004.jpg
  04_wings/B-WING-EAST/zooms/page_004/   # plan_zoom_*.jpg + full_wing + manifest
  05_metadata/page_004.json
  summary.json
  job.json
```

## Secrets

Copy `.env.example` → `.env` and add API keys locally. `.env` and `storage/jobs/` are gitignored — **do not commit keys or job artifacts**.
