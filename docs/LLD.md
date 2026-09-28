# Low-Level Design (LLD)
## Cost Estimation Using Text and Object — Construction Drawing Cost Estimator (POC)

### 1. Repository Layout

```
Cost Estimation Using Text and Object/
├── drawing-zoom-split/              # Backend + processing pipeline
│   ├── app.py                       # FastAPI app (port 8767)
│   ├── model/                       # YOLO weights (*.pt), auto-discovered
│   ├── pricing/
│   │   ├── device_prices.csv        # Unit prices (generated + overrides applied)
│   │   └── price_overrides.json     # Persistent manual price edits
│   ├── storage/<job_id>/            # One folder per upload (see §4)
│   └── src/pipeline/
│       ├── job.py                   # Job lifecycle + stage orchestration
│       ├── config.py                # Env-driven configuration
│       ├── page_render.py           # PDF → page images (PyMuPDF)
│       ├── extraction/              # Legend / requirement / notes extraction
│       ├── count_stage.py           # YOLO counting, legend matching, invoicing
│       ├── yolo_detect.py           # Model loading, detection, ignored classes
│       ├── yolo_class_map.py        # Annotation-name → legend-name variants
│       ├── pricing.py               # Pricing CSV + overrides + invoice build
│       ├── invoice_pdf.py           # ReportLab invoice PDF
│       └── invoice_csv (in count_stage) # invoice.csv writer
└── i-TAB Estimator Pro/             # React frontend (Vite, port 8080)
    └── src/
        ├── lib/api.ts               # Typed API client (all backend calls)
        ├── lib/estimator-data.tsx   # EstimatorProvider: jobs, selection, overrides
        └── routes/
            ├── index.tsx            # Upload page (single + batch, recent uploads)
            ├── report.tsx           # Project report (counts, overlay, legend)
            ├── pricing.tsx          # Symbol pricing editor
            ├── invoice.tsx          # Invoice view (editable tax)
            └── metrics.tsx          # Model statistics
```

### 2. Backend (app.py)

#### 2.1 Job queue and lifecycle

- `_job_executor = ThreadPoolExecutor(max_workers=2)` — at most 2 jobs process
  concurrently; others stay `queued`.
- `_running: set[str]` guarded by `_job_lock` prevents duplicate submissions; an upload
  of a file whose job is already running returns the existing job.
- `@app.on_event("startup") _resume_incomplete_jobs()` — re-queues any job left in
  `queued`/`processing` by a previous run (restart-safe).
- Job status values: `queued → processing → done | failed`.
- Stages (written to job.json, surfaced in the UI): `extracting → rendering →
  loading_model → classifying → drawing_crop → wing_map → wing_crop → counting → done`.

#### 2.2 REST API

| Method & Path | Purpose |
|---|---|
| `POST /api/jobs` | Upload PDF/image; creates or reuses a job; returns job.json |
| `GET /api/jobs` | List all jobs (sorted by update time) |
| `GET /api/jobs/{id}` | Single job status/progress/ETA |
| `GET /api/jobs/{id}/counts` | result.json: totals, per-wing counts, invoice, unmatched, model used |
| `GET /api/jobs/{id}/overlay` | Wing images + detection boxes for the overlay viewer |
| `GET /api/jobs/{id}/corrections` / `PUT` | Human count corrections (+ optional notes) |
| `PUT /api/jobs/{id}/tax` | Update tax rate; rebuilds invoice.json/csv/pdf |
| `GET /api/jobs/{id}/files/…` | Serve artifacts (invoice.pdf, invoice.csv, images) — path-traversal safe |
| `GET /api/jobs/{id}/symbol-table` | Extracted legend rows |
| `GET /api/pricing` / `PUT /api/pricing/{device_key}` | Price list; manual override with reason (probes CSV writability first) |
| `GET /api/stats` | Aggregated detection stats + embedded model training metrics + verification accuracy |

### 3. Pipeline Detail (process_job)

1. **extracting** — legend/requirement extraction into `symbol_table.json` +
   `technical symbol.pdf` / `requirement.pdf`. Sheet-notes extraction disabled
   (`SHEET_NOTES_ENABLED=false`).
2. **rendering** — PDF pages → PNG at `VISION_PDF_DPI` (default 200) via PyMuPDF.
3. **loading_model** — connect vision LLM. Runtime `VISION_LLM_RUNTIME=groq`
   (`GROQ_MODEL_ID=qwen/qwen3.6-27b`) or local/HF Llama 3.2 11B Vision.
4. **classifying** — per page: drawing vs. legend/notes/schedule.
5. **drawing_crop** — crop the drawing region of each drawing page.
6. **wing_map** — vision LLM maps named wings on each drawing.
7. **wing_crop** — cut each wing into its own image under `04_wings/`.
8. **counting** (`count_stage.py`):
   - Zoom tiles are generated per wing (`05_zoom/`).
   - **Model auto-selection**: every `model/*.pt` is scored by how many of its class
     names map (via `yolo_class_map`) onto the extracted legend; the best scorer wins.
     Classes `object` and `classroom 15` are ignored (test-only classes).
   - YOLO inference per tile; per-hit payload keeps `wing_folder`, class, confidence,
     bbox (drives the overlay viewer).
   - Dedup/aggregate → per-wing, per-page, and total counts.
   - **Legend matching**: `yolo_class_map.py` normalises annotation names and maps
     name variants (endings differ across drawing sets) to legend rows; unmatched
     classes are reported separately (`yolo_unmatched`) and excluded from pricing.
   - Outputs `06_counts/result.json`.
9. **invoice** (`pricing.py` + `invoice_pdf.py`):
   - `ensure_pricing_csv()` regenerates `device_prices.csv`, always re-applying
     `price_overrides.json` (manual edits survive refreshes; `PermissionError` on a
     locked CSV is caught and logged, not fatal).
   - `build_invoice(tax_rate=INVOICE_TAX_RATE (default 0.045))` → lines
     (description, qty, unit price, line total), subtotal, tax, grand total.
   - `invoice.json`, `invoice.csv`, `invoice.pdf` (ReportLab; customer-facing —
     no pipeline internals like model name, per-wing tables, or processing time).

### 4. Job Folder Schema (`storage/<job_id>/`)

```
00_source/            original upload
01_pages/             rendered page PNGs
02_classified/        page classification results
03_diagrams/          cropped drawing regions
04_wings/             per-wing crops
05_zoom/              zoom tiles per wing
06_counts/            result.json, invoice.json, invoice.csv, invoice.pdf
job.json              id, status, stage, message, progress (0–1), eta_seconds,
                      page_count, wing_count, actual_seconds, timestamps
symbol_table.json     extracted legend rows
corrections.json      human-verified counts: [{device_key, corrected_count, note}]
summary.json          flat index + skipped pages
```

### 5. Frontend Detail

- **api.ts** — typed client; `API_BASE="/api"` proxied by Vite to `:8767` (no CORS).
- **estimator-data.tsx (EstimatorProvider)**:
  - `jobsQuery` polls `/api/jobs` every 2.5 s while any job is busy, else 15 s.
  - Selection defaults to the newest finished job.
  - `lineItems` merge order: local unsaved edits → saved corrections → model counts.
  - `saveCorrections()` PUTs corrections and invalidates queries.
- **Pages**:
  - *Upload*: single-PDF and batch (≤10) modes; recent uploads with status, time
    taken, and direct invoice download.
  - *Project Report*: metadata, detected-symbol counts, detection-confidence table
    (avg/min per class), detection overlay (wing image + bounding boxes + confidence),
    extracted symbol legend (mfg/model, part no., note → "Not available" when absent),
    unpriced-symbol warning, editable counts with optional reason notes.
  - *Symbol Pricing*: inline unit-cost editing with reason; "manual" badge on
    overridden rows; writes through to CSV + overrides file.
  - *Invoice*: customer-view invoice; editable tax rate (PUT + PDF regeneration);
    PDF/CSV download.
  - *Model Statistics*: overall model performance summary (training mAP50, precision,
    recall) + per-class table combining training metrics with observed detections.

### 6. Error Handling & Edge Cases

- Duplicate upload while running → returns the existing job (no 409 for the user).
- Backend restart mid-job → job re-queued automatically at startup; pipeline reruns
  from the beginning (stages are not checkpointed in the POC).
- `device_prices.csv` locked by Excel → write probed first; clear 400 error to the UI;
  background refresh skips instead of crashing.
- Pages where no wing/drawing is found are recorded in `summary.json` as skipped.
- Detections that match no legend row are counted but never priced (reported as
  `yolo_unmatched`).

### 7. Configuration (env vars, config.py)

| Variable | Default | Meaning |
|---|---|---|
| `UI_PORT` | 8766 (running on 8767) | Backend port |
| `INVOICE_TAX_RATE` | 0.045 | Default invoice tax |
| `SHEET_NOTES_ENABLED` | false | Sheet-note extraction toggle |
| `VISION_LLM_RUNTIME` | groq | Vision LLM backend (groq / hf) |
| `GROQ_MODEL_ID` | qwen/qwen3.6-27b | Groq vision model |
| `VISION_PDF_DPI` | 200 | Page render DPI |
