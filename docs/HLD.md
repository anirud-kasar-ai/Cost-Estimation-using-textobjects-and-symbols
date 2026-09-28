# High-Level Design (HLD)
## Cost Estimation Using Text and Object — Construction Drawing Cost Estimator (POC)

### 1. Purpose

Automate cost estimation from construction drawing PDFs. The system ingests a
multi-page, multi-wing drawing set, extracts the symbol legend, detects and
counts device symbols on every wing of every drawing page, prices the counted
devices, and produces a customer-ready cost invoice (PDF/CSV) — with a web UI
for uploads, review, manual correction, and pricing management.

### 2. Scope of the POC

- Input: construction drawing PDFs (multi-page, multiple wings per page) or single images.
- Output: symbol counts per wing/page, matched legend descriptions, priced invoice
  (JSON, CSV, PDF), detection overlays for visual verification, model metrics.
- Human-in-the-loop: manual count corrections with notes, manual price overrides
  with reasons, editable invoice tax rate.

### 3. System Overview

```
┌─────────────────────┐        ┌──────────────────────┐        ┌──────────────────────────┐
│  React Frontend     │  /api  │  FastAPI Backend     │        │  Processing Pipeline     │
│  (i-TAB Estimator   │ ─────► │  app.py :8767        │ ─────► │  drawing-zoom-split/     │
│   Pro, Vite :8080)  │ proxy  │  Job queue (2 wkrs)  │        │  src/pipeline/           │
└─────────────────────┘        └──────────────────────┘        └──────────────────────────┘
                                        │                                 │
                                        ▼                                 ▼
                               ┌──────────────────┐    ┌───────────────────────────────────┐
                               │ File storage     │    │ Models & external services        │
                               │ storage/<job>/   │    │ • YOLO .pt weights (local)        │
                               │ pricing/*.csv    │    │ • Vision LLM (Groq / HF) for page │
                               │ *.json overrides │    │   classification & wing mapping   │
                               └──────────────────┘    └───────────────────────────────────┘
```

### 4. Major Components

| Component | Technology | Responsibility |
|---|---|---|
| Frontend UI | React, TanStack Start/Query, Vite (port 8080) | Upload (single + batch), live progress/ETA, project report, detection overlay, pricing editor, invoice view, model metrics |
| Backend API | FastAPI (port 8767) | Upload handling, job lifecycle, REST API, serving artifacts (invoice PDF/CSV, overlays, legend) |
| Job Queue | ThreadPoolExecutor (max 2 concurrent) | Bounded parallel processing of uploads; re-queues unfinished jobs on restart |
| Processing Pipeline | Python (PyMuPDF, OpenCV, Ultralytics YOLO) | PDF → pages → wings → zoom tiles → symbol detection → counting → invoice |
| Vision LLM | Groq-hosted Qwen (default) or HF Llama 3.2 11B Vision | Page classification (drawing vs. legend/notes) and wing mapping |
| Symbol Detector | YOLO (custom-trained .pt weights) | Detect device symbols on zoom tiles; auto-selects best model per PDF based on legend match |
| Legend Extraction | OCR/table extraction | Extracts the symbol legend (name, description, mfg/model, part number, note) |
| Pricing Engine | CSV + JSON overrides | device_prices.csv with persistent manual overrides (price_overrides.json) |
| Invoice Generator | ReportLab | Customer-ready invoice PDF with lines, subtotal, configurable tax, grand total |

### 5. End-to-End Flow

1. **Upload** — user drops PDF(s) in the UI; backend creates a job folder and queues it
   (duplicate uploads attach to the running job instead of failing).
2. **Extract** — symbol legend and requirement tables are extracted from legend pages.
3. **Render & Classify** — pages rendered to images; a vision LLM classifies each page
   (drawing vs. legend/notes) and maps wings on drawing pages.
4. **Crop & Zoom** — each drawing is cropped, each wing cut out, and zoom tiles generated
   per wing so small symbols are detectable.
5. **Detect & Count** — the YOLO model (auto-selected per PDF via legend matching) detects
   symbols on every tile; counts are deduplicated and aggregated per wing, per page, and
   for the whole PDF; detected classes are mapped to legend rows via a name-variant map.
6. **Price & Invoice** — counts × unit prices (with any manual overrides) produce
   invoice.json/csv/pdf with subtotal, tax (default 4.5%, user-editable), grand total.
7. **Review** — UI shows the report (counts, confidence, detection overlay, legend),
   allows manual count corrections (with notes) and price edits (with reasons), and
   regenerates the invoice on tax change.

### 6. Data Stores (file-based, no DB in POC)

- `storage/<job_id>/` — one folder per upload: source PDF, rendered pages, wing crops,
  zoom tiles, counts, invoice artifacts, `job.json` (status/progress), `symbol_table.json`,
  `corrections.json`.
- `pricing/device_prices.csv` — unit prices; `price_overrides.json` — persistent manual edits.
- `model/*.pt` — YOLO weights with embedded training metrics (mAP50, precision, recall).

### 7. Key Design Decisions

- **File-system job store** instead of a database — simplest thing that works for a POC;
  every artifact is inspectable on disk.
- **Bounded concurrency (2 workers)** — protects GPU/CPU during batch uploads; extra
  uploads wait in a visible "queued" state.
- **Startup job resume** — jobs left queued/processing by a restart are automatically
  re-queued (crash-safe job lifecycle).
- **Model auto-selection** — multiple YOLO weight files are scored against the extracted
  legend; the best-matching model is used per PDF (roof vs. tech drawings).
- **Human-in-the-loop corrections** — model output is editable everywhere it matters
  (counts, prices, tax) and edits persist with audit notes.

### 8. Non-Functional Characteristics

- **Progress & ETA**: per-stage progress written to `job.json`, polled by the UI (2.5 s
  while busy), with live ETA refinement.
- **Resilience**: duplicate-upload reuse, restart re-queue, locked-CSV detection with a
  clear user-facing error.
- **Portability**: everything runs locally (Windows dev box) except the optional
  Groq-hosted vision LLM call.
