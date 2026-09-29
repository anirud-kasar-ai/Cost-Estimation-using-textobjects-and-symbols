# UI Understanding Guide — Cost Estimate Using Site Construction Drawing

This document explains the web UI of the project so a new teammate can understand what each screen does, how the pieces connect, and how to run it locally.

---

## 1. What the product does

The user uploads a **construction drawing PDF**. The system then:

1. Extracts the **symbol legend** and requirement sheets from the PDF.
2. Classifies pages and crops the plan diagrams into per-wing images.
3. Runs a **YOLO model** to detect electrical/technical symbols on each wing.
4. Matches detections to legend rows, applies unit prices from a **pricing catalog**, and produces a **costed estimate / invoice** (JSON, CSV, and printable PDF).

The UI lets the user upload drawings, watch processing progress, visually verify detections, correct counts, edit prices, compare projects, and download the final invoice.

---

## 2. Architecture at a glance

```
i-TAB Estimator Pro (React frontend, port 5173)
        │  /api/*  (Vite dev proxy)
        ▼
drawing-zoom-split/app.py (FastAPI backend, port 8766)
        │
        ▼
Pipeline stages → storage/jobs/{job_id}/ (images, result.json, invoice.pdf …)
```

| Piece | Location | Tech |
|---|---|---|
| Frontend UI | `i-TAB Estimator Pro/` | React, TanStack Start/Router/Query, Tailwind CSS, shadcn/ui components |
| Backend API + pipeline | `drawing-zoom-split/` | FastAPI, OpenCV, YOLO (Ultralytics), vision LLM (Groq/HF) |
| Job artifacts | `drawing-zoom-split/storage/jobs/{job_id}/` | Files on disk (no database) |

Key frontend files:

- `src/components/app-shell.tsx` — common layout: header (logo, currency selector, dark/light toggle), sidebar navigation, breadcrumbs, footer.
- `src/lib/api.ts` — typed client for every backend endpoint (base URL `/api`).
- `src/lib/estimator-data.tsx` — global state (`EstimatorProvider` / `useEstimator` hook): job list polling, selected project, editable line items, corrections auto-save, currency formatting.
- `src/routes/*.tsx` — one file per page (see below).

There is also an older, minimal single-file UI at `drawing-zoom-split/static/index.html`, served directly by the backend at `http://127.0.0.1:8766/`. The React app is the main UI; the static page is a fallback/debug view of the same API.

---

## 3. How to run it locally

```powershell
# 1. Backend (FastAPI + pipeline)
cd drawing-zoom-split
.\.venv\Scripts\Activate.ps1        # after initial venv + pip install (see SETUP.md)
python app.py                        # serves http://127.0.0.1:8766

# 2. Frontend (React UI)
cd "i-TAB Estimator Pro"
npm i
npm run dev                          # serves http://localhost:5173
```

The Vite dev server proxies all `/api/*` calls to `http://127.0.0.1:8766` (see `vite.config.ts`), so both must be running.

---

## 4. Pages (sidebar navigation)

### 4.1 Upload Drawings — `/` (`src/routes/index.tsx`)

The entry point of the app.

- **Drag-and-drop / click-to-browse** zone that accepts a single PDF.
- After clicking **Process Drawing**, the file is POSTed to `/api/jobs`; the backend creates a job and processes it in a background worker (max 2 concurrent jobs; extras wait as "queued").
- A **live progress card** tracks the job by polling `/api/jobs` (every 2.5 s while a job is running, 15 s otherwise). Backend stages are mapped to five visible steps:
  1. Extracting Legend & Requirements
  2. Classifying Pages
  3. Cropping Diagrams & Wings
  4. Detecting Symbols (YOLO)
  5. Pricing & Invoice
- Shows a progress bar, ETA, and the elapsed time when done, plus a **View Report** button.
- **Recent Uploads table** — every past job with filename, upload time, size, processing time, status badge (Queued / Processing / Complete / Failed), an invoice PDF download link, and a View Report link. Sortable and paginated (10 per page).

If the same file is uploaded again, the job is re-processed under the same job id (id is derived from the filename).

### 4.2 Project Report — `/report` (`src/routes/report.tsx`)

The main results screen for the selected project. Sections top to bottom:

- **Project Metadata** — source file, size, page count, status, upload time, detection model used, legend row count, wings scanned, processing time.
- **Detected Symbols (by model class)** — per-class detection tallies.
- **Detection Confidence** — total symbols, and per-class average / minimum confidence.
- **Detection Overlay (visual verification)** — each wing image is displayed with the YOLO bounding boxes drawn on top (colored per class, tooltip shows class + confidence). A toggle shows/hides detections that were *not* matched to any legend row.
  - **Review mode** is the human-correction workflow: click a box to remove a false positive (count −1), or pick a symbol class from a dropdown and click anywhere on the drawing to add a missed symbol (count +1). Every change auto-saves.
- **Symbol Legend** — the legend rows extracted from the PDF, with a link to download the extracted legend/technical-symbol PDF.
- **Detailed Costing Report** — one editable line per priced device: count (model count kept visible next to any human-corrected count), unit price, note, and line total. Edits here are **corrections**:
  - Edits auto-save 2.5 s after the last change (`PUT /api/jobs/{id}/corrections`).
  - Saved corrections feed the accuracy metrics on the Model Performance page.
- Download buttons: **Invoice CSV** and **Invoice PDF**.

### 4.3 Invoice — `/invoice` (`src/routes/invoice.tsx`)

A printable, client-facing cost estimate for the selected project.

- Rendered as a white A4-style document: header with estimate number (`EST-<JOB_ID>`), issue date and validity window, **From / Bill To** blocks, line items, subtotal, tax, and grand total.
- **Editable client details** (name, address, contact) — saved via `PUT /api/jobs/{id}/client`, which also regenerates the backend estimate PDF.
- **Editable tax rate** — click the tax row, enter a percentage; saved via `PUT /api/jobs/{id}/tax`, which rebuilds `invoice.json`, `invoice.csv`, and `invoice.pdf` on the backend.
- Actions: **Print** (browser print with print-friendly styling), **Download Estimate (PDF)** (the backend-generated PDF), and Back to Report.

### 4.4 Pricing Catalog — `/pricing` (`src/routes/pricing.tsx`)

The unit-cost table behind every estimate (backed by `pricing/device_prices.csv`).

- Lists every device class with its **symbol image** (a real sample cropped from processed drawings — hover to enlarge; falls back to a letter badge when no drawing containing that symbol has been processed yet), unit, unit price, and last-updated time.
- **Click a price to edit it** (`PUT /api/pricing/{device_key}`). Manually edited prices are marked "overridden" and are **locked** — they never drift when synthetic prices refresh. An optional reason can be recorded and is shown as a tooltip.

### 4.5 Model Performance — `/metrics` (`src/routes/metrics.tsx`)

A dashboard aggregated across **all** finished jobs (`GET /api/stats`).

- KPI cards: total symbols detected, drawings processed, average processing time, active YOLO model.
- Charts: symbol detections by class and detection share per class.
- **Class-wise model performance table**: per class — its symbol image (same source as the Pricing Catalog; hover to enlarge), which model(s) were trained on it, detection count, legend match rate, average confidence, and training metrics (mAP50, precision, recall) read from the model checkpoints.
- **Overall model accuracy**: computed from the human corrections saved on the Report page (`1 − |model − human| / human` over all verified lines), plus jobs/lines verified counts.
- Model inventory: every `.pt` weights file with size, modified date, classes, and which one is active.

### 4.6 Project Comparison — `/compare` (`src/routes/compare.tsx`)

Pick **two processed projects** and see them side by side: project details, per-class symbol counts, and cost deltas. Useful for comparing two revisions of the same drawing (cost impact of design changes) or benchmarking similar projects.

---

## 5. Global UI behaviors

- **Project selection** is global: choosing a job on any page (upload table, report, compare) sets the "selected project" used by Report and Invoice. Default = newest finished job.
- **Currency selector** (header): USD, EUR, GBP, INR, AED. Base prices are always USD; other currencies are display-only conversions at fixed rates. The choice persists in `localStorage`.
- **Dark/light theme toggle** (header), persisted as well.
- **Corrections priority**: what you see in an editable count/price is *(this-session edit) → (saved human correction) → (model value)*. The original model value stays visible so drift can be measured.
- **Resilience**: if the backend restarts mid-job, unfinished jobs are automatically re-queued on startup, so the UI progress card recovers on its own.

---

## 6. Backend API quick reference

| Endpoint | Used by | Purpose |
|---|---|---|
| `POST /api/jobs` | Upload page | Upload PDF, create + start a job |
| `GET /api/jobs` | All pages (polling) | List jobs with status/stage/progress |
| `GET /api/jobs/{id}/counts` | Report | Detection counts, rows, invoice JSON |
| `GET /api/jobs/{id}/overlay` | Report | Wing images + bounding boxes for the overlay view |
| `GET/PUT /api/jobs/{id}/corrections` | Report | Load/save human-verified counts and prices |
| `PUT /api/jobs/{id}/tax` | Invoice | Set tax rate, rebuild invoice files |
| `PUT /api/jobs/{id}/client` | Invoice | Save Bill-To block, regenerate estimate PDF |
| `GET /api/jobs/{id}/invoice(.csv/.pdf)` | Report, Invoice, Upload | Download invoice in each format |
| `GET /api/jobs/{id}/requirement.pdf`, `technical-symbol.pdf`, `sheet-notes.pdf` | Report | Download extracted PDFs |
| `GET /api/jobs/{id}/files/{path}` | Report overlay | Serve wing images from job storage |
| `GET /api/pricing`, `PUT /api/pricing/{key}` | Pricing | Read catalog, set price overrides |
| `GET /api/pricing/{key}/symbol.png` | Pricing | Sample image of the symbol, cropped from real detections |
| `GET /api/stats` | Metrics | Cross-job aggregates, per-class metrics, model inventory |
| `GET /api/health` | — | Runtime/model configuration check |

---

## 7. Typical end-to-end workflow

1. **Upload** a drawing PDF on the home page and watch the 5-step progress.
2. When complete, open the **Project Report**: check metadata, browse the wing overlays.
3. Turn on **Review mode** and fix any wrong detections (click boxes to remove, click the drawing to add). Adjust counts/prices in the **Detailed Costing Report** if needed — everything auto-saves as corrections.
4. Open the **Invoice** page, fill the client's Bill-To details, set the tax rate, then **Print** or **Download the estimate PDF**.
5. Over time, check **Model Performance** to see how accurate the model is versus the saved human corrections, and keep the **Pricing Catalog** up to date.
