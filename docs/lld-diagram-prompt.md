# Gemini Prompt — LLD (Low-Level Design) Diagram

Copy-paste the prompt below into Gemini.

---

Create a detailed LOW-LEVEL DESIGN diagram for a proof-of-concept called "Cost Estimation
Using Text and Object" — an AI system that reads construction drawing PDFs and produces a
priced cost invoice. This is an engineering diagram: dense but organized, monospace-style
labels for file/module names, grouped swimlanes, and numbered sequence arrows. Modern flat
design, landscape orientation.

Draw three swimlanes top to bottom:

SWIMLANE 1 — Frontend "i-TAB Estimator Pro (React + TanStack, Vite :8080)":
- Box "src/lib/api.ts — typed API client (API_BASE=/api, Vite proxy → :8767)".
- Box "src/lib/estimator-data.tsx — EstimatorProvider: polls /api/jobs every 2.5s while
  busy; merge order for line items: local edits → saved corrections → model counts".
- Five route boxes: "index.tsx Upload (single + batch ≤10)", "report.tsx Report
  (counts, confidence, detection overlay, legend)", "pricing.tsx Pricing editor",
  "invoice.tsx Invoice (editable tax)", "metrics.tsx Model statistics".

SWIMLANE 2 — Backend "FastAPI app.py (:8767)":
- Box "Job queue: ThreadPoolExecutor(max_workers=2), _running set + lock,
  startup hook re-queues unfinished jobs after restart".
- Box listing REST endpoints in small text:
  "POST /api/jobs (upload, dedupe to running job)", "GET /api/jobs",
  "GET /api/jobs/{id}/counts", "GET /api/jobs/{id}/overlay",
  "GET+PUT /api/jobs/{id}/corrections", "PUT /api/jobs/{id}/tax (rebuild invoice)",
  "GET+PUT /api/pricing (override with reason, CSV-lock probe)", "GET /api/stats",
  "GET /api/jobs/{id}/files/* (safe path serving)".

SWIMLANE 3 — Pipeline "drawing-zoom-split/src/pipeline/ — process_job(job_id)":
Show as a numbered sequence of 9 stage boxes with the module name under each stage:
1. "extracting — extraction/ → symbol_table.json (legend rows)"
2. "rendering — page_render.py (PyMuPDF, 200 DPI) → 01_pages/*.png"
3. "loading_model — connect Vision LLM (Groq qwen / Llama 3.2 Vision)"
4. "classifying — Vision LLM: drawing vs legend/notes → 02_classified"
5. "drawing_crop — crop drawing region → 03_diagrams"
6. "wing_map + wing_crop — Vision LLM wing mapping → 04_wings/*.png"
7. "zoom + counting — count_stage.py + yolo_detect.py: zoom tiles (05_zoom),
   YOLO auto-model-selection (score model/*.pt classes vs legend via
   yolo_class_map.py), detect per tile (class, conf, bbox, wing_folder),
   aggregate per wing → page → total; unmatched classes → yolo_unmatched"
8. "pricing — pricing.py: device_prices.csv + price_overrides.json,
   build_invoice(tax_rate=0.045 default)"
9. "invoice — invoice_pdf.py (ReportLab) → 06_counts/invoice.{json,csv,pdf}"

At the bottom add a storage strip "storage/<job_id>/" showing the folder schema inline:
"00_source | 01_pages | 02_classified | 03_diagrams | 04_wings | 05_zoom |
06_counts (result.json, invoice.*) | job.json (status/stage/progress/eta) |
symbol_table.json | corrections.json | summary.json".

Sequence arrows to draw across swimlanes, numbered:
(1) Upload POST /api/jobs → job folder created, status queued.
(2) Executor picks job → runs stages 1–9, writing job.json progress after each stage.
(3) UI polls GET /api/jobs → progress bar + ETA.
(4) On done: UI fetches counts, overlay, legend, invoice.
(5) User edits: PUT corrections / PUT pricing / PUT tax → artifacts updated,
    invoice regenerated.

Also add two small callout notes: "Restart-safe: startup re-queues queued/processing
jobs" on the backend lane, and "Ignored YOLO classes: object, classroom 15" on stage 7.

Title: "Cost Estimation Using Text and Object — Low-Level Design".
