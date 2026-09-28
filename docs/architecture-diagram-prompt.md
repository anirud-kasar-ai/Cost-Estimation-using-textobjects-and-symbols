# Gemini Prompt — Architecture Diagram

Copy-paste the prompt below into Gemini (or any diagram-capable AI) to generate the
architecture diagram for this POC.

---

Create a clean, professional system architecture diagram for a proof-of-concept called
"Cost Estimation Using Text and Object" — an AI system that reads construction drawing
PDFs and produces a priced cost invoice. Use a left-to-right flow with clearly labeled
layers, rounded rectangles for components, arrows showing data flow, and distinct colors
per layer. Style: modern flat design, suitable for a technical presentation slide.

Draw these four layers:

LAYER 1 — Presentation (React frontend, Vite dev server, port 8080):
One box "i-TAB Estimator Pro (React + TanStack)" containing five small page boxes:
"Upload (single/batch)", "Project Report + Detection Overlay", "Symbol Pricing Editor",
"Invoice (editable tax)", "Model Statistics". Arrow labeled "REST /api (Vite proxy,
polling every 2.5s)" going to Layer 2.

LAYER 2 — API & Orchestration (FastAPI backend, port 8767):
Boxes: "REST API (jobs, counts, overlay, corrections, pricing, tax, stats, files)",
"Job Queue — ThreadPoolExecutor, max 2 concurrent jobs, auto-resume on restart".
Arrow to Layer 3 labeled "process_job()".

LAYER 3 — AI Processing Pipeline (Python), shown as a horizontal chain of 8 steps
with arrows between them:
1. "Legend Extraction (OCR → symbol_table.json)"
2. "PDF Page Rendering (PyMuPDF, 200 DPI)"
3. "Page Classification (Vision LLM — Groq Qwen / Llama 3.2 Vision)"
4. "Drawing Crop + Wing Mapping (Vision LLM)"
5. "Wing Crop + Zoom Tiles"
6. "YOLO Symbol Detection (auto-selected .pt model per PDF)"
7. "Legend Matching + Count Aggregation (per wing → per page → total)"
8. "Pricing + Invoice Generation (ReportLab PDF, CSV, 4.5% tax)"
Add a small side box "YOLO model registry: model/*.pt with embedded training metrics"
feeding step 6, and a side box "Vision LLM API (Groq cloud)" feeding steps 3–4.

LAYER 4 — Storage (file system, no database):
Boxes: "storage/<job_id>/ — source, pages, wings, zoom tiles, counts, invoice.pdf,
job.json, corrections.json", "pricing/device_prices.csv + price_overrides.json".

Also show two human-in-the-loop feedback arrows drawn from Layer 1 back into Layer 4:
"Manual count corrections (with notes)" and "Manual price overrides (with reasons)",
and one arrow from Layer 1 to Layer 2 labeled "Editable tax rate → invoice regenerated".

DATA FLOW — below the four layers, add a horizontal band titled "Data Journey" showing
how one uploaded PDF is transformed at each step, as a chain of small document/data
icons connected by numbered arrows:

1. "drawing.pdf (multi-page, multi-wing)" →
2. "symbol_table.json (legend rows: name, description, mfg/model, part no.)" →
3. "01_pages/*.png (rendered pages)" →
4. "02_classified (drawing vs legend/notes per page)" →
5. "04_wings/*.png (one image per wing)" →
6. "05_zoom tiles (small crops so tiny symbols are visible)" →
7. "detections (class, confidence, bounding box, wing)" →
8. "06_counts/result.json (counts per wing → per page → PDF total)" →
9. "invoice.json → invoice.csv + invoice.pdf (lines, subtotal, tax, grand total)"

Annotate each arrow in this band with the component that performs the transformation
(e.g. arrow 2→3 "PyMuPDF render", 3→4 "Vision LLM classify", 6→7 "YOLO detect",
7→8 "legend matching + aggregation", 8→9 "pricing engine + ReportLab").
Visually align each data artifact under the pipeline step in Layer 3 that produces it.

Title the diagram: "Cost Estimation Using Text and Object — POC Architecture".
