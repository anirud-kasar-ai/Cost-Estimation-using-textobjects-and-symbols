# Gemini Prompt — HLD (High-Level Design) Diagram

Copy-paste the prompt below into Gemini.

---

Create a simple, presentation-friendly HIGH-LEVEL DESIGN diagram for a proof-of-concept
called "Cost Estimation Using Text and Object" — an AI system that reads construction
drawing PDFs and produces a priced cost invoice. Keep it high level: only 6 major blocks,
large readable labels, generous spacing, modern flat design with one accent color per
block. This is for a management/stakeholder slide, so avoid low-level details like file
names, ports, or function names.

Blocks and flow (left to right):

1. "User" (person icon) — uploads construction drawing PDFs and reviews results.
2. "Web UI — i-TAB Estimator Pro (React)" — upload with live progress and ETA,
   project report, pricing editor, invoice view, model statistics.
3. "Backend API + Job Queue (FastAPI)" — receives uploads, queues jobs
   (max 2 processed in parallel), tracks status, serves results.
4. "AI Processing Pipeline (Python)" — one block containing four short bullet steps:
   "Extract symbol legend (OCR)", "Understand pages & wings (Vision LLM)",
   "Detect & count symbols (YOLO)", "Price counts & build invoice".
5. "AI Models & Services" (side block feeding block 4): "Custom-trained YOLO symbol
   detectors (local)" and "Vision LLM (Groq cloud)".
6. "Results Store (file system)" — job artifacts, symbol counts, price list,
   final invoice (PDF/CSV).

Arrows:
- User → Web UI: "upload PDF / review & correct".
- Web UI → Backend: "REST API".
- Backend → Pipeline: "process job".
- Pipeline ↔ Models & Services: "inference".
- Pipeline → Results Store: "counts + invoice".
- Results Store → Web UI: "report, overlay, invoice download".
- One feedback arrow User → Results Store labeled "human corrections
  (counts, prices, tax)" drawn as a dashed line.

Add a small caption strip at the bottom with the value proposition:
"PDF in → symbol counts → priced invoice out, with human-in-the-loop verification".

Title: "Cost Estimation Using Text and Object — High-Level Design".
