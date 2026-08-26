# Architecture — Dual-Pathway Drawing Split

## Overview

Construction bid PDFs combine vector text (sheet numbers, wing titles, room tags) with raster plan linework. This pipeline uses two interpretation paths that merge before OpenCV cropping.

## Path 1 — Vector / Text (PyMuPDF)

Reads the PDF text layer directly:

- **Sheet codes** — T-020, A1.1
- **Wing titles** — B-WING (EAST), ADMIN BLDG
- **Room tag prefixes** — B, C, AD from tags like B-13, C-19

Module: `src/pipeline/sheet_scan.py`

## Path 2 — Vision (LLM)

For page classification and wing anchoring on scanned or ambiguous sheets:

- **Page classify** — is this a plan/diagram page?
- **Plan area** — coarse bounding box for diagram crop
- **Wing map** — named regions per wing/building panel

Module: `src/pipeline/llama_vision.py`

Default runtime: Groq with `qwen/qwen3.6-27b`.

## CV Boundary Engine (OpenCV)

Vision and PDF text provide **anchors only** — never used as crop edges.

- **Diagram crop** — `diagram_crop.py` removes title block, notes, legend
- **Wing split** — `wing_crop.py` + `ink_profile.py` cut at sparsest ink between wing labels, using room-tag clusters when wings share walls

## Data flow

```
PDF
 ├─ PyMuPDF text → wing labels, room tags, sheet no
 └─ Page render → JPEG
       └─ Vision LLM → classify + wing boxes
             └─ CV crop diagram
                   └─ CV split wings → 04_wings/{NAME}/page_NNN.jpg
```

## Design rules

1. PDF text labels win over vision model names when both exist.
2. Room tags help place boundaries between adjacent wings; not exported in v1.
3. Quality gates reject blank/sliver crops (min ink, min plan linework %).
4. Each PDF gets an isolated job folder under `storage/jobs/{pdf_stem}/`.

## Dependencies

| Package | Role |
|---------|------|
| PyMuPDF | PDF render + text |
| Pillow | Image I/O |
| OpenCV | Cropping |
| httpx | Groq API |
| python-dotenv | Config |

Optional: Transformers + torch for local vision LLM.
