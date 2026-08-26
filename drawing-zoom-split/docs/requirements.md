# Requirements — v1

## Functional

1. **Batch input** — Accept a directory of PDF files; process each independently.
2. **Page filter** — Keep only pages classified as plan/diagram sheets.
3. **Diagram crop** — Remove title block, notes, legend from kept pages.
4. **Wing split** — Produce one JPEG per wing/building area on multi-wing sheets.
5. **Wing folders** — Store crops under `04_wings/{WING-NAME}/page_NNN.jpg`.
6. **Metadata** — JSON per kept page with wing map and CV boxes.
7. **Verify** — Script validates JPEG quality and metadata ↔ disk consistency.

## Non-functional

- Standalone: no edits to sibling projects.
- Windows path safe folder names.
- Configurable via `.env` and `config/defaults.yaml`.

## Acceptance tests

Using real bid PDFs from `Real data/`:

| # | Criterion |
|---|-----------|
| 1 | N PDFs → N job folders under `storage/jobs/` |
| 2 | Non-plan pages in `summary.json` skipped list |
| 3 | Multi-wing sheets produce separate wing folders |
| 4 | Single-wing sheets produce one wing folder |
| 5 | `verify_output.py` exits 0 |

## Out of scope (v1)

- Per-room image crops (B-01, B-02)
- Symbol detection (DPSS)
- DWG/DXF input
- Web UI
