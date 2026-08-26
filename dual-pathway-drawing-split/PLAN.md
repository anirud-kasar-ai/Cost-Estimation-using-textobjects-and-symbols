# Dual-Pathway PDF Drawing Split — Plan

See the approved plan in the Cursor plans folder. This file mirrors the living implementation doc for the project.

## Goal

Process multiple construction PDFs. For each PDF:

1. Keep only plan/diagram pages.
2. Crop the drawing area.
3. Split by wing/building name.
4. Store wing images under wing-named folders (per-PDF job folders).

**v1 scope:** wing-level crops only. Room tags (B1, B2) inform wing boundaries but are not exported as separate images.

## Project layout

```
dual-pathway-drawing-split/
  src/pipeline/     CV + vision pipeline (copied from vision-extraction)
  src/batch.py      multi-PDF batch runner
  scripts/          CLI entry points
  storage/jobs/     output (gitignored)
  docs/             architecture, requirements, validation
```

## Pipeline stages

| Stage | Module | Output |
|-------|--------|--------|
| 0 Batch | `src/batch.py` | one job folder per PDF |
| 1 Pages | `page_render.py`, `sheet_scan.py` | `02_pages/` |
| 2 Diagram | `diagram_crop.py` | `03_drawings/` |
| 3 Wings | `wing_crop.py`, `llama_vision.py` | `04_wings/` |
| 4 Verify | `scripts/verify_output.py` | pass/fail report |

## Future phases

- v2: per-room crops
- v3: symbol spotting (DPSS)
- v4: DWG/DXF vector path
- v5: BIM export
