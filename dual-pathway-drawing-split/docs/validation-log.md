# Validation Log

**Date:** 2026-08-18  
**Command:**

```powershell
cd dual-pathway-drawing-split
.\.venv\Scripts\python.exe scripts/run_batch.py `
  --input "D:\Cost Estimation Using Text and Object\Real data" `
  --files 1948BIDDrawings.pdf 1952BIDDrawings-Shadeland-Sunrise010625.pdf 1962BIDMHESDrawings.pdf `
  --verify
```

## Summary

| Stage | Result |
|-------|--------|
| Batch (3 PDFs) | **PASS** 3/3 |
| Verify (366 checks) | **PASS** 0 issues |
| Unit tests | **PASS** 2/2 |

## Per-PDF results

| PDF | Job folder | Diagrams | Wings | Kept pages | Verify |
|-----|------------|----------|-------|------------|--------|
| `1948BIDDrawings.pdf` | `1948BIDDrawings` | 29 | 52 | 29/35 | PASS |
| `1952BIDDrawings-Shadeland-Sunrise010625.pdf` | `1952BIDDrawings-Shadeland-Sunrise010625` | 3 | 3 | 3/6 | PASS |
| `1962BIDMHESDrawings.pdf` | `1962BIDMHESDrawings` | 5 | 19 | 5/8 | PASS |

## Output checks

Each job folder contains:

- `01_source/` — original PDF
- `02_pages/` — kept plan pages only
- `03_drawings/` — one diagram crop per kept page
- `04_wings/{WING-NAME}/page_NNN.jpg` — wing-level crops grouped by name
- `04_wings/{WING-NAME}/rois/{page_key}/roi.png` and `roi_overlay.png` — per-wing ink-tight ROI + overlay
- `04_wings/{WING-NAME}/zooms/{page_key}/plan_zoom_r##_c##.jpg` and `zooms_manifest.json` — overlapping zoom tiles for review
- `05_metadata/page_NNN.json` — wing map, label points, CV boxes
- `summary.json`, `job.json`

Example wing folders from `1948BIDDrawings` (T-020 multi-wing sheet):

- `ADMIN-BLDG/`
- `B-WING-EAST/`
- `C-WING-EAST/`

## Artifacts

- `storage/jobs/batch_manifest.json` — batch run summary
- `storage/verify_report.json` — verify script output

## Notes

- Credentials loaded from local `.env` or sibling `vision-extraction/.env` (no secrets copied).
- Room tags (B, C prefixes) are used for wing boundary placement only; v1 does not export per-room crops.
- Non-plan pages (schedules, details, cover) correctly skipped per `summary.json`.
- ROI/zoom artifacts are written per wing-crop instance as `rois/<page_key>/` and `zooms/<page_key>/` to avoid cross-page overwrites.

## Backfill older jobs

Some job folders (generated before ROI/zoom export was added) may have wing crops in `04_wings/` but missing:
- `04_wings/<wing>/rois/<page_key>/...`
- `04_wings/<wing>/zooms/<page_key>/...`

To backfill those artifacts without re-running the LLM:

```powershell
cd dual-pathway-drawing-split
.\.venv\Scripts\python.exe scripts\backfill_wing_roi_zooms.py 1961BIDDrawings "1961BIDDrawings test"
.\.venv\Scripts\python.exe scripts\verify_output.py 1961BIDDrawings
```

## Real data ROI + zoom test (2026-08-18)

Representative Real data PDFs from `Real data/test.md`:

```powershell
cd dual-pathway-drawing-split
.\.venv\Scripts\python.exe scripts\verify_realdata_roi_zoom.py
.\.venv\Scripts\python.exe scripts\verify_realdata_roi_zoom.py --backfill-missing
.\.venv\Scripts\python.exe scripts\run_batch.py `
  --input "D:\Cost Estimation Using Text and Object\Real data" `
  --files 1954BIDDrawings-AyersES011325.pdf `
  --verify
```

| PDF | Job | Wings | ROI/zoom | Verify |
|-----|-----|-------|----------|--------|
| `1961BIDDrawings test.pdf` | `1961BIDDrawings test` | 23 | 23 | PASS |
| `1948BIDDrawings.pdf` | `1948BIDDrawings` | 52 | 52 | PASS |
| `1954BIDDrawings-AyersES011325.pdf` | `1954BIDDrawings-AyersES011325` | 2 | 2 | PASS |
| `1962BIDMHESDrawings.pdf` | `1962BIDMHESDrawings` | 19 | 19 | PASS |
| `1952BIDDrawings-Shadeland-Sunrise010625.pdf` | `1952BIDDrawings-Shadeland-Sunrise010625` | 3 | 3 | PASS |

Artifacts:
- `storage/realdata_roi_zoom_verify.json` — per-PDF ROI audit
- `storage/verify_report.json` — full verify_output results

Notes:
- `1961BIDDrawings test` was backfilled (23 wings) after initial wing-only run.
- Fresh Ayers batch run confirms ROI/zoom are written by the main pipeline (`job.py`), not only backfill.
