# Real Data — Diagram Scan + OP Verification Report

**Date:** 2026-08-14  
**Modules:** `sheet_identity.py`, `tiled_detection.py`, `zoom_tiles.py`  
**OP root:** `hvac-cost-estimator/backend/storage/realdata_op_verify/`

## How to re-run

```bash
cd hvac-cost-estimator/backend
# Classify-only (all Real data PDFs)
.venv\Scripts\python.exe -m pytest tests/test_real_data_diagram_scan.py -q

# Extract + validate OP (page/ROI/zooms) for key PDFs
.venv\Scripts\python.exe scripts\verify_realdata_op.py
```

## Summary

| Stage | Result |
|-------|--------|
| Diagram scan (17 PDFs) | **TEST PASS** 17/17 |
| OP extract+validate (5 PDFs, 62 plan folders) | **TEST PASS** 62/62 |
| Unit tests (sheet/zoom/tiles) | **TEST PASS** |

---

## Part A — Diagram page scan (classify)

See prior cases: every Real data PDF meets keep/skip expectations.  
Pytest: `tests/test_real_data_diagram_scan.py`.

---

## Part B — Stored OP verification (folders + crops)

Checked **each** `floor_plans/{slug}/` for:

| Check | Rule |
|-------|------|
| `meta.json` | page_number, display_name, slug, sheet_kind |
| `page.jpg` | valid JPEG |
| `rois/*_roi.png` + `*_roi.json` | present |
| `zooms/full_diagram.jpg` | present; ≥55% of page area |
| `zooms/plan_zoom_*.jpg` | JPEG only; readable ink |
| `zooms_manifest.json` | count matches files; overlap ≥0.10 |
| Coverage | overlapping tiles cover ROI (blank margins may be skipped) |

### OP runs — TEST PASS

| PDF | Plans | Pass | Fail | OP folder |
|-----|------:|-----:|-----:|-----------|
| `1961BIDDrawings test.pdf` | 21 | 21 | 0 | `…/1961BIDDrawings_test_2` |
| `1948BIDDrawings.pdf` | 33 | 33 | 0 | `…/1948BIDDrawings_2` |
| `1954BIDDrawings-AyersES011325.pdf` | 1 | 1 | 0 | `…/1954BIDDrawings-AyersES011325_2` |
| `1962BIDMHESDrawings.pdf` | 5 | 5 | 0 | `…/1962BIDMHESDrawings_2` |
| `1952BIDDrawings-Shadeland-Sunrise010625.pdf` | 2 | 2 | 0 | `…/1952BIDDrawings-Shadeland-Sunrise010625_2` |

**Expected folder layout (each diagram):**

```
floor_plans/T-021_Floor_Plans_Demo_Wing_A/
  meta.json
  page.jpg
  rois/sheet_005_roi.png
  rois/sheet_005_roi.json
  zooms/full_diagram.jpg
  zooms/plan_zoom_r##_c##.jpg
  zooms/zooms_manifest.json
```

### Issues found in OP (fixed)

1. **Empty / margin zoom tiles** — sparse sheets (e.g. T-021) produced mostly-white crops between notes and plan.  
   **Fix:** trim outer whitespace on zoom ROI; skip blank tiles (`HVAC_ZOOM_SKIP_BLANK_TILES`, ink ratio ≥ 0.012). Manifest records `skipped_blank`.
2. **Formats** — zooms forced to JPEG; `full_diagram.jpg` always written; manifest count synced to kept tiles only.

### Visual spot-checks

| Sheet | Check | Result |
|-------|-------|--------|
| T-021 A Wing | `full_diagram` keeps plan; `plan_zoom_r01_c02` shows rooms + numbered tags | **PASS** |
| T-701 Single Lines | full diagram keeps IDF blocks + network SLD | **PASS** |
| Ayers Roof Plan | full diagram keeps roof wings + callouts | **PASS** |

---

## Optimizations applied

1. Tech `INFRASTRUCTURE` titles + title-block sheet numbers  
2. Roof / arch floor / RCP / campus site titles  
3. Phase 1 whole-PDF scan → extract → ROI/zooms  
4. Zoom content trim + blank-tile skip  
5. Verify script: `scripts/verify_realdata_op.py`

## Known residual

- Detail-only / schedule-only sheets stay skipped (expected).  
- Left **notes column** can still appear in some kept tiles (ink-rich); primary plan zooms remain correct.  
- Older `realdata_op_verify/*` folders without `_2` may coexist from the first OP pass; latest verified set uses `*_2` names in the table above.
