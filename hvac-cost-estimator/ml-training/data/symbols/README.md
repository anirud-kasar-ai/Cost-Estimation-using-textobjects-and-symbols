# Symbol detection annotations (bounding boxes)

Train the Detectron2 **detection + classification** branch on sheet-level boxes,
not legend crops alone.

## Target slice (re-annotate)

Start with a small Mt. Diablo / MHES sample set already under `Real data/`:

| Bid set | Suggested sheets |
|---------|------------------|
| Mt. Diablo Elementary (`1962BIDMDESDrawings*.pdf`) | floor plans with device symbols |
| MHES / RVES (`1945BIDDrawingsRVES*.pdf`) | same |

Re-annotate a **slice** (2–4 plan sheets) with axis-aligned boxes around each
device symbol *in place on the drawing*, plus class labels.

## Layout

```
ml-training/data/symbols/
  images/                 # page PNGs (export via pdf_to_image / pipeline pages/)
  annotations.json        # COCO format (this folder's template)
  README.md               # this file
```

## COCO categories

Use one category per classifier label (matches `ml/classifier.py`):

1. `supply_air_diffuser`
2. `return_air_grille`
3. `exhaust_grille`
4. `co2_sensor`
5. `temperature_sensor`
6. `thermostat`
7. `vav_box`
8. `raceway`  ← elongated boxes; pipeline converts length → LF via scale bar

Alternatively annotate a single `device_symbol` class and keep the CNN
classifier for sub-type (matches the older two-stage design). Prefer per-class
boxes if you want single-stage detection.

## Workflow

1. Render pages: `python -m ml.pdf_to_image …` or copy from
   `backend/storage/<project_id>/pages/`.
2. Label in [Label Studio](https://labelstud.io/) / CVAT → export **COCO**.
3. Place export at `annotations.json` (see `annotations.template.json`).
4. Run `backend/scripts/train_symbol_detector.py` once the training loop is
   implemented.

## Notes

- Boxes must be on the **plan sheet**, not legend glyph crops.
- Include raceway/conduit as long thin boxes so scale calibration can produce LF.
- Keep image filenames stable so COCO `file_name` matches `images/`.
