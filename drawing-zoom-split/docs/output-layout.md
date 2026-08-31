# Output Layout

Each processed PDF produces one job folder named after the PDF stem.

## Folder structure

```
storage/jobs/{pdf_stem}/
  READ_ME_FIRST.txt
  01_source/{original.pdf}
  02_pages/page_004.jpg              kept plan pages only
  03_drawings/page_004_diagram.jpg   cropped diagram
  04_wings/
    B-WING-EAST/page_004.jpg         one image per wing
    B-WING-EAST/zooms/page_004/      588px tiles + full_wing + zooms_manifest.json
    C-WING-EAST/page_004.jpg
  05_metadata/page_004.json          wing map + boxes
  summary.json                       flat index
  job.json                           job status
```

## Wing folder naming

Uppercase slug of sheet text: non-alphanumerics → hyphen.

Examples: `B-WING-EAST`, `ADMIN-BLDG`, `E-WING-NORTH`

## Metadata schema (`05_metadata/page_NNN.json`)

| Field | Description |
|-------|-------------|
| `page` | 1-based page number |
| `sheet_no` | Sheet code from PDF text (e.g. T-020) |
| `title` | Combined sheet + wing titles |
| `diagram_image` | Path to diagram JPEG |
| `wing_map` | Merged PDF-text + vision wing anchors |
| `label_points` | PDF wing title centers (diagram coords) |
| `tag_points` | Room tag prefix positions |
| `wings[]` | Per-wing image path, CV boxes, pixel size |
| `wings_without_image` | Labels on sheet but plan elsewhere |

## Batch manifest

After a batch run, `storage/jobs/batch_manifest.json` lists all PDFs processed with status, diagram count, and wing count.
