# Current Approach — Cost Estimation Using Text and Object

**Purpose of this document:** a complete, implementation-faithful briefing of what the system does today, so another model (or engineer) can propose a clearer, more reliable approach without guessing.

This is **not** a full quantity-takeoff + pricing product yet. Today the system stops at **legend-gated symbol counts on technology / low-voltage floor plans**. Dollar cost is a later stage and is not implemented in the tracked tools.

---

## 1. Product goal (intended)

From construction **bid drawing PDFs**, produce counts of **technology-legend equipment** on each wing/plan so those quantities can later be priced.

Typical drawing family (Garland-style AutoCAD/Revit plots):

- Multi-page bid sets with cover, notes, legends, details, and **floor-plan sheets**.
- One plan sheet often contains **several wing/building panels** (e.g. `B-WING (EAST)`, `C-WING`).
- Symbols are **tiny 1–2 px vector ink**: filled triangles, letter tags (`J`, `AP`, `WP`), raceway part numbers (`2300` / `5400` / `5500`), hourglass cameras, etc.
- The **legend lives on a separate technical-symbol sheet**, not on the plan.

**Success metric today:** unique physical instances of each legend key on a wing, with overlay boxes, not a dollar total.

---

## 2. Repo layout (what actually exists)

Four related codebases used to exist. **dual-pathway-drawing-split** is the canonical Stage 1 implementation (PDF upload → render → extract → wing crop → zoom tiles). It does **not** count symbols. Duplicate copies (`vision-extraction/`, `drawing-zoom-split/`) may still exist locally with older job storage.

| Path | Role | Status |
|------|------|--------|
| `dual-pathway-drawing-split/` | Dual-pathway PDF → wing crops + 653 px overlapping zoom tiles. | Tracked. UI on port 8766. |
| `symbol-count-vision/` | Count legend symbols on a zoom tile or zoom folder. CV owns boxes; taxonomy owns look-alikes. | Tracked. UI on port 8767. |
| `Real data/` | Sample bid PDFs (Git LFS). | Input corpus. |

There is **no live `hvac-cost-estimator` wiring**. Comments mention it as a future/sibling product. Cost/pricing is out of scope of the current pipeline.

**Preferred human workflow**

1. Run drawing split on a bid PDF → get `04_wings/{WING}/zooms/page_XXX/` (tiles + `full_wing.jpg` + `zooms_manifest.json`).
2. Upload that zoom folder + a **technical-symbol PDF** into `symbol-count-vision`.
3. Read `result.json` + `overlay.png`.

---

## 3. End-to-end data flow

```
Bid PDF
  │
  ├─ Path A: PDF text layer (PyMuPDF)  → sheet codes, wing titles, room tags
  └─ Path B: page JPEG + vision LLM    → is this a plan? coarse plan box? named wing regions
            │
            ▼
     OpenCV (owns all crop edges)
            │
            ├─ diagram crop (drop title block, notes, legend)
            ├─ wing split (cut at sparsest ink between labels)
            └─ ROI + overlapping 588 px zoom tiles (10% overlap)
            │
            ▼
     Zoom folder
       plan_zoom_r##_c##.jpg
       full_wing.jpg
       zooms_manifest.json
            │
            ▼
     Symbol count (legend-gated)
       parse technical-symbol PDF → catalog + glyph crops
       detect per tile (CV + Gemini classify / count-verify)
       map boxes into ROI space
       NMS merge (count once in overlap)
       glyph / vision judge (drop false hits)
       optional post-merge LLM count verify
            │
            ▼
     result.json  (counts per legend key)
     overlay.png  (boxes on full_wing)
```

**Hard design rule:** never detect on a stitched mosaic of the whole wing. Detect on **each zoom tile**, then merge in ROI coordinates. Full-wing images are used as the merge frame / overlay canvas, not as the primary detector input.

---

## 4. Stage 1 — Dual-pathway drawing split

**Code:** `dual-pathway-drawing-split/src/pipeline/`

**Architecture docs:** `dual-pathway-drawing-split/docs/architecture.md`, `docs/output-layout.md`.

### 4.1 Why two pathways

Bid PDFs mix:

- **Vector text** (sheet numbers, wing titles, room tags) that PyMuPDF can read exactly.
- **Raster linework** (walls, symbols) that needs a rendered JPEG.

Scanned or text-poor sheets need a vision model. Text-rich sheets should **not** trust the model’s names or boxes.

### 4.2 Path 1 — Vector / text (`sheet_scan.py`)

Reads the PDF text layer **per sheet, on its own terms** (sheet codes and titles vary wildly across sets).

Extracts:

- **Sheet codes** — e.g. `T-020`, `A1.1`
- **Wing titles** — largest type that is not notes/legend/key-plan boilerplate (`B-WING (EAST)`, `ADMIN BLDG`)
- **Room tags** — `B-13`, `C-19`, `AD-17A`; the **prefix** (`B`, `C`, `AD`) is used later to place a cut between wings that share walls

Rejects non-plan wording: KEY PLAN, LEGEND, NOTES, COVER SHEET, DETAIL, SCHEDULE, etc.

**Design rule:** PDF text labels **win over vision model names** when both exist.

### 4.3 Path 2 — Vision LLM (`llama_vision.py`)

Default runtime: **Groq** `qwen/qwen3.6-27b` (`GROQ_API_KEY`). Optional: Hugging Face endpoint or local Llama 3.2 Vision.

Two JSON tasks:

1. **Page classify** — `{has_diagram, plan_area[x1,y1,x2,y2], reason}`  
   True for floor/site/RCP/roof plans and similar. False for cover, index, legend-only, notes, typical details.

2. **Wing map** — `{wings: [{name, box}]}` on the **already cropped** diagram  
   Copy titles exactly as printed. One box per titled plan. Boxes may touch; should not heavily overlap.

**Design rule:** vision (and PDF text) provide **anchors only**. Their coordinates are too coarse to crop with. Earlier versions that cropped on model boxes landed on the wrong wing.

### 4.4 CV boundary engine (OpenCV owns edges)

| Step | Module | What it does |
|------|--------|----------------|
| Render | `page_render.py` | PDF page → JPEG |
| Diagram crop | `diagram_crop.py` | Remove title block, notes, legend, key-plan text panels. Extent = union of plan bodies at each wing label. |
| Wing split | `wing_crop.py` + `ink_profile.py` | Each title claims the linework block under it. If several titles share one block, cut at: drawn rule between them → room-tag clusters → **sparsest ink** in the span. |
| Quality gates | same | Reject blank/sliver crops (min ink, min plan-linework %, min side px). |
| Zoom export | `roi_zoom.py` | Ink-tight ROI inside the wing, then overlapping tiles. |

**Zoom tile contract (critical for counting)**

- Tile size: **588 px** (default; 10% closer than the previous 653 px)
- Overlap: **10%**
- Skip nearly blank tiles (`min ink ≈ 0.002`)
- Always write:
  - `plan_zoom_r##_c##.jpg`
  - `full_wing.jpg` (or equivalent ROI frame)
  - `zooms_manifest.json` with each tile’s `bbox_roi`

Without the manifest, symbol-count merge alignment is guessed from overlap snap and is marked low-confidence.

### 4.5 Output contract

```
storage/jobs/{pdf_stem}/
  01_source/{original.pdf}
  02_pages/page_NNN.jpg
  03_drawings/page_NNN_diagram.jpg
  04_wings/{WING-SLUG}/page_NNN.jpg
  04_wings/{WING-SLUG}/zooms/page_NNN/
      plan_zoom_*.jpg
      full_wing.jpg
      zooms_manifest.json
  05_metadata/page_NNN.json
  summary.json
  job.json
```

v1 **does not** export per-room crops. Room tags only help place wing boundaries.

---

## 5. Stage 2 — Legend symbol counting

**Code:** `symbol-count-vision/pipeline/`  
**Entry:** FastAPI `app.py` (port 8767)  
**Core:** `symbol_count.py`, `tile_merge.py`, `llm_verify.py`, `legend_reference.py`

### 5.1 Inputs

1. **Symbol file** — technical-symbol PDF (preferred) or `symbol_table.json`
2. Either:
   - one `plan_zoom_*.jpg`, or
   - a zoom **folder** from stage 1 (all tiles + `full_wing.jpg` + `zooms_manifest.json`)

### 5.2 Legend catalog

`pipeline/extraction/symbol_table_extractor.py` parses several legend layouts:

- Technical-symbol summary tables (glyph cell + description + tag + part)
- Plan-sheet “SYMBOLS” / “ABBREVIATIONS” sections
- Numbered legends
- Spatial technology-legend tables

Each row becomes a `SymbolEntry`:

- `symbol` — short tag (`#`, `J`, `AP`, `WP`)
- `description` — e.g. `DATA PERMANENT LINK`, `NETWORK CAMERA`
- `part_number` / `mfg_model` — e.g. `WM5400`
- `symbol_image_png` — cropped glyph artwork when present

**Count key** (`tag_match.count_key`): the display identity used in all counts. Glyph PNGs are saved under `07_glyphs/`.

**Persistent reference legends** (`legend_reference.py`):

- Built once from a technical-symbol PDF (`scripts/build_reference_legend.py`).
- Stored under `storage/reference_legends/<name>/` as glyph PNGs + `descriptions.json` (vision-generated shape descriptions; hand edits preserved).
- At job time a stored description is reused **only if the entry provably matches**:
  1. SHA-1 of the glyph PNG is identical, **or**
  2. key + description + tag + part all match.
- Key-only reuse is forbidden: one project’s `#` artwork must not describe another project’s `#`.

### 5.3 Detection philosophy

This is **not** open-ended object detection.

- The legend is a **closed vocabulary**. Anything not in the legend is dropped.
- **Localization** (the bounding box) is always a CV contour or Tesseract word box. An LLM never supplies a box that is used as geometry.
- **Classification** is a `ContextRule` on `pipeline/taxonomy/` first; vision is a label-only choice among 2–3 keys when rules cannot resolve.
- Linear / run items are **never** counted as discrete glyphs (`count_semantics: linear_suppressed`). Raceway **part-number text** is counted via `rect_digit_label`.

### 5.4 Per-image detector stack

1. `pipeline/detect/candidates.py` — filled triangles, hourglass pairs, bowtie squares, enclosing circles, Tesseract tags, rect+digit clusters, conduit stubs. No legend key yet.
2. `pipeline/classify/context_resolver.py` — apply taxonomy `ContextRule` (AP = triangle in circle; hourglass pair = one camera; bowtie = DATA POLE; bare triangle = `#`; collinear J's = J-HOOK).
3. `pipeline/classify/vision_classify.py` — ambiguous crops only; tight crop + glyph refs; returns a key or `none`, never coordinates.
4. Cross-tile NMS in ROI space (`SYMBOL_COUNT_NMS_IOU` default **0.3**).
5. `pipeline/count/finalize.py` — `discrete` / `qty_expand` / `linear_suppressed` / `cluster_as_one`.

Look-alike rules live only in the taxonomy (`ContextRule`). They are not restated in NMS suppressors or LLM prompt prose.

### 5.5 Folder pipeline — `count_symbols_on_zoom_folder`

Never stitch first. Detect independently on every tile, map via `bbox_roi`, NMS merge, optional glyph/judge eval, finalize counts. Overlay on `full_wing`.

### 5.6 Count semantics (`pipeline/count/finalize.py`)

Driven by taxonomy `count_semantics`:

- `discrete` — one mark = one count.
- `qty_expand` — `# = QTY`: adjacent digit expands the jack count; overlay stays one box.
- `linear_suppressed` — count stays 0 (continuous conduit/raceway runs).
- `cluster_as_one` — J-HOOK rows and hourglass pairs already merged in the resolver.

LLM count-verify does **not** add instances without a CV box. Fail toward CV.

### 5.7 Vision LLM roles

| Call | Job |
|------|-----|
| Vision classify | Ambiguous tight crops only. Label choice from 2–3 keys. **No boxes.** |
| Vision OCR fallback | Low-confidence Tesseract crop → text only, keep the CV/OCR box. |
| Glyph describe | One cached call per legend sheet (shape description). |
| Per-crop judge | Merge-eval confirmation on an existing CV box. |

There is no symbol-locate call and no LLM-proposed bounding box anywhere in the count path.

---

## 6. Domain conventions the pipeline encodes

These are treated as **mandatory literacy**, not optional hints. They come from real Garland-style telecom bid sets.

### 6.1 What a `#` is

- Small **standalone solid filled triangle** (any orientation) on/near a wall or raceway, often with a free digit 1–9 beside it.
- Digit = jack quantity (`# = QTY`).
- **Not** a direction arrow / north arrow / door swing.
- Pipe callouts `1|5`, `1|13` **locate** the same drops; they are not a second legend symbol.

### 6.2 What a `#` is not (look-alike triangles)

| Shape | True class | Why people confuse it |
|-------|------------|------------------------|
| Filled triangle inside a circle-crosshair | AP (access point) | Inner triangle looks like a drop |
| Two opposing triangles + X-box (hourglass) | NETWORK CAMERA (qty 1) | Looks like two drops |
| Square with opposing filled triangles (bowtie) | DATA POLE | Inner triangles look like drops |
| Filled triangle on a hollow camera-body rectangle | Camera glyph | Not a drop |

### 6.3 Letter tags

- `J`, `G`, `AP`, `WP`, `TGB`, `NVR`, … — one count per distinct inked instance matching the legend tag.
- **J vs J-HOOK:** a **row of J’s on one horizontal line** (`—J—J—J—`) is **one J-HOOK**, not many junction boxes. Isolated single `J` stays JUNCTION BOX.
- Implemented as `ContextRule.requires_collinear_min` on the J-HOOK taxonomy entry (`pipeline/classify/context_resolver.py`).

### 6.4 Surface raceway

- Open rectangle on the wall/raceway + digits `2300` / `5400` / `5500` (often **vertical**).
- Map digits-only labels to `SURFACE RACEWAY (WMxxxx)` **without inventing the letters WM** if they are not printed.
- Count **one** per box+label cluster, not box and digits separately.

### 6.5 Conduit stub vs letter E

- Stub = capital **E whose middle bar is much longer**, sitting on conduit.
- Plain `E`, `EAST`, room code `A-6E` are **not** stubs.
- CV contour test: middle-band ink on the right half ≫ top/bottom bands.

### 6.6 Never count as equipment

- Elevations AFF: `+48`, `+49`
- Room / space IDs: `A-1`, `A-6E`, `RR/B`, circled `1.03`
- KEY PLAN inset, wing titles, grid bubbles, dimension strings
- Boxed lone keynote digit (usually a keyed note, not a jack)
- Break-line zigzags, hatch, door swings, furniture

### 6.7 Dense wall runs

A few inches of wall can contain: raceway box + `5400` + `+48` + filled triangle + QTY digit + keynote `1`.  
Parse **roles separately**. Count only legend keys. Example: triangle+`2` → one `#` mark (qty 2); rectangle+`5400` → one WM5400; `+48` → ignore.

---

## 7. What is deliberately not done yet

| Item | Status |
|------|--------|
| Dollar cost / unit pricing / labor | Not implemented |
| HVAC equipment takeoff | Out of this stack (`hvac-cost-estimator` mentioned only as sibling) |
| Per-room crops | v2 idea; v1 is wing-level |
| DWG/DXF vector path | Future |
| BIM export | Future |
| Detecting on a full stitched wing | Explicitly rejected (too dense / too small) |
| Open-vocabulary “find all symbols” | Rejected; legend is closed set |
| Using vision boxes as crop edges | Rejected; OpenCV owns edges |
| Counting linear conduit/raceway **length** | Linear keys forced to 0; only discrete labels counted |

---

## 8. Design principles (keep these if you redesign)

1. **Text layer for names, CV for geometry, vision for recall and disambiguation.** Do not let the LLM invent crop edges or legend keys.
2. **Closed vocabulary.** Only uploaded-legend keys may appear in counts.
3. **Tile-first counting.** Overlap must be merged in a shared ROI frame using a manifest, not by stitching then detecting.
4. **One physical instance, one count.** Overlapping CV+vision boxes on the same ink are one object.
5. **Look-alikes are first-class.** J-HOOK vs J, stub vs E, `#` vs AP/camera/pole triangles are encoded in both CV and prompts.
6. **Fail toward CV.** If the LLM errors or returns 0 against a CV hit, keep CV (except explicit fill-miss / unique-instance rules).
7. **Glyph descriptions are per-artwork, not per-key.** Different sheets reuse `#` / `AP` with different drawings.
8. **QTY is a drawing convention, not a second symbol.** Overlay boxes = marks; Count column may be expanded jacks.

---

## 9. Known pain points (why a clearer approach may be needed)

These are the places the current stack is heaviest / most brittle:

1. **Prompt + heuristic pile-up (fixed).** Look-alike rules now live only in the taxonomy `ContextRule` JSON.
2. **Duplicate drawing-split copies (fixed).** Canonical Stage 1 is `dual-pathway-drawing-split/` only; it does not embed symbol counting.
3. **API cost / latency** on folder jobs is lower: vision classify runs only on ambiguous crops, not per-tile locate.
4. **Merge without `zooms_manifest.json`** is still a guessed grid; counts can double or vanish at tile seams.
5. **Tiny ink at 653 px** still loses some vertical `5400` labels and 1-letter tags; Tesseract + crop OCR fallback is the recovery path.
6. **Camera / drop / AP triangle confusion** is encoded as ContextRules; eval on the synthetic look-alike set is the regression gate.
7. **No quantity-takeoff → price table yet**, so “cost estimation” in the repo name is still a future stage.
8. **Legend parser** is still a special-case machine for several PDF table styles.
9. **Isolated letter tags** (single `J`) still depend on Tesseract recall.

---

## 10. Models and important knobs

| Knob | Default | Meaning |
|------|---------|---------|
| Zoom tile | 588 px, 10% overlap | Stage 1 export (10% closer than the previous 653 px) |
| `CV_TEMPLATE_THRESHOLD` | 0.72 | Glyph match |
| `SYMBOL_COUNT_NMS_IOU` | 0.3 | Class NMS (tighter; boxes are CV contours) |
| `LLM_PROVIDER` | gemini (locked; Groq/Llama unused) | Vision backend |
| `SYMBOL_COUNT_USE_LLM` | true | Master LLM switch (classify ambiguous only) |
| `FOLDER_USE_VISION_OCR` | false | Legacy per-tile locate; unused |
| `FOLDER_VISION_COUNT_VERIFY` | false | Notes-only; does not invent boxes |
| `VISION_CLASSIFY_MARGIN_PX` | 8 | Context pad around CV crop for classify |
| `TESSERACT_CMD` | PATH | Required for CV OCR |

---

## 11. What we want from a clearer approach

Please propose an architecture that:

1. Preserves the **closed-legend, tile-first, CV-owns-crops** constraints unless you have a strong reason to drop one — and say why.
2. Collapses duplicated look-alike / `#=QTY` / raceway rules into **one source of truth** used by both CV and any LLM.
3. Makes **stage 1 (split)** and **stage 2 (count)** a single explicit contract (manifest schema, legend schema, count schema) instead of three near-copies.
4. Reduces vision calls without losing recall on tiny tags and `#` drops.
5. Separates **marks vs QTY vs priced units** so a later cost table can attach unit prices cleanly.
6. States how you would evaluate on `Real data/` bid PDFs (per-symbol precision/recall, not just “looks right”).
7. Does **not** assume DWG/DXF or BIM unless that is clearly better; current input is PDF plots.

Be concrete: pipeline stages, data objects, which detector owns which symbol class, and what the LLM is allowed to do vs forbidden to do.
