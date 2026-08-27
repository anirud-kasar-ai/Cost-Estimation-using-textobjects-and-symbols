from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import config
from pipeline.cv.nms import (
    BoundingBox,
    SymbolDetection,
    nms_by_class,
    suppress_cross_label,
    suppress_cross_source_duplicates,
    suppress_hourglass_on_triangles,
    suppress_nearby_same_label,
    suppress_triangles_near_tags,
    suppress_triangles_near_templates,
)
from pipeline.cv.lookalike import (
    conduit_stub_legend_key,
    detect_conduit_stubs,
    reclassify_j_hook_clusters,
)
from pipeline.cv.ocr_tags import (
    ap_skip_centers_from_ocr,
    clock_combo_skip_centers,
    detect_tags_from_ocr,
    tesseract_available,
)
from pipeline.cv.tag_match import count_key, tagged_legend_keys, text_legend_keys
from pipeline.cv.template_match import detect_symbols_from_glyphs
from pipeline.cv.triangle_drops import (
    detect_data_pole_boxes,
    detect_drop_marks,
    device_glyph_skip_centers,
)
from pipeline.overlay import draw_detections_overlay
from pipeline.symbol_legend import (
    SymbolTableInfo,
    parse_symbol_file,
)


# Overlap + dense-zoom guidance injected into the vision count verifier.
# Keeps CV and vision mapped to one physical symbol when boxes touch.
SYMBOL_COUNT_OVERLAP_PROMPT = """
OVERLAP / DENSE ZOOM (critical — do not double-count):
- Zoom tiles and wing crops are dense: symbols, QTY digits, pipe tags (1|5, 1|13),
  elevations (+48), and letter tags often sit within a few pixels of each other.
- Count each PHYSICAL symbol or tag ONCE. If two boxes / two detectors (CV + vision)
  cover the same ink, that is ONE instance — never 2.
- A solid drop triangle + its adjacent QTY digit (+ optional leader) = ONE '#' mark
  (digit is quantity, not a second symbol).
- A device glyph with its letter tag under/beside it (AP under circle, WP by horn) =
  ONE equipment instance for that tag — do not count the glyph and the letters twice.
- Clustered wall runs of the same tag (e.g. several J along a path) count separately
  only when centers are clearly distinct and not the same overlapping ink.
  Exception: two or more J's on one horizontal line are ONE J-HOOK, not many J-boxes.
- Ignore elevations like +48 / +49, sheet callouts like 1/D5, KEY PLAN text, and
  room labels (A-1, A-2, A-3, RR/B) — they are not legend equipment counts.
- Pipe callouts (1|5, 1|8, 1|13) are keynotes tied to nearby drops; do not invent
  extra '#' marks from the pipe numbers alone if the triangle is already counted.
- Prefer unique spatial centers: if CV already lists a hit near a location, do not
  add another for the same legend key at that same spot.
""".strip()

SYMBOL_COUNT_VISION_MAP_PROMPT = """
CV ↔ VISION MAPPING:
- CV detections (OCR, vision OCR, template, triangles, hourglass) are location hints.
- Your job: (1) confirm real symbols CV found, (2) ADD legend symbols CV missed that
  are clearly visible, (3) REMOVE CV false positives (blank wall, hatch, dimensions).
- When CV count > 0 and you see the same symbols, keep a count that matches unique
  physical instances (may be lower than CV if CV double-fired on overlap).
- When CV count = 0 but you clearly see alphabetic tags (AP, WP, J, …) or numeric
  raceway labels (2300/5400/5500), report the correct positive count — fill CV misses.
- Never invent symbols that are not inked on the plan.
""".strip()

# Domain knowledge for construction / low-voltage CAD bid sets (Garland-style).
CAD_CONSTRUCTION_SYMBOL_KNOWLEDGE = """
CONSTRUCTION CAD / LOW-VOLTAGE PLAN LITERACY (mandatory):

You are reading a contractor bid-set technology floor plan (AutoCAD/Revit plot),
NOT a schematic diagram and NOT a photograph. Thin 1–2 px vector ink is real.

HOW TO READ THIS DRAWING FAMILY:
1) '#' DATA PERMANENT LINK (data drop / jack cluster):
   - LOOKS LIKE: small SOLID FILLED black triangle (any orientation: left/right/up/down)
     sitting on or next to a wall/raceway line, often with a free digit 1–9 beside it.
   - MEANING: one physical drop mark. Digit = QTY of jacks (# = QTY convention).
   - COUNT: one mark per triangle. Do NOT call these "direction arrows" or "arrowheads"
     on telecom plans — filled triangles next to raceway/walls are data drops.
   - RELATED: framed keynotes "1|13", "1 13", "1|5" (or stacked 1 over 13) locate the
     same drops; they are NOT a separate legend symbol and NOT raceway model numbers.

2) SURFACE RACEWAY (Wiremold WM2300 / WM5400 / WM5500):
   - LOOKS LIKE: small open rectangle / box ON the wall or raceway linework, often with
     digits 2300 / 5400 / 5500 printed beside it (frequently VERTICAL).
   - MEANING: one surface raceway device/section labeled with that series.
   - COUNT: one instance per distinct raceway BOX + its part-number label cluster.
     Do not count the digit string and the box as two. Do not invent WM letters if
     only 5400 is printed — map 5400 → SURFACE RACEWAY (WM5400) legend key.

3) LETTER EQUIPMENT TAGS (J, G, AP, WP, TGB, NVR, …):
   - LOOKS LIKE: short upright or rotated capital letters on/near walls or under glyphs
     (AP under a circle; WP by a horn/speaker; a SINGLE J along a cable path).
   - COUNT: one per distinct letter-tag instance matching the legend tag.
   - J vs J-HOOK: a row of J's on one line (—J—J—J—) is J-HOOK, not junction boxes.

4) NETWORK CAMERA:
   - LOOKS LIKE: two opposing solid triangles with a small X-box between (hourglass),
     often with QTY digits outside the tips — that assembly is ONE camera, not drops.

4b) DATA POLE:
   - LOOKS LIKE: square with two opposing filled triangles (bowtie). Not '#' drops.

4c) CONDUIT STUB:
   - LOOKS LIKE: letter E with a long middle bar on conduit. Not the word EAST.

5) NEVER COUNT AS LEGEND EQUIPMENT:
   - Elevations AFF: "+48", "+49" (mounting height only).
   - Room / space IDs: A-1, A-2, 1.03 in a circle, RR/B, RR/G.
   - KEY PLAN inset, sheet title "A-WING (EAST)", grid bubbles, dimension strings.
   - Boxed keynote digits alone (square with "1") unless they are clearly a legend
     equipment tag — usually they are keyed notes, not jacks or raceway.
   - Break-line zigzags, hatch, door swings, furniture outlines.

6) DENSE WALL RUNS (typical zoom tile):
   - A single wall can show: raceway box + "5400" + "+48" + filled triangle + QTY digit
     + keynote "1" all within inches. Parse each role separately; count only legend keys.
   - Example: triangle + "2" → one '#' mark (qty 2). Rectangle + "5400" → one WM5400.
     "+48" → ignore. Square "1" keynote → ignore unless legend says otherwise.
""".strip()

_SOURCE_LABELS = {
    "template": "template match",
    "ocr": "text/OCR",
    "vision_ocr": "vision OCR",
    "vision_detect": "vision locate",
    "triangle": "drop marks",
    "callout_drop": "callout drops",
    "hourglass": "camera marks",
    "bowtie": "data pole marks",
    "jhook": "j-hook runs",
    "stub": "conduit stub marks",
}


def legend_count_keys(legend: SymbolTableInfo) -> set[str]:
    """All display keys that may appear in counts (legend only)."""
    return {count_key(e) for e in legend.entries if count_key(e)}


def filter_to_legend(
    detections: list[SymbolDetection],
    valid_keys: set[str],
) -> list[SymbolDetection]:
    return [d for d in detections if d.symbol in valid_keys]


def _entry_method(
    entry: Any,
    display: str,
    *,
    detected_sources: dict[str, set[str]],
    linear_keys: set[str],
) -> str:
    """How this legend row is (or would be) detected on the plan."""
    sources = detected_sources.get(display)
    if sources:
        return " + ".join(
            sorted(_SOURCE_LABELS.get(s, s) for s in sources)
        )
    part = (entry.part_number or "").strip()
    from pipeline.cv.tag_match import _part_aliases

    if _part_aliases(part):
        return "text/OCR (part label)"
    if display in linear_keys:
        return "excluded (linear item)"
    glyph_source = getattr(entry, "glyph_source", None)
    if entry.symbol_image_png and glyph_source is None:
        return "template (from file)"
    if entry.symbol_image_png and glyph_source == "library":
        return "template (symbol library)"
    if (entry.symbol or "").strip():
        return "text/OCR"
    return "no graphic in file"


def _build_rows(
    legend: SymbolTableInfo,
    counts: dict[str, int],
    cv_counts: dict[str, int] | None = None,
    *,
    mark_counts: dict[str, int] | None = None,
    detected_sources: dict[str, set[str]] | None = None,
    linear_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per unique count_key (legend PDFs often repeat the same glyph)."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in legend.entries:
        display = count_key(entry)
        if not display or display in seen:
            continue
        seen.add(display)
        count = int(counts.get(display, 0))
        marks = int((mark_counts or {}).get(display, count))
        row = {
            "symbol": entry.symbol or "",
            "display_key": display,
            "description": entry.description,
            "mfg_model": entry.mfg_model or "",
            "part_number": entry.part_number or "",
            "marks": marks,
            "count": count,
            "method": _entry_method(
                entry,
                display,
                detected_sources=detected_sources or {},
                linear_keys=linear_keys or set(),
            ),
        }
        if cv_counts is not None:
            row["cv_count"] = int(cv_counts.get(display, 0))
        rows.append(row)
    rows.sort(key=lambda r: (-r["count"], str(r["symbol"]).upper(), r["description"]))
    return rows


def _is_camera_label(label: str) -> bool:
    upper = (label or "").upper()
    return "CAMERA" in upper or upper == "MOUNT" or upper.startswith("MOUNT ")


def _filter_camera_templates_near_geometry(
    detections: list[SymbolDetection],
    camera_boxes: list[dict[str, Any]],
    *,
    max_center_dist: float = 90.0,
) -> list[SymbolDetection]:
    """Remove camera-family *template* hits.

    Geometric hourglass / X-box detection is the source of truth. Template
    matching on camera glyphs false-fires on keynote boxes (``1|13``) and
    wall junctions — even at high thresholds on busy plan zooms.
    """
    del camera_boxes, max_center_dist  # geometry-only policy
    return [
        d
        for d in detections
        if not (_is_camera_label(d.symbol) and d.source == "template")
    ]


def _data_pole_display_key(legend: SymbolTableInfo) -> str | None:
    import re

    for entry in legend.entries:
        display = count_key(entry)
        if not display:
            continue
        blob = f"{display} {entry.description or ''}"
        if re.search(r"DATA\s+POLE", blob, re.I):
            return display
    return None


def _camera_display_key(legend: SymbolTableInfo) -> str | None:
    """Legend key for boxed-hourglass camera marks drawn on the plan."""
    candidates: list[str] = []
    for entry in legend.entries:
        display = count_key(entry)
        if not display:
            continue
        desc = (entry.description or "").upper()
        if desc == "NETWORK CAMERA":
            return display
        if "CAMERA" in desc:
            candidates.append(display)
    return candidates[0] if candidates else None


def _aggregate_detection_counts(
    detections: list[SymbolDetection],
    hash_key: str | None,
) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    mark_counts: dict[str, int] = {}
    for det in detections:
        mark_counts[det.symbol] = mark_counts.get(det.symbol, 0) + 1
        counts[det.symbol] = counts.get(det.symbol, 0) + max(1, int(det.qty))
    return counts, mark_counts


def _detections_to_dict(
    detections: list[SymbolDetection],
    *,
    evaluations: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, det in enumerate(detections):
        item: dict[str, Any] = {
            "symbol": det.symbol,
            "score": det.score,
            "source": det.source,
            "qty": max(1, int(det.qty)),
            "box": {
                "x1": det.box.x1,
                "y1": det.box.y1,
                "x2": det.box.x2,
                "y2": det.box.y2,
            },
        }
        if det.tile_file:
            item["tile_file"] = det.tile_file
        if evaluations and i in evaluations:
            item["evaluation"] = evaluations[i]
        out.append(item)
    return out


def detect_symbols_raw(
    *,
    image: Image.Image,
    legend: SymbolTableInfo,
    glyph_dir: Path | None = None,
    symbol_file_suffix: str = ".json",
    nms_iou: float | None = None,
    apply_local_nms: bool = True,
    use_vision_ocr: bool | None = None,
    exclude_top_pct: float | None = None,
) -> tuple[list[SymbolDetection], list[str], dict[str, Any]]:
    """Run all detectors; return legend-gated detections (+ notes/meta).

    When ``apply_local_nms`` is True (single-image path), class NMS and
    cross-label suppression run here. Folder merge sets it False and applies
    NMS once in ROI space after translating all tiles.

    For zoom tiles, pass ``exclude_top_pct=0`` (no title-block band — the top
    of a tile is plan content). Folder jobs may set ``use_vision_ocr=True`` to
    run Gemini text OCR + symbol locate on every tile (costly, higher recall).
    """
    notes: list[str] = list(legend.notes or [])
    meta: dict[str, Any] = {
        "linear_keys": set(),
        "hash_key": None,
        "camera_key": None,
        "glyphs_available": False,
    }
    if not legend.entries:
        return [], ["Empty legend catalog"], meta

    nms_iou = float(nms_iou if nms_iou is not None else config.SYMBOL_COUNT_NMS_IOU)
    title_band = float(
        exclude_top_pct if exclude_top_pct is not None else config.CV_TITLE_BAND_PCT
    )
    run_vision = (
        bool(use_vision_ocr)
        if use_vision_ocr is not None
        else config.vision_ocr_enabled()
    )
    valid_keys = legend_count_keys(legend)
    raw_detections: list[SymbolDetection] = []
    linear_keys: set[str] = set()

    if not tesseract_available():
        notes.append(
            "Tesseract OCR is not installed — text-tag symbol detection was skipped. "
            "Install Tesseract and ensure it is on PATH."
        )
    else:
        raw_detections.extend(
            detect_tags_from_ocr(image, legend, exclude_top_pct=title_band)
        )

    if run_vision and config.vision_ocr_enabled():
        from pipeline.cv.ocr_tags import detect_tags_from_vision_ocr

        vision_hits = detect_tags_from_vision_ocr(
            image, legend, exclude_top_pct=title_band
        )
        raw_detections.extend(vision_hits)
        if vision_hits:
            notes.append(
                f"Vision OCR (architect prompt) added {len(vision_hits)} text tag hit(s)."
            )

        # Full legend symbol locate (geometry + tags) via Gemini/Groq.
        from pipeline.llm_verify import detect_legend_symbols_with_llm

        # linear_keys filled after glyphs below; precompute from descriptions now
        from pipeline.cv.template_match import LINEAR_GLYPH_RE

        pre_linear = {
            count_key(e)
            for e in legend.entries
            if LINEAR_GLYPH_RE.search(count_key(e) + " " + (e.description or ""))
            and (e.symbol or "").strip() != "#"
        }
        located = detect_legend_symbols_with_llm(
            image, legend, linear_keys=pre_linear
        )
        if located.get("ok") and located.get("detections"):
            y_min = image.size[1] * title_band
            n_add = 0
            for item in located["detections"]:
                if float(item["y1"]) < y_min:
                    continue
                sym = item["symbol"]
                if sym not in valid_keys:
                    continue
                kind = str(item.get("kind") or "")
                source = "vision_detect"
                qty = max(1, int(item.get("qty") or 1))
                if sym == "#" or kind == "drop":
                    source = "vision_detect"
                raw_detections.append(
                    SymbolDetection(
                        symbol=sym,
                        box=BoundingBox(
                            x1=float(item["x1"]),
                            y1=float(item["y1"]),
                            x2=float(item["x2"]),
                            y2=float(item["y2"]),
                        ),
                        score=max(0.55, min(1.0, float(item.get("confidence") or 0.7))),
                        source=source,
                        qty=qty,
                    )
                )
                n_add += 1
            if n_add:
                notes.append(
                    f"Vision symbol locate added {n_add} legend instance(s) "
                    f"({located.get('model')})."
                )
        elif located.get("error"):
            notes.append(f"Vision symbol locate skipped: {located['error']}")

    glyphs_available = bool(glyph_dir and glyph_dir.is_dir() and any(glyph_dir.glob("*.png")))
    meta["glyphs_available"] = glyphs_available
    if glyphs_available:
        raw_detections.extend(detect_symbols_from_glyphs(image, legend, glyph_dir))
        from pipeline.cv.template_match import LINEAR_GLYPH_RE

        linear_keys = {
            count_key(e)
            for e in legend.entries
            if LINEAR_GLYPH_RE.search(count_key(e) + " " + (e.description or ""))
            and (e.symbol or "").strip() != "#"
        }
        meta["linear_keys"] = linear_keys
        if linear_keys:
            notes.append(
                "Linear/run items are not counted by glyph template matching "
                "(they match every straight wall line). Part labels like "
                "WM2300/2300 on the plan are still counted via text/OCR: "
                + ", ".join(sorted(linear_keys))
            )
    elif symbol_file_suffix.lower() == ".json":
        notes.append(
            "Graphical template matching skipped: upload a PDF legend to extract glyph templates."
        )

    tags = tagged_legend_keys(legend)
    if not tags and not glyphs_available:
        notes.append(
            "Legend has no text tags and no glyph templates — nothing to match on the plan. "
            "Use symbol_table.json or a technical symbol PDF that includes embedded glyphs."
        )
    hash_key = next((display for tag, display in tags if tag == "#"), None)
    camera_key = _camera_display_key(legend)
    pole_key = _data_pole_display_key(legend)
    meta["hash_key"] = hash_key
    meta["camera_key"] = camera_key
    meta["pole_key"] = pole_key
    camera_boxes: list[dict[str, Any]] = []
    callout_hints: list[dict[str, Any]] = []
    if hash_key:
        from pipeline.cv.callout_boxes import detect_pipe_callout_hints
        from pipeline.cv.ocr_tags import collect_pipe_callout_ocr_hints

        callout_hints = detect_pipe_callout_hints(
            image, exclude_top_pct=title_band
        )
        callout_hints.extend(
            collect_pipe_callout_ocr_hints(image, exclude_top_pct=title_band)
        )
    if hash_key or camera_key:
        skip = ap_skip_centers_from_ocr(
            [d for d in raw_detections if d.source in {"ocr", "vision_ocr", "template"}]
        )
        skip.extend(device_glyph_skip_centers(image, exclude_top_pct=title_band))
        skip.extend(clock_combo_skip_centers(image, exclude_top_pct=title_band))
        for pole in detect_data_pole_boxes(image, exclude_top_pct=title_band):
            skip.append((float(pole["cx"]), float(pole["cy"])))
        protect = [(float(h["cx"]), float(h["cy"])) for h in callout_hints]
        drop_boxes, camera_boxes = detect_drop_marks(
            image,
            skip_centers=skip,
            exclude_top_pct=title_band,
            protect_centers=protect or None,
        )
        if hash_key:
            for box in drop_boxes:
                raw_detections.append(
                    SymbolDetection(
                        symbol=hash_key,
                        box=BoundingBox(
                            x1=box["x1"],
                            y1=box["y1"],
                            x2=box["x2"],
                            y2=box["y2"],
                        ),
                        score=0.85,
                        source="triangle",
                        qty=int(box.get("qty") or 1),
                    )
                )
        if camera_key:
            for box in camera_boxes:
                raw_detections.append(
                    SymbolDetection(
                        symbol=camera_key,
                        box=BoundingBox(
                            x1=box["x1"],
                            y1=box["y1"],
                            x2=box["x2"],
                            y2=box["y2"],
                        ),
                        score=0.8,
                        source="hourglass",
                    )
                )
        if pole_key:
            for box in detect_data_pole_boxes(image, exclude_top_pct=title_band):
                raw_detections.append(
                    SymbolDetection(
                        symbol=pole_key,
                        box=BoundingBox(
                            x1=box["x1"],
                            y1=box["y1"],
                            x2=box["x2"],
                            y2=box["y2"],
                        ),
                        score=0.8,
                        source="bowtie",
                    )
                )
        if hash_key:
            from pipeline.cv.callout_boxes import reinforce_hash_from_callouts

            raw_detections = reinforce_hash_from_callouts(
                image=image,
                hash_key=hash_key,
                existing=raw_detections,
                callout_hints=callout_hints,
                exclude_top_pct=title_band,
            )
        if tesseract_available() and drop_boxes and text_legend_keys(legend):
            from pipeline.cv.ocr_tags import _ocr_part_labels_near_centers
            from pipeline.cv.tag_match import text_legend_keys as _tlk

            raw_detections.extend(
                _ocr_part_labels_near_centers(
                    image,
                    _tlk(legend),
                    [(b["cx"], b["cy"]) for b in drop_boxes],
                    image.size[1] * title_band,
                )
            )

    elif pole_key:
        for box in detect_data_pole_boxes(image, exclude_top_pct=title_band):
            raw_detections.append(
                SymbolDetection(
                    symbol=pole_key,
                    box=BoundingBox(
                        x1=box["x1"],
                        y1=box["y1"],
                        x2=box["x2"],
                        y2=box["y2"],
                    ),
                    score=0.8,
                    source="bowtie",
                )
            )

    # Camera templates alone are unreliable (match keynotes / junctions).
    raw_detections = _filter_camera_templates_near_geometry(
        raw_detections, camera_boxes
    )
    raw_detections = suppress_hourglass_on_triangles(raw_detections)
    raw_detections = suppress_triangles_near_tags(raw_detections)
    raw_detections = suppress_triangles_near_templates(raw_detections)
    stub_key = conduit_stub_legend_key(legend)
    if stub_key:
        for box in detect_conduit_stubs(image, exclude_top_pct=title_band):
            raw_detections.append(
                SymbolDetection(
                    symbol=stub_key,
                    box=BoundingBox(
                        x1=box["x1"],
                        y1=box["y1"],
                        x2=box["x2"],
                        y2=box["y2"],
                    ),
                    score=0.78,
                    source="stub",
                )
            )
    raw_detections = reclassify_j_hook_clusters(raw_detections, legend)
    raw_detections = filter_to_legend(raw_detections, valid_keys)

    if apply_local_nms:
        kept = nms_by_class(raw_detections, iou_threshold=nms_iou, center_dist=28.0)
        kept = suppress_cross_label(kept, iou_threshold=0.6, source="template")
        kept = suppress_nearby_same_label(kept, source="ocr", max_center_dist=80.0)
        kept = suppress_nearby_same_label(kept, source="vision_ocr", max_center_dist=80.0)
        kept = suppress_nearby_same_label(kept, source="vision_detect", max_center_dist=80.0)
        kept = suppress_cross_source_duplicates(
            kept,
            sources={"ocr", "vision_ocr", "vision_detect", "template"},
            max_center_dist=36.0,
            iou_threshold=0.15,
        )
        kept = filter_to_legend(kept, valid_keys)
        return kept, notes, meta

    return raw_detections, notes, meta


def _finalize_counts(
    *,
    legend: SymbolTableInfo,
    kept: list[SymbolDetection],
    notes: list[str],
    meta: dict[str, Any],
    image: Image.Image | None = None,
    method: str = "single_image_cv",
    raw_detection_count: int | None = None,
    nms_iou: float | None = None,
    extra: dict[str, Any] | None = None,
    pre_eval_detections: list[SymbolDetection] | None = None,
    detection_items: list[dict[str, Any]] | None = None,
    rejected_detection_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    valid_keys = legend_count_keys(legend)
    kept = filter_to_legend(kept, valid_keys)
    nms_iou = float(nms_iou if nms_iou is not None else config.SYMBOL_COUNT_NMS_IOU)
    hash_key = meta.get("hash_key")
    linear_keys: set[str] = set(meta.get("linear_keys") or set())
    glyphs_available = bool(meta.get("glyphs_available"))

    pre_source = pre_eval_detections if pre_eval_detections is not None else kept
    cv_counts, cv_mark_counts = _aggregate_detection_counts(pre_source, hash_key)
    counts, mark_counts = _aggregate_detection_counts(kept, hash_key)

    qty_summed = False
    qty_breakdown: str | None = None
    if hash_key:
        marks = [
            d
            for d in kept
            if d.symbol == hash_key
            and d.source in {"triangle", "callout_drop", "vision_detect"}
        ]
        qty_total = sum(max(1, int(d.qty)) for d in marks)
        if marks and qty_total > len(marks):
            qty_summed = True
            digits = "+".join(str(max(1, int(d.qty))) for d in marks)
            qty_breakdown = (
                f"{len(marks)} drop mark(s) × QTY digits ({digits}) = {qty_total} jacks"
            )
            notes.append(
                f"'#' uses the drawing's '# = QTY' convention: {qty_breakdown}. "
                f"Overlay boxes = {len(marks)}; Count column = {qty_total}."
            )

    llm_summary: dict[str, Any] | None = None
    folder_vision = (
        method == "detect_merge_evaluate_count" and config.FOLDER_VISION_COUNT_VERIFY
    )
    single_vision = method == "single_image_cv"
    if (
        image is not None
        and config.llm_enabled()
        and (single_vision or folder_vision)
    ):
        from pipeline.llm_verify import verify_counts_with_llm

        detection_hints = [
            {
                "symbol": d.symbol,
                "source": d.source,
                "qty": max(1, int(d.qty)),
                "cx": round((d.box.x1 + d.box.x2) / 2.0, 1),
                "cy": round((d.box.y1 + d.box.y2) / 2.0, 1),
            }
            for d in kept
        ]
        llm = verify_counts_with_llm(
            image,
            legend,
            counts,
            linear_keys=linear_keys,
            extra_rules=[
                CAD_CONSTRUCTION_SYMBOL_KNOWLEDGE,
                SYMBOL_COUNT_OVERLAP_PROMPT,
                SYMBOL_COUNT_VISION_MAP_PROMPT,
            ],
            detection_hints=detection_hints,
            extra_429_retries=(
                int(config.FOLDER_VISION_429_RETRIES) if folder_vision else 0
            ),
        )
        llm_summary = {
            "used": llm["ok"],
            "provider": llm.get("provider"),
            "model": llm["model"],
            "notes": llm.get("notes"),
            "error": llm.get("error"),
            "mode": "folder_post_merge" if folder_vision else "single_image",
        }
        if llm["ok"] and llm["counts"]:
            disagreements: list[str] = []
            llm_added: list[str] = []
            for key, llm_value in llm["counts"].items():
                if key not in valid_keys or key in linear_keys:
                    continue
                cv_value = counts.get(key, 0)
                cv_marks = mark_counts.get(key, 0)
                if key == hash_key:
                    mark_n = mark_counts.get(key, 0)
                    if qty_summed:
                        if llm_value and llm_value != cv_value:
                            notes.append(
                                f"LLM counted {llm_value} visible '#' mark(s); "
                                f"kept QTY-expanded jack count {cv_value} "
                                f"from {mark_n} mark(s)."
                            )
                        continue
                    if folder_vision and mark_n > 0:
                        # Vision fills CV misses: adopt the higher Gemini count;
                        # keep CV when the model saw fewer (NMS-merged) marks.
                        if llm_value > mark_n:
                            counts[key] = llm_value
                            notes.append(
                                f"Folder vision raised '#' mark count "
                                f"{mark_n} → {llm_value} (Gemini count adopted)."
                            )
                        elif llm_value and llm_value != cv_value:
                            notes.append(
                                f"LLM saw {llm_value} '#' mark(s); kept CV {cv_value}."
                            )
                        continue
                    if mark_n == 0 and llm_value > 0:
                        counts[key] = llm_value
                        llm_added.append(f"{key}={llm_value}")
                        continue
                    if not folder_vision and llm_value > 0:
                        counts[key] = llm_value
                    continue
                if llm_value > 0:
                    if cv_marks == 0 and cv_value == 0:
                        # Vision fill for CV misses (alphabetic / numeric tags).
                        counts[key] = llm_value
                        llm_added.append(f"{key}={llm_value}")
                        continue
                    # Prefer unique-instance count: take LLM when it lowers
                    # double-counts, or raises when CV under-detected.
                    counts[key] = llm_value
                elif cv_value > 0:
                    disagreements.append(key)
            counts = {k: v for k, v in counts.items() if v > 0 and k in valid_keys}
            method = (
                "detect_merge_evaluate_count_llm"
                if folder_vision
                else "cv_plus_llm_verify"
            )
            if llm_added:
                notes.append(
                    "Vision filled CV misses (unique instances): " + ", ".join(llm_added)
                )
            if disagreements:
                notes.append(
                    "LLM reported 0 for CV-detected symbols (kept CV counts): "
                    + ", ".join(sorted(disagreements))
                )
            if llm.get("notes"):
                notes.append(f"LLM: {llm['notes']}")
        elif llm.get("error"):
            notes.append(f"LLM verification skipped: {llm['error']}")
        elif folder_vision:
            notes.append("Folder post-merge vision count verify ran but returned no counts.")

    counts = {k: v for k, v in counts.items() if k in valid_keys and v > 0}
    cv_counts = {k: v for k, v in cv_counts.items() if k in valid_keys}
    mark_counts = {k: v for k, v in mark_counts.items() if k in valid_keys}

    detected_sources: dict[str, set[str]] = {}
    for det in kept:
        detected_sources.setdefault(det.symbol, set()).add(det.source)

    rows = _build_rows(
        legend,
        counts,
        cv_counts if (llm_summary or pre_eval_detections is not None) else None,
        mark_counts=mark_counts,
        detected_sources=detected_sources,
        linear_keys=linear_keys,
    )
    missing_graphics = [
        r["display_key"] for r in rows if r["method"] == "no graphic in file"
    ]
    if missing_graphics:
        notes.append(
            "No symbol graphic is available for: "
            + ", ".join(sorted(missing_graphics))
            + ". These cannot be counted graphically — upload a legend PDF "
            "that shows the symbol artwork."
        )

    box_total = len(kept)
    unit_total = sum(int(v) for v in counts.values())
    payload: dict[str, Any] = {
        "status": "done",
        "method": method,
        "legend_entry_count": legend.entry_count,
        "raw_detection_count": (
            int(raw_detection_count) if raw_detection_count is not None else box_total
        ),
        "post_nms_count": box_total,
        "unit_total": unit_total,
        "qty_summed": qty_summed,
        "qty_breakdown": qty_breakdown,
        "nms_iou": nms_iou,
        "tesseract_available": tesseract_available(),
        "glyph_templates_used": glyphs_available,
        "llm": llm_summary,
        "cv_counts": cv_counts,
        "mark_counts": mark_counts,
        "counts": counts,
        "rows": rows,
        "detections": detection_items if detection_items is not None else _detections_to_dict(kept),
        "notes": notes,
    }
    if rejected_detection_items is not None:
        payload["rejected_detections"] = rejected_detection_items
        payload["rejected_detection_count"] = len(rejected_detection_items)
    if pre_eval_detections is not None:
        payload["validated_counts"] = counts
        payload["pre_eval_cv_counts"] = cv_counts
    if extra:
        payload.update(extra)
    return payload


def count_symbols_in_image_cv(
    *,
    image: Image.Image,
    legend: SymbolTableInfo,
    glyph_dir: Path | None = None,
    symbol_file_suffix: str = ".json",
    nms_iou: float | None = None,
) -> dict[str, Any]:
    if not legend.entries:
        return {
            "status": "failed",
            "reason": "Empty legend catalog",
            "counts": {},
            "rows": [],
            "detections": [],
            "notes": [],
        }

    nms_iou = float(nms_iou if nms_iou is not None else config.SYMBOL_COUNT_NMS_IOU)
    kept, notes, meta = detect_symbols_raw(
        image=image,
        legend=legend,
        glyph_dir=glyph_dir,
        symbol_file_suffix=symbol_file_suffix,
        nms_iou=nms_iou,
        apply_local_nms=True,
    )
    pre_eval = list(kept)
    detection_items: list[dict[str, Any]] | None = None
    rejected_items: list[dict[str, Any]] | None = None
    eval_summary: dict[str, Any] | None = None

    if config.MERGE_EVAL_ENABLED and glyph_dir:
        from pipeline.merge_evaluator import evaluate_merged_detections

        evaluated, eval_summary = evaluate_merged_detections(
            roi_image=image,
            legend=legend,
            glyph_dir=glyph_dir,
            detections=kept,
        )
        kept = [ev.detection for ev in evaluated if ev.verdict == "accepted"]
        rejected = [ev for ev in evaluated if ev.verdict == "rejected"]
        detection_items = [ev.to_dict() for ev in evaluated if ev.verdict == "accepted"]
        rejected_items = [ev.to_dict() for ev in rejected]
        if rejected:
            notes.append(
                f"Post-detection validation rejected {len(rejected)} false hit(s) "
                f"(glyph/vision judge on single image)."
            )

    extra: dict[str, Any] = {}
    if eval_summary is not None:
        extra["evaluation"] = eval_summary

    return _finalize_counts(
        legend=legend,
        kept=kept,
        notes=notes,
        meta=meta,
        image=image,
        method="single_image_cv",
        raw_detection_count=len(pre_eval),
        nms_iou=nms_iou,
        pre_eval_detections=pre_eval,
        detection_items=detection_items,
        rejected_detection_items=rejected_items,
        extra=extra,
    )


def run_symbol_count_job(
    symbol_file: Path,
    image_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    legend, glyph_dir = parse_symbol_file(symbol_file, output_dir)

    image = Image.open(image_file).convert("RGB")
    payload = count_symbols_in_image_cv(
        image=image,
        legend=legend,
        glyph_dir=glyph_dir,
        symbol_file_suffix=symbol_file.suffix.lower(),
    )

    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_detections_overlay(
        image=image,
        detections=payload.get("detections") or [],
        output_path=output_dir / "overlay.png",
    )

    try:
        image.save(output_dir / "plan_image.png", format="PNG")
    except Exception:  # noqa: BLE001
        pass

    return payload


__all__ = [
    "count_symbols_in_image_cv",
    "detect_symbols_raw",
    "filter_to_legend",
    "legend_count_keys",
    "run_symbol_count_job",
]
