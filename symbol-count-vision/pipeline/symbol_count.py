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
    suppress_hash_on_devices,
    suppress_in_boxes,
    suppress_nearby_same_label,
    suppress_overlapping_devices,
    suppress_stub_on_devices,
)
from pipeline.cv.ocr_tags import tesseract_available
from pipeline.cv.tag_match import count_key, tagged_legend_keys
from pipeline.cv.template_match import LINEAR_GLYPH_RE
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
  J-HOOK vs J, AP vs #, hourglass vs drop: follow TAXONOMY_CONTEXT_RULES only.
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

# Drawing literacy that is NOT look-alike geometry (those live in the taxonomy).
CAD_CONSTRUCTION_SYMBOL_KNOWLEDGE = """
CONSTRUCTION CAD / LOW-VOLTAGE PLAN LITERACY (mandatory):

You are reading a contractor bid-set technology floor plan (AutoCAD/Revit plot),
NOT a schematic diagram and NOT a photograph. Thin 1–2 px vector ink is real.

Look-alike / context rules are NOT listed here. Use TAXONOMY_CONTEXT_RULES only.

NEVER COUNT AS LEGEND EQUIPMENT:
- Elevations AFF: "+48", "+49" (mounting height only).
- Room / space IDs: A-1, A-2, 1.03 in a circle, RR/B, RR/G.
- KEY PLAN inset, sheet title "A-WING (EAST)", grid bubbles, dimension strings.
- Boxed keynote digits alone (square with "1") unless they are a legend tag.
- Break-line zigzags, hatch, door swings, furniture outlines.
- Pipe callouts (1|5, 1|13) as their own symbol — they only locate nearby '#' drops.

DENSE WALL RUNS: raceway box + digits + elevation + triangle + QTY + keynote can
sit within inches. Parse roles separately; count only legend keys.
""".strip()

_SOURCE_LABELS = {
    "template": "template match",
    "ocr": "text/OCR",
    "vision_ocr": "vision OCR",
    "vision_classify": "vision classify",
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
            "classify_source": det.classify_source,
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


def _candidates_to_detections(
    candidates: list[Any],
    taxonomy: Any,
    valid_keys: set[str],
) -> list[SymbolDetection]:
    from pipeline.count.finalize import apply_count_semantics

    out: list[SymbolDetection] = []
    for cand in candidates:
        key = cand.resolved_key
        classify_source = cand.classify_source
        if not key or cand.status != "resolved":
            if cand.candidate_keys:
                key = cand.candidate_keys[0]
                classify_source = classify_source or "cv_fallback"
            elif (cand.extras or {}).get("legend_hint"):
                key = str(cand.extras["legend_hint"])
                classify_source = classify_source or "cv_fallback"
            else:
                continue
        if key not in valid_keys:
            continue
        out.append(
            SymbolDetection(
                symbol=key,
                box=cand.bbox,
                score=float(cand.score),
                source=str(cand.source),
                qty=max(1, int(cand.qty or 1)),
                classify_source=classify_source,
            )
        )
    return apply_count_semantics(out, taxonomy)


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
    """CV propose → context resolve → optional vision classify (label only).

    Localization is always a CV contour/OCR box. LLMs never supply boxes.
    Folder merge sets ``apply_local_nms=False`` and NMS once in ROI space.
    """
    from pipeline.classify.context_resolver import resolve_candidates
    from pipeline.classify.glyph_identify import identify_candidates_by_glyphs
    from pipeline.classify.vision_classify import classify_ambiguous
    from pipeline.cv.ocr_tags import clear_ocr_page_cache
    from pipeline.detect.candidates import propose_candidates
    from pipeline.taxonomy.builder import taxonomy_from_legend

    notes: list[str] = list(legend.notes or [])
    meta: dict[str, Any] = {
        "linear_keys": set(),
        "hash_key": None,
        "camera_key": None,
        "glyphs_available": False,
        "taxonomy": None,
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
        else config.llm_enabled()
    )
    valid_keys = legend_count_keys(legend)
    glyphs_available = bool(
        glyph_dir and glyph_dir.is_dir() and any(glyph_dir.glob("*.png"))
    )
    meta["glyphs_available"] = glyphs_available

    taxonomy = taxonomy_from_legend(
        legend, glyph_dir, source=symbol_file_suffix
    )
    meta["taxonomy"] = taxonomy
    linear_keys = {
        e.key for e in taxonomy.entries if e.count_semantics == "linear_suppressed"
    }
    meta["linear_keys"] = linear_keys
    hash_key = next(
        (e.key for e in taxonomy.entries if e.count_semantics == "qty_expand"),
        None,
    )
    if hash_key is None:
        hash_key = next((e.key for e in taxonomy.entries if e.key == "#"), None)
    camera_key = next(
        (e.key for e in taxonomy.entries if e.shape_class == "hourglass"), None
    )
    pole_key = next(
        (e.key for e in taxonomy.entries if e.shape_class == "bowtie"), None
    )
    meta["hash_key"] = hash_key
    meta["camera_key"] = camera_key
    meta["pole_key"] = pole_key

    if not tesseract_available():
        notes.append(
            "Tesseract OCR is not installed — text-tag symbol detection was skipped. "
            "Install Tesseract and ensure it is on PATH."
        )

    candidates = propose_candidates(
        image,
        legend,
        taxonomy,
        tile_id="tile",
        exclude_top_pct=title_band,
    )

    if hash_key:
        from pipeline.cv.ocr_tags import collect_pipe_callout_ocr_hints, collect_qty_digit_hints
        from pipeline.cv.triangle_drops import find_triangle_tip_near
        from pipeline.detect.candidates import Candidate

        # Reuse the tile's single OCR pass. Per-box Tesseract on every contour
        # was the main remaining slowdown.
        callout_hints = collect_pipe_callout_ocr_hints(image, exclude_top_pct=title_band)
        for hint in callout_hints:
            tip = find_triangle_tip_near(image, hint["cx"], hint["cy"])
            if not tip:
                continue
            candidates.append(
                Candidate(
                    tile_id="tile",
                    bbox=BoundingBox(
                        x1=float(tip["x"]),
                        y1=float(tip["y"]),
                        x2=float(tip["x"] + tip["w"]),
                        y2=float(tip["y"] + tip["h"]),
                    ),
                    shape_class_guess="triangle_like",
                    score=0.8,
                    source="callout_drop",
                )
            )
        qty_hints = collect_qty_digit_hints(image, exclude_top_pct=title_band)
        for cand in candidates:
            if cand.source not in {"triangle", "callout_drop"}:
                continue
            for hint in qty_hints:
                dist = ((cand.cx - hint["cx"]) ** 2 + (cand.cy - hint["cy"]) ** 2) ** 0.5
                if dist <= 36.0:
                    cand.qty = max(int(cand.qty or 1), int(hint["qty"]))

    resolved = resolve_candidates(candidates, taxonomy)
    if run_vision and config.llm_enabled():
        n_amb = sum(1 for c in resolved if c.status == "ambiguous")
        resolved = classify_ambiguous(image, resolved, taxonomy, legend=legend)
        n_vis = sum(1 for c in resolved if c.classify_source == "vision_classify")
        if n_amb:
            notes.append(
                f"Vision classify: {n_amb} ambiguous candidate(s), "
                f"{n_vis} labeled by vision."
            )

    resolved, glyph_stats = identify_candidates_by_glyphs(
        image,
        resolved,
        legend=legend,
        glyph_dir=glyph_dir,
        taxonomy=taxonomy,
        valid_keys=valid_keys,
    )
    if glyph_stats.get("compared") or glyph_stats.get("no_glyph"):
        notes.append(
            "Glyph identify: after detect, score each mark only against its "
            "own glyph file if that file exists in the glyph folder "
            f"(compared {glyph_stats.get('compared', 0)}, "
            f"no glyph file {glyph_stats.get('no_glyph', 0)}, "
            f"confirmed {glyph_stats.get('confirmed', 0)}, "
            f"named {glyph_stats.get('assigned', 0)})."
        )

    # QTY stays 1 unless OCR already saw a digit beside the drop. Per-triangle
    # Tesseract (3 PSM modes × strips) was hundreds of extra OCR calls per tile.

    raw_detections = _candidates_to_detections(resolved, taxonomy, valid_keys)
    raw_detections = filter_to_legend(raw_detections, valid_keys)

    if linear_keys:
        notes.append(
            "Linear/run items are not counted as discrete glyphs; part labels "
            "like WM2300/2300 are still counted via OCR: "
            + ", ".join(sorted(linear_keys))
        )
    if not tagged_legend_keys(legend) and not glyphs_available:
        notes.append(
            "Legend has no text tags and no glyph templates — nothing to match "
            "on the plan. Use symbol_table.json or a technical symbol PDF that "
            "includes embedded glyphs."
        )
    elif symbol_file_suffix.lower() == ".json" and not glyphs_available:
        notes.append(
            "Graphical glyph crops unavailable: upload a PDF legend to extract artwork."
        )

    if apply_local_nms:
        kept = nms_by_class(raw_detections, iou_threshold=nms_iou, center_dist=20.0)
        kept = suppress_cross_label(kept, iou_threshold=0.6, source="template")
        kept = suppress_nearby_same_label(kept, source="ocr", max_center_dist=48.0)
        kept = suppress_cross_source_duplicates(
            kept,
            sources={"ocr", "triangle", "hourglass", "bowtie", "stub", "jhook"},
            max_center_dist=24.0,
            iou_threshold=0.15,
        )
        kept = suppress_overlapping_devices(kept)
        kept = suppress_hash_on_devices(kept)
        kept = suppress_stub_on_devices(kept)
        from pipeline.cv.ocr_tags import find_keyplan_regions

        kept = suppress_in_boxes(kept, find_keyplan_regions(image))
        kept = filter_to_legend(kept, valid_keys)
        clear_ocr_page_cache(image)
        return kept, notes, meta

    clear_ocr_page_cache(image)
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
    taxonomy = meta.get("taxonomy")
    if taxonomy is not None:
        from pipeline.count.finalize import apply_count_semantics

        kept = apply_count_semantics(kept, taxonomy)
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
    # Count-verify is diagnostic only: it must not invent counts without CV boxes.
    if image is not None and config.llm_enabled() and folder_vision:
        from pipeline.llm_verify import verify_counts_with_llm
        from pipeline.taxonomy.schema import taxonomy_rules_prompt_block

        extra_rules = [
            CAD_CONSTRUCTION_SYMBOL_KNOWLEDGE,
            SYMBOL_COUNT_OVERLAP_PROMPT,
            taxonomy_rules_prompt_block(meta.get("taxonomy")),
        ]
        llm = verify_counts_with_llm(
            image,
            legend,
            counts,
            linear_keys=linear_keys,
            extra_rules=extra_rules,
            extra_429_retries=int(config.FOLDER_VISION_429_RETRIES),
            taxonomy=taxonomy,
        )
        llm_summary = {
            "used": llm["ok"],
            "provider": llm.get("provider"),
            "model": llm["model"],
            "notes": llm.get("notes"),
            "error": llm.get("error"),
            "mode": "folder_post_merge_notes_only",
        }
        if llm.get("notes"):
            notes.append(f"LLM (notes only, CV counts kept): {llm['notes']}")
        elif llm.get("error"):
            notes.append(f"LLM verification skipped: {llm['error']}")

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
        remapped_n = int((eval_summary or {}).get("remapped") or 0)
        if remapped_n:
            notes.append(
                f"Post-detection validation remapped {remapped_n} mark(s) "
                f"to a better-matching legend glyph."
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
    try:
        from pipeline.taxonomy.builder import taxonomy_from_legend

        taxonomy_from_legend(legend, glyph_dir, source=str(symbol_file)).save(
            output_dir / "symbol_taxonomy.json"
        )
    except Exception:  # noqa: BLE001
        pass

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
