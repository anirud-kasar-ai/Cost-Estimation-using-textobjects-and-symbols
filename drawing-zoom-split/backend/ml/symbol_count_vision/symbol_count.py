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
    suppress_nearby_same_label,
)
from pipeline.cv.ocr_tags import (
    ap_skip_centers_from_ocr,
    detect_tags_from_ocr,
    tesseract_available,
)
from pipeline.cv.tag_match import count_key, tagged_legend_keys, text_legend_keys
from pipeline.cv.template_match import detect_symbols_from_glyphs
from pipeline.cv.triangle_drops import detect_drop_marks
from pipeline.overlay import draw_detections_overlay
from pipeline.symbol_legend import (
    SymbolTableInfo,
    parse_symbol_file,
)


_SOURCE_LABELS = {
    "template": "template match",
    "ocr": "text/OCR",
    "triangle": "drop marks",
    "hourglass": "camera marks",
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


def _detections_to_dict(detections: list[SymbolDetection]) -> list[dict[str, Any]]:
    return [
        {
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
        for det in detections
    ]


def detect_symbols_raw(
    *,
    image: Image.Image,
    legend: SymbolTableInfo,
    glyph_dir: Path | None = None,
    symbol_file_suffix: str = ".json",
    nms_iou: float | None = None,
    apply_local_nms: bool = True,
) -> tuple[list[SymbolDetection], list[str], dict[str, Any]]:
    """Run all detectors; return legend-gated detections (+ notes/meta).

    When ``apply_local_nms`` is True (single-image path), class NMS and
    cross-label suppression run here. Folder merge sets it False and applies
    NMS once in ROI space after translating all tiles.
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
    valid_keys = legend_count_keys(legend)
    raw_detections: list[SymbolDetection] = []
    linear_keys: set[str] = set()

    if not tesseract_available():
        notes.append(
            "Tesseract OCR is not installed — text-tag symbol detection was skipped. "
            "Install Tesseract and ensure it is on PATH."
        )
    else:
        raw_detections.extend(detect_tags_from_ocr(image, legend))

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
    meta["hash_key"] = hash_key
    meta["camera_key"] = camera_key
    camera_boxes: list[dict[str, Any]] = []
    if hash_key or camera_key:
        skip = ap_skip_centers_from_ocr([d for d in raw_detections if d.source == "ocr"])
        drop_boxes, camera_boxes = detect_drop_marks(
            image,
            skip_centers=skip,
            exclude_top_pct=config.CV_TITLE_BAND_PCT,
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
        if tesseract_available() and drop_boxes and text_legend_keys(legend):
            from pipeline.cv.ocr_tags import _ocr_part_labels_near_centers
            from pipeline.cv.tag_match import text_legend_keys as _tlk

            raw_detections.extend(
                _ocr_part_labels_near_centers(
                    image,
                    _tlk(legend),
                    [(b["cx"], b["cy"]) for b in drop_boxes],
                    image.size[1] * float(config.CV_TITLE_BAND_PCT),
                )
            )

    # Camera templates alone are unreliable (match keynotes / junctions).
    raw_detections = _filter_camera_templates_near_geometry(
        raw_detections, camera_boxes
    )
    raw_detections = filter_to_legend(raw_detections, valid_keys)

    if apply_local_nms:
        kept = nms_by_class(raw_detections, iou_threshold=nms_iou)
        kept = suppress_cross_label(kept, iou_threshold=0.6, source="template")
        kept = suppress_nearby_same_label(kept, source="ocr", max_center_dist=80.0)
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
) -> dict[str, Any]:
    valid_keys = legend_count_keys(legend)
    kept = filter_to_legend(kept, valid_keys)
    nms_iou = float(nms_iou if nms_iou is not None else config.SYMBOL_COUNT_NMS_IOU)
    hash_key = meta.get("hash_key")
    linear_keys: set[str] = set(meta.get("linear_keys") or set())
    glyphs_available = bool(meta.get("glyphs_available"))

    cv_counts: dict[str, int] = {}
    mark_counts: dict[str, int] = {}
    for det in kept:
        mark_counts[det.symbol] = mark_counts.get(det.symbol, 0) + 1
        cv_counts[det.symbol] = cv_counts.get(det.symbol, 0) + max(1, int(det.qty))

    qty_summed = False
    qty_breakdown: str | None = None
    if hash_key:
        marks = [d for d in kept if d.source == "triangle"]
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

    counts = dict(cv_counts)
    llm_summary: dict[str, Any] | None = None
    if image is not None and method == "single_image_cv" and config.llm_enabled():
        from pipeline.llm_verify import verify_counts_with_llm

        llm = verify_counts_with_llm(image, legend, cv_counts)
        llm_summary = {
            "used": llm["ok"],
            "provider": llm.get("provider"),
            "model": llm["model"],
            "notes": llm.get("notes"),
            "error": llm.get("error"),
        }
        if llm["ok"] and llm["counts"]:
            disagreements: list[str] = []
            for key, llm_value in llm["counts"].items():
                if key not in valid_keys:
                    continue
                cv_value = cv_counts.get(key, 0)
                if qty_summed and key == hash_key:
                    mark_n = mark_counts.get(key, 0)
                    if llm_value and llm_value != cv_value:
                        notes.append(
                            f"LLM counted {llm_value} visible '#' mark(s); "
                            f"kept QTY-expanded jack count {cv_value} "
                            f"from {mark_n} mark(s)."
                        )
                    continue
                if llm_value > 0:
                    counts[key] = llm_value
                elif cv_value > 0:
                    disagreements.append(key)
            counts = {k: v for k, v in counts.items() if v > 0 and k in valid_keys}
            method = "cv_plus_llm_verify"
            if disagreements:
                notes.append(
                    "LLM reported 0 for CV-detected symbols (kept CV counts): "
                    + ", ".join(sorted(disagreements))
                )
            if llm.get("notes"):
                notes.append(f"LLM: {llm['notes']}")
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
        cv_counts if llm_summary else None,
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
        "detections": _detections_to_dict(kept),
        "notes": notes,
    }
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
    return _finalize_counts(
        legend=legend,
        kept=kept,
        notes=notes,
        meta=meta,
        image=image,
        method="single_image_cv",
        raw_detection_count=len(kept),
        nms_iou=nms_iou,
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
