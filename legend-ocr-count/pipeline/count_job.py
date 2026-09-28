from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from pipeline import config
from pipeline.callout_ocr import CalloutHit, detect_callouts
from pipeline.legend_match import attach_legend_match
from pipeline.legend_ocr import LegendEntry, load_image, parse_legend_file, _strip_leading_index
from pipeline.nms import nms_yolo_hits
from pipeline.overlay import draw_detection_overlay
from pipeline.tile_overlap import load_tile_rois, merge_overlapping_tile_hits
from pipeline.yolo_class_map import resolve_class_map
from pipeline.yolo_detect import YoloHit, detect_symbols, yolo_available

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str], None] | None


def _emit(cb: ProgressCb, stage: str) -> None:
    if cb:
        cb(stage)


def _collect_plan_images(plan_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        p
        for p in sorted(plan_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    ]

    def rank(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        if "full" in name or ("plan" in name and "zoom" not in name):
            return (0, name)
        if "zoom" in name:
            return (2, name)
        return (1, name)

    return sorted(images, key=rank)


def _assign_yolo_to_legend(
    hits: list[YoloHit],
    legend: list[LegendEntry],
) -> tuple[list[YoloHit], Counter[str], dict[str, str]]:
    """Assign legend numbers to YOLO hits.

    Returns (matched_hits, unmatched_class_counts, unmatched_reason_by_class).
    Unmatched detections are listed for the UI but not counted toward legend rows.
    """
    assigned: list[YoloHit] = []
    unmatched: Counter[str] = Counter()
    reasons: dict[str, str] = {}
    for hit in hits:
        mapped = resolve_class_map(hit.class_name)
        m = attach_legend_match(
            hit.class_name,
            legend,
            min_score=config.YOLO_MATCH_MIN_SCORE,
        )
        if m is None:
            unmatched[hit.class_name] += 1
            if mapped is None:
                reasons[hit.class_name] = "detected - not in annotation-to-legend map"
            else:
                reasons[hit.class_name] = (
                    f"detected - no legend row for '{mapped.legend_phrase}'"
                )
            continue
        assigned.append(
            YoloHit(
                class_id=hit.class_id,
                class_name=hit.class_name,
                confidence=hit.confidence,
                x1=hit.x1,
                y1=hit.y1,
                x2=hit.x2,
                y2=hit.y2,
                image_name=hit.image_name,
                legend_number=m.legend_number,
                count_weight=max(1, int(m.count_weight)),
                rescued=hit.rescued,
            )
        )
    return assigned, unmatched, reasons


def _unmatched_yolo_rows(
    counts: Counter[str],
    reasons: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "class_name": cls_name,
            "count": int(cnt),
            "reason": reasons.get(cls_name, "detected - not matched to legend"),
        }
        for cls_name, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

def _build_rows(
    legend: list[LegendEntry],
    callouts: list[CalloutHit],
    yolo_hits: list[YoloHit],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    callout_counts = Counter(h.number for h in callouts)
    # Weighted YOLO counts (CONDUIT STUB 2 contributes 2 toward CONDUIT STUB)
    yolo_by_legend: Counter[int] = Counter()
    for h in yolo_hits:
        if h.legend_number is None:
            continue
        yolo_by_legend[h.legend_number] += max(1, int(h.count_weight or 1))

    rows: list[dict[str, Any]] = []
    for entry in legend:
        yolo_n = int(yolo_by_legend.get(entry.number, 0))
        call_n = int(callout_counts.get(entry.number, 0))
        if not config.CALLOUT_OCR_ENABLED:
            count = yolo_n
            method = "yolo" if yolo_n > 0 else "none"
        elif config.YOLO_PRIMARY_COUNTS and yolo_n > 0:
            count = yolo_n
            method = "yolo" if call_n == 0 else "yolo+ocr_callout"
        elif call_n > 0:
            count = call_n
            method = "ocr_callout"
        else:
            count = 0
            method = "none"

        matched_classes = sorted(
            {
                h.class_name
                for h in yolo_hits
                if h.legend_number == entry.number
            }
        )
        # YOLO-only: skip legend rows with no YOLO hits (symbols not in the trained set)
        if not config.CALLOUT_OCR_ENABLED and yolo_n <= 0:
            continue
        rows.append(
            {
                "number": entry.number,
                "symbol": str(entry.number),
                "description": _strip_leading_index(entry.description),
                "detail_ref": entry.detail_ref,
                "count": count,
                "yolo_count": yolo_n,
                "callout_count": call_n,
                "yolo_classes": matched_classes,
                "method": method,
            }
        )

    legend_nums = {e.number for e in legend}
    unknown_callouts: list[dict[str, Any]] = []
    for num, cnt in sorted(callout_counts.items()):
        if num not in legend_nums:
            unknown_callouts.append(
                {
                    "number": num,
                    "symbol": str(num),
                    "description": "(not in uploaded legend)",
                    "detail_ref": None,
                    "count": int(cnt),
                    "method": "ocr_callout",
                }
            )

    # Unmatched YOLO is tracked separately in run_count_job
    return rows, unknown_callouts


def run_count_job(
    legend_path: Path,
    plan_path: Path | None,
    out_dir: Path,
    *,
    folder_dir: Path | None = None,
    progress: ProgressCb = None,
) -> dict[str, Any]:
    """Parse legend, OCR callouts + YOLO symbols, write result.json + overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress, "parsing_legend")
    legend = parse_legend_file(legend_path)
    logger.info("Parsed %d legend entries from %s", len(legend), legend_path.name)

    images: list[tuple[str, Image.Image]] = []
    if folder_dir is not None and folder_dir.is_dir():
        paths = _collect_plan_images(folder_dir)
        if not paths:
            raise RuntimeError("Folder upload has no plan images.")
        _emit(progress, f"loading_{len(paths)}_images")
        for p in paths:
            images.append((p.name, load_image(p)))
    else:
        if plan_path is None or not plan_path.is_file():
            raise RuntimeError("Plan image missing.")
        _emit(progress, "loading_plan")
        images.append((plan_path.name, load_image(plan_path)))

    use_yolo = bool(config.YOLO_ENABLED and yolo_available())
    use_callouts = bool(config.CALLOUT_OCR_ENABLED)
    all_callouts: list[CalloutHit] = []
    all_yolo: list[YoloHit] = []
    unmatched_counts: Counter[str] = Counter()
    unmatched_reasons: dict[str, str] = {}
    per_image: list[dict[str, Any]] = []

    for name, img in images:
        callouts: list[CalloutHit] = []
        if use_callouts:
            _emit(progress, f"ocr_callouts:{name}")
            callouts = detect_callouts(img, image_name=name)
            all_callouts.extend(callouts)
        else:
            _emit(progress, f"skip_ocr_callouts:{name}")

        yolo_hits: list[YoloHit] = []
        if use_yolo:
            _emit(progress, f"yolo_detect:{name}")
            yolo_hits = detect_symbols(img, image_name=name)
            before = len(yolo_hits)
            yolo_hits = nms_yolo_hits(
                yolo_hits,
                same_class_iou=config.YOLO_NMS_IOU,
                cross_class_iou=config.YOLO_CROSS_NMS_IOU,
                compact_center_px=config.YOLO_NMS_CENTER_PX,
            )
            if len(yolo_hits) != before:
                logger.info(
                    "YOLO NMS on %s: %d → %d (dropped overlap duplicates)",
                    name,
                    before,
                    len(yolo_hits),
                )
            yolo_hits, um_counts, um_reasons = _assign_yolo_to_legend(yolo_hits, legend)
            unmatched_counts.update(um_counts)
            unmatched_reasons.update(um_reasons)
            all_yolo.extend(yolo_hits)

        per_image.append(
            {
                "image": name,
                "width": img.size[0],
                "height": img.size[1],
                "callout_count": len(callouts),
                "yolo_count": len(yolo_hits),
                "numbers": sorted({h.number for h in callouts}),
                "yolo_classes": sorted({h.class_name for h in yolo_hits}),
            }
        )

    overlay_yolo_local = list(all_yolo)
    overlap_merged = False
    if use_yolo and folder_dir is not None and len(images) > 1:
        _emit(progress, "merging_tile_overlap")
        rois = load_tile_rois(folder_dir, default_overlap_pct=config.TILE_OVERLAP_PCT)
        if rois:
            sizes = {
                Path(name).name.lower(): img.size for name, img in images
            }
            before = len(all_yolo)
            all_yolo = merge_overlapping_tile_hits(all_yolo, rois, image_sizes=sizes)
            all_yolo = nms_yolo_hits(
                all_yolo,
                same_class_iou=config.YOLO_NMS_IOU,
                cross_class_iou=config.YOLO_CROSS_NMS_IOU,
                compact_center_px=config.YOLO_NMS_CENTER_PX,
            )
            overlap_merged = True
            logger.info(
                "Folder overlap merge: %d → %d YOLO hits",
                before,
                len(all_yolo),
            )
        else:
            logger.info(
                "Folder has no zooms_manifest / plan_zoom grid — "
                "counts are per-image NMS only (overlapping tiles may still double-count)."
            )

    rows, unknown_callouts = _build_rows(legend, all_callouts, all_yolo)
    unmatched_yolo = _unmatched_yolo_rows(unmatched_counts, unmatched_reasons)
    total_count = sum(r["count"] for r in rows)
    rescued_kept = sum(1 for h in all_yolo if h.rescued)

    _emit(progress, "writing_overlay")
    primary_name, primary_img = images[0]
    if len(images) == 1:
        overlay_callouts = all_callouts
        overlay_yolo = overlay_yolo_local
    else:
        overlay_callouts = [
            h for h in all_callouts if (h.image_name or primary_name) == primary_name
        ]
        overlay_yolo = [
            h for h in overlay_yolo_local if (h.image_name or primary_name) == primary_name
        ]

    plan_png = out_dir / "plan_image.png"
    primary_img.save(plan_png)
    overlay_path = out_dir / "overlay.png"
    draw_detection_overlay(
        primary_img,
        callouts=overlay_callouts,
        yolo_hits=overlay_yolo,
        out_path=overlay_path,
    )

    if len(images) > 1:
        _write_folder_summary(images, all_callouts, all_yolo, out_dir / "folder_summary.png")

    if use_yolo and use_callouts:
        method = "yolo+ocr_callout"
        note = (
            "Primary counts use YOLO symbol detections matched to legend descriptions when "
            "possible; otherwise circled callout OCR counts are used. Overlapping YOLO boxes "
            "and overlapping zoom-tile regions (10%) are merged so each mark counts once."
        )
    elif use_yolo:
        method = "yolo"
        note = (
            "Callout OCR is paused (CALLOUT_OCR_ENABLED=false). Counts are YOLO symbol "
            "detections matched to legend descriptions only. Set CALLOUT_OCR_ENABLED=true "
            "in .env and restart to re-enable circled callout OCR."
        )
    elif use_callouts:
        method = "ocr_callout"
        note = "YOLO model not loaded — counts are OCR circled callouts only."
    else:
        method = "none"
        note = "Both YOLO and callout OCR are disabled."

    if use_yolo and config.YOLO_RESCUE_ENABLED:
        note += (
            f" A second verification pass (conf>={config.YOLO_RESCUE_CONF:.2f}"
            f"{', TTA' if config.YOLO_RESCUE_TTA else ''}) rescued "
            f"{rescued_kept} symbols the primary threshold missed; overlapping "
            "rescues are merged so each symbol counts once."
        )

    result: dict[str, Any] = {
        "method": method,
        "yolo_enabled": use_yolo,
        "callout_ocr_enabled": use_callouts,
        "yolo_model": str(config.YOLO_MODEL_PATH.name) if use_yolo else None,
        "yolo_conf": config.YOLO_CONF if use_yolo else None,
        "yolo_rescue": (
            {
                "enabled": bool(config.YOLO_RESCUE_ENABLED),
                "conf": config.YOLO_RESCUE_CONF,
                "tta": bool(config.YOLO_RESCUE_TTA),
                "rescued_kept": rescued_kept,
            }
            if use_yolo
            else None
        ),
        "note": note,
        "legend_file": legend_path.name,
        "legend_entries": len(legend),
        "mode": "folder" if len(images) > 1 else "single",
        "images_processed": len(images),
        "total_count": total_count,
        "total_callouts": sum(r["callout_count"] for r in rows)
        + sum(u["count"] for u in unknown_callouts),
        "total_yolo": len(all_yolo),
        "overlap_merged": overlap_merged,
        "rows": rows,
        "unknown_numbers": unknown_callouts,
        "yolo_unmatched": unmatched_yolo,
        "detections": [h.to_dict() for h in all_callouts],
        "yolo_detections": [h.to_dict() for h in all_yolo],
        "per_image": per_image,
        "overlay_image": primary_name,
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _emit(progress, "done")
    return result


def _write_folder_summary(
    images: list[tuple[str, Image.Image]],
    callouts: list[CalloutHit],
    yolo_hits: list[YoloHit],
    out_path: Path,
) -> None:
    lines = ["Folder summary (YOLO + OCR callouts)", ""]
    call_by: dict[str, list[CalloutHit]] = {}
    yolo_by: dict[str, list[YoloHit]] = {}
    for h in callouts:
        call_by.setdefault(h.image_name or "", []).append(h)
    for h in yolo_hits:
        yolo_by.setdefault(h.image_name or "", []).append(h)
    for name, _img in images:
        cnums = Counter(h.number for h in call_by.get(name, []))
        ycls = Counter(h.class_name for h in yolo_by.get(name, []))
        c_parts = ", ".join(f"○{n}×{c}" for n, c in sorted(cnums.items())) or "—"
        y_parts = ", ".join(f"{k}×{c}" for k, c in sorted(ycls.items())) or "—"
        lines.append(f"{name}")
        lines.append(f"  callouts: {c_parts}")
        lines.append(f"  yolo: {y_parts}")

    canvas = Image.new("RGB", (1000, 40 + 20 * len(lines)), (250, 250, 252))
    draw = ImageDraw.Draw(canvas)
    y = 16
    for line in lines:
        draw.text((20, y), line, fill=(20, 24, 32))
        y += 20
    canvas.save(out_path)
