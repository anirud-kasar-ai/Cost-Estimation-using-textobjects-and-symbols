"""YOLO symbol count on zoom tiles + invoice (drawing-zoom-split stage)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from pipeline import config
from pipeline.legend_match import LegendEntry, attach_legend_match
from pipeline.nms import nms_yolo_hits
from pipeline.pricing import build_invoice, ensure_pricing_csv
from PIL import Image as PILImage

from pipeline.text_mask import (
    dense_text_regions,
    filter_hits_in_regions,
    filter_hits_on_text,
    load_wing_text_words,
)
from pipeline.verify import verifier_available, verify_hits
from pipeline.tile_overlap import load_tile_rois, merge_overlapping_tile_hits
from pipeline.yolo_detect import YoloHit, detect_symbols, yolo_available

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str], None]

COUNTS_DIR = "06_counts"
_DTL_TAIL_RE = re.compile(r",?\s*[DO]TL\.?\s+[A-Z0-9/'`\-?]+\s*$", re.IGNORECASE)
_LEADING_INDEX_RE = re.compile(r"^\s*\d{1,3}\s*[.)\-:,]\s*")


def _strip_leading_index(text: str) -> str:
    return _LEADING_INDEX_RE.sub("", text or "").strip()


def _split_detail(description: str) -> tuple[str, str | None]:
    raw = _strip_leading_index(description or "")
    m = _DTL_TAIL_RE.search(raw)
    if not m:
        return raw.strip(), None
    detail = m.group(0).lstrip(", ").strip()
    desc = raw[: m.start()].strip(" ,;")
    return desc, detail


def legend_from_symbol_table(path: Path) -> list[LegendEntry]:
    """Build matchable legend rows from extracted symbol_table.json."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    out: list[LegendEntry] = []
    for i, row in enumerate(entries, start=1):
        if not isinstance(row, dict):
            continue
        full = str(row.get("description") or "").strip()
        if not full:
            continue
        desc, detail = _split_detail(full)
        if not desc:
            continue
        # Prefer part_number / mfg as detail when no DTL
        part = str(row.get("part_number") or row.get("mfg_model") or "").strip()
        if not detail and part:
            detail = part
        out.append(
            LegendEntry(
                number=i,
                description=desc.upper(),
                detail_ref=(detail or "").upper() or None,
                raw_line=full.upper(),
            )
        )
    return out


def _assign_hits(hits: list[YoloHit], legend: list[LegendEntry]) -> list[YoloHit]:
    assigned: list[YoloHit] = []
    for hit in hits:
        match = attach_legend_match(
            hit.class_name,
            legend,
            min_score=config.YOLO_MATCH_MIN_SCORE,
        )
        weight = match.count_weight if match else 1
        legend_number = match.legend_number if match else None
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
                legend_number=legend_number,
                count_weight=max(1, int(weight or 1)),
                rescued=hit.rescued,
            )
        )
    return assigned


def drop_unmatched_rescues(hits: list[YoloHit]) -> tuple[list[YoloHit], int]:
    """Legend gate for rescue-band hits.

    A confident detection may stand alone, but a low-confidence rescued hit for
    a class that isn't even in this drawing's legend is almost certainly noise.
    Returns (kept_hits, dropped_count).
    """
    kept = [h for h in hits if not (h.rescued and h.legend_number is None)]
    return kept, len(hits) - len(kept)


def merge_fullwing_large_hits(
    roi_hits: list[YoloHit],
    wing_hits: list[YoloHit],
    *,
    min_px: float | None = None,
) -> list[YoloHit]:
    """Union tile detections with LARGE boxes from a full-wing pass.

    Tiles overlap ~147px: a symbol bigger than that can be cut at a boundary
    in every tile that sees it, so the tile pass never sees the whole shape.
    The full-wing pass does. Only confident, non-rescued wing boxes at/above
    the size floor are added, and tile boxes that are partial views of an
    accepted large box (same class, mostly contained in it) are absorbed so
    nothing counts twice.
    """
    floor = float(config.YOLO_FULLWING_MIN_PX if min_px is None else min_px)
    large = [
        h
        for h in wing_hits
        if not h.rescued
        and h.confidence >= float(config.YOLO_CONF)
        and max(h.x2 - h.x1, h.y2 - h.y1) >= floor
    ]
    if not large:
        return roi_hits

    def ios(inner: YoloHit, outer: YoloHit) -> float:
        """Intersection over the *inner* box (containment, not IoU)."""
        ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
        ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, (inner.x2 - inner.x1) * (inner.y2 - inner.y1))
        return inter / area

    kept_tiles = [
        t
        for t in roi_hits
        if not any(lg.class_id == t.class_id and ios(t, lg) >= 0.60 for lg in large)
    ]
    return nms_yolo_hits(
        kept_tiles + large,
        same_class_iou=config.YOLO_NMS_IOU,
        cross_class_iou=config.YOLO_CROSS_NMS_IOU,
        compact_center_px=config.YOLO_NMS_CENTER_PX,
    )


def _detect_folder(folder: Path, weights: Path | None = None) -> list[YoloHit]:
    """Run YOLO on zoom tiles (preferred) or full_wing.jpg."""
    if not folder.is_dir() or not yolo_available(weights):
        return []

    tile_hits: list[YoloHit] = []
    zoom_files = sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and p.name.lower().startswith("plan_zoom_")
        ]
    )
    for path in zoom_files:
        try:
            with Image.open(path) as im:
                hits = detect_symbols(
                    im.convert("RGB"), image_name=path.name, weights=weights
                )
        except OSError:
            continue
        # De-dup within this tile only (primary vs rescue/TTA double boxes).
        # Cross-tile de-dup happens after mapping to shared ROI coordinates —
        # comparing raw tile-local boxes from different tiles merges distinct
        # symbols that merely sit at similar positions inside their own tiles.
        tile_hits.extend(
            nms_yolo_hits(
                hits,
                same_class_iou=config.YOLO_NMS_IOU,
                cross_class_iou=config.YOLO_CROSS_NMS_IOU,
                compact_center_px=config.YOLO_NMS_CENTER_PX,
            )
        )

    if tile_hits:
        rois = load_tile_rois(folder, default_overlap_pct=config.TILE_OVERLAP_PCT)
        if rois:
            merged = merge_overlapping_tile_hits(tile_hits, rois)
        else:
            # No tile geometry available — conservative global NMS fallback.
            merged = nms_yolo_hits(
                tile_hits,
                same_class_iou=config.YOLO_NMS_IOU,
                cross_class_iou=config.YOLO_CROSS_NMS_IOU,
                compact_center_px=config.YOLO_NMS_CENTER_PX,
            )
        # Extra full-wing pass for symbols too large to fit whole in any tile.
        full = folder / "full_wing.jpg"
        if config.YOLO_FULLWING_PASS and rois and full.is_file():
            wing_hits: list[YoloHit] = []
            try:
                with Image.open(full) as im:
                    wing_hits = detect_symbols(
                        im.convert("RGB"), image_name=full.name, weights=weights
                    )
            except OSError:
                wing_hits = []
            if wing_hits:
                before = len(merged)
                merged = merge_fullwing_large_hits(merged, wing_hits)
                added = len(merged) - before
                if added > 0:
                    logger.info(
                        "Full-wing pass added %d large-symbol detections in %s",
                        added,
                        folder.name,
                    )
        return merged

    full = folder / "full_wing.jpg"
    if full.is_file():
        try:
            with Image.open(full) as im:
                hits = detect_symbols(
                    im.convert("RGB"), image_name=full.name, weights=weights
                )
        except OSError:
            return []
        return nms_yolo_hits(
            hits,
            same_class_iou=config.YOLO_NMS_IOU,
            cross_class_iou=config.YOLO_CROSS_NMS_IOU,
            compact_center_px=config.YOLO_NMS_CENTER_PX,
        )
    return []


def _build_rows(
    legend: list[LegendEntry],
    hits: list[YoloHit],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_legend: Counter[int] = Counter()
    class_by_legend: dict[int, set[str]] = {}
    unmatched: Counter[str] = Counter()
    unmatched_reason: dict[str, str] = {}

    for hit in hits:
        if hit.legend_number is None:
            unmatched[hit.class_name] += max(1, int(hit.count_weight or 1))
            from pipeline.yolo_class_map import resolve_class_map

            if resolve_class_map(hit.class_name) is None:
                unmatched_reason[hit.class_name] = "detected — not in annotation-to-legend map"
            else:
                unmatched_reason[hit.class_name] = "detected — no matching legend row"
            continue
        by_legend[hit.legend_number] += max(1, int(hit.count_weight or 1))
        class_by_legend.setdefault(hit.legend_number, set()).add(hit.class_name)

    rows: list[dict[str, Any]] = []
    for entry in legend:
        yolo_n = int(by_legend.get(entry.number, 0))
        if yolo_n <= 0:
            continue
        rows.append(
            {
                "number": entry.number,
                "description": _strip_leading_index(entry.description),
                "detail_ref": entry.detail_ref,
                "count": yolo_n,
                "yolo_count": yolo_n,
                "yolo_classes": sorted(class_by_legend.get(entry.number, set())),
                "method": "yolo",
            }
        )

    unmatched_rows = [
        {
            "class_name": name,
            "count": int(cnt),
            "reason": unmatched_reason.get(name, "detected — not matched to legend"),
        }
        for name, cnt in sorted(unmatched.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return rows, unmatched_rows


def write_invoice_csv(out_dir: Path, invoice: dict[str, Any]) -> Path:
    """Human-readable invoice CSV (lines + subtotal/tax/grand total)."""
    import csv

    inv_csv = out_dir / "invoice.csv"
    with inv_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "description",
                "device_key",
                "quantity",
                "unit",
                "unit_price_usd",
                "line_total_usd",
                "yolo_classes",
            ],
        )
        writer.writeheader()
        for line in invoice.get("lines") or []:
            writer.writerow(
                {
                    "description": line.get("description"),
                    "device_key": line.get("device_key"),
                    "quantity": line.get("quantity"),
                    "unit": line.get("unit"),
                    "unit_price_usd": line.get("unit_price_usd"),
                    "line_total_usd": line.get("line_total_usd"),
                    "yolo_classes": "|".join(line.get("yolo_classes") or []),
                }
            )
        writer.writerow({})
        writer.writerow(
            {
                "description": "SUBTOTAL",
                "line_total_usd": invoice.get("subtotal_usd"),
            }
        )
        writer.writerow(
            {
                "description": f"TAX ({invoice.get('tax_rate')})",
                "line_total_usd": invoice.get("tax_usd"),
            }
        )
        writer.writerow(
            {
                "description": "GRAND TOTAL",
                "line_total_usd": invoice.get("grand_total_usd"),
            }
        )
    return inv_csv


def run_symbol_count_stage(
    job_root: Path,
    pages_out: list[dict[str, Any]],
    *,
    job_id: str,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Detect symbols on wing zooms, match to legend, write counts + invoice."""

    def _prog(stage: str, message: str) -> None:
        if progress:
            progress(stage, message)

    out_dir = job_root / COUNTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.YOLO_ENABLED:
        payload = {
            "method": "none",
            "yolo_enabled": False,
            "note": "YOLO_ENABLED=false",
            "rows": [],
            "total_count": 0,
            "invoice": None,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    if not yolo_available():
        payload = {
            "method": "none",
            "yolo_enabled": True,
            "note": f"YOLO model missing or ultralytics not installed: {config.YOLO_MODEL_PATH}",
            "rows": [],
            "total_count": 0,
            "invoice": None,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    _prog("counting", "Loading symbol legend for YOLO matching")
    legend = legend_from_symbol_table(job_root / "symbol_table.json")
    ensure_pricing_csv(refresh=True)

    # Model selection: score every available weight file against this PDF's
    # legend and use the one whose classes match the most legend rows (e.g. a
    # tech-symbol model for technology sets, a roof model for roofing sets).
    # If no model's classes appear in the legend, skip instead of producing
    # false positives.
    chosen_weights: Path | None = None
    if legend:
        from pipeline.legend_match import match_class_to_legend
        from pipeline.yolo_detect import available_model_paths, class_names

        scored: list[tuple[int, Path, list[str]]] = []
        for weights in available_model_paths():
            names = sorted(class_names(weights).values())
            matched = [
                name
                for name in names
                if match_class_to_legend(
                    name, legend, min_score=config.YOLO_MATCH_MIN_SCORE
                )
                is not None
            ]
            scored.append((len(matched), weights, names))
            logger.info(
                "[%s] model %s: %d/%d classes match the legend",
                job_id,
                weights.name,
                len(matched),
                len(names),
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] > 0:
            chosen_weights = scored[0][1]
            _prog(
                "counting",
                f"Using model {chosen_weights.name} "
                f"({scored[0][0]} classes match this PDF's legend)",
            )
        elif scored:
            all_classes = sorted({n for _, _, names in scored for n in names})
            note = (
                "Symbol detection skipped: this PDF's legend "
                f"({len(legend)} rows) contains none of the classes of any "
                f"available model ({', '.join(p.name for _, p, _ in scored)}). "
                f"Known classes: {', '.join(all_classes)}. Detections on this "
                "drawing set would be false positives."
            )
            invoice = build_invoice([], job_id=job_id, refresh_prices=False)
            payload = {
                "method": "skipped_legend_mismatch",
                "yolo_enabled": True,
                "yolo_model": None,
                "legend_entries": len(legend),
                "model_classes": all_classes,
                "note": note,
                "rows": [],
                "yolo_unmatched": [],
                "per_wing": [],
                "total_count": 0,
                "invoice": invoice,
            }
            (out_dir / "result.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            (out_dir / "invoice.json").write_text(
                json.dumps(invoice, indent=2), encoding="utf-8"
            )
            logger.warning("[%s] %s", job_id, note)
            return payload

    all_hits: list[YoloHit] = []
    det_dicts: list[dict[str, Any]] = []
    per_wing: list[dict[str, Any]] = []
    folders_seen = 0
    text_masked_total = 0
    dense_masked_total = 0
    rescued_legend_dropped = 0
    verifier_dropped_total = 0

    for page in pages_out:
        if not page.get("kept"):
            continue
        for wing in page.get("wings") or []:
            manifest_rel = wing.get("zooms_manifest")
            zooms_full = wing.get("zooms_full_image")
            folder: Path | None = None
            if manifest_rel:
                folder = (job_root / Path(manifest_rel).parent).resolve()
            elif zooms_full:
                folder = (job_root / Path(zooms_full).parent).resolve()
            if folder is None or not folder.is_dir():
                continue
            folders_seen += 1
            _prog(
                "counting",
                f"YOLO on {wing.get('name') or 'wing'} tiles ({folders_seen})",
            )
            raw = _detect_folder(folder, weights=chosen_weights)
            if raw and (config.TEXT_MASK_ENABLED or config.TEXT_DENSE_MASK_ENABLED):
                words = load_wing_text_words(job_root, page, wing, folder)
                if config.TEXT_MASK_ENABLED:
                    raw, masked = filter_hits_on_text(raw, words)
                    if masked:
                        text_masked_total += len(masked)
                        logger.info(
                            "[%s] %s: text mask dropped %d detections on labels (%s)",
                            job_id,
                            wing.get("name"),
                            len(masked),
                            ", ".join(
                                sorted({f"{m.class_name}@{m.confidence:.2f}" for m in masked})[:6]
                            ),
                        )
                if config.TEXT_DENSE_MASK_ENABLED:
                    regions = dense_text_regions(words)
                    if regions:
                        raw, region_masked = filter_hits_in_regions(raw, regions)
                        if region_masked:
                            dense_masked_total += len(region_masked)
                            logger.info(
                                "[%s] %s: dense-text region mask dropped %d detections "
                                "inside legend/title-block areas (%d regions)",
                                job_id,
                                wing.get("name"),
                                len(region_masked),
                                len(regions),
                            )
            if raw and verifier_available():
                full_wing = folder / "full_wing.jpg"
                if full_wing.is_file():
                    try:
                        with PILImage.open(full_wing) as wing_im:
                            raw, rejected = verify_hits(raw, wing_im.convert("RGB"))
                    except OSError:
                        rejected = []
                    if rejected:
                        verifier_dropped_total += len(rejected)
                        logger.info(
                            "[%s] %s: verifier rejected %d low-conf detections (%s)",
                            job_id,
                            wing.get("name"),
                            len(rejected),
                            ", ".join(
                                sorted({f"{m.class_name}@{m.confidence:.2f}" for m in rejected})[:6]
                            ),
                        )
            assigned = _assign_hits(raw, legend)
            if legend:
                assigned, dropped = drop_unmatched_rescues(assigned)
                if dropped:
                    rescued_legend_dropped += dropped
                    logger.info(
                        "[%s] %s: legend gate dropped %d rescued hits with no legend match",
                        job_id,
                        wing.get("name"),
                        dropped,
                    )
            all_hits.extend(assigned)
            folder_rel = str(folder.relative_to(job_root)).replace("\\", "/")
            for h in assigned:
                d = h.to_dict()
                d["wing_folder"] = folder_rel
                det_dicts.append(d)
            per_wing.append(
                {
                    "wing": wing.get("name"),
                    "page": page.get("page"),
                    "folder": folder_rel,
                    "yolo_count": len(assigned),
                    "matched": sum(1 for h in assigned if h.legend_number is not None),
                }
            )

    # Final cross-wing NMS is intentionally skipped — different wings are separate areas.
    rows, unmatched = _build_rows(legend, all_hits)
    total = int(sum(int(r["count"]) for r in rows))
    rescued_kept = sum(1 for h in all_hits if h.rescued)
    _prog("pricing", "Building cost invoice from symbol counts")
    invoice = build_invoice(rows, job_id=job_id, refresh_prices=True)

    payload: dict[str, Any] = {
        "method": "yolo",
        "yolo_enabled": True,
        "yolo_model": (chosen_weights or Path(config.YOLO_MODEL_PATH)).name,
        "yolo_conf": config.YOLO_CONF,
        "yolo_rescue": {
            "enabled": bool(config.YOLO_RESCUE_ENABLED),
            "conf": config.YOLO_RESCUE_CONF,
            "tta": bool(config.YOLO_RESCUE_TTA),
            "rescued_kept": rescued_kept,
            "rescued_legend_dropped": rescued_legend_dropped,
        },
        "text_mask": {
            "enabled": bool(config.TEXT_MASK_ENABLED),
            "masked_dropped": text_masked_total,
            "dense_region_dropped": dense_masked_total,
        },
        "verifier": {
            "enabled": verifier_available(),
            "max_conf": config.VERIFY_MAX_CONF,
            "drop_prob": config.VERIFY_DROP_PROB,
            "rejected": verifier_dropped_total,
        },
        "legend_entries": len(legend),
        "zoom_folders_processed": folders_seen,
        "total_count": total,
        "total_yolo": len(all_hits),
        "rows": rows,
        "yolo_unmatched": unmatched,
        "per_wing": per_wing,
        "yolo_detections": det_dicts,
        "invoice": invoice,
        "counts_dir": COUNTS_DIR,
        "note": (
            "YOLO detections on ROI zoom tiles, matched to extracted symbol legend, "
            "priced from pricing/device_prices.csv"
            + (
                f". A second verification pass (conf>={config.YOLO_RESCUE_CONF:.2f}"
                f"{', TTA' if config.YOLO_RESCUE_TTA else ''}) rescued "
                f"{rescued_kept} symbols the primary threshold missed; overlapping "
                "rescues are merged so each symbol counts once."
                if config.YOLO_RESCUE_ENABLED
                else ""
            )
            + (
                f" {text_masked_total} detections that sat on drawing text "
                "(labels/notes) were discarded via the PDF text layer."
                if config.TEXT_MASK_ENABLED and text_masked_total
                else ""
            )
        ),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "invoice.json").write_text(json.dumps(invoice, indent=2), encoding="utf-8")

    # Human-readable invoice CSV for the job
    write_invoice_csv(out_dir, invoice)

    # Printable invoice PDF alongside the CSV/JSON
    try:
        from pipeline.invoice_pdf import generate_invoice_pdf

        job_meta = None
        job_json = job_root / "job.json"
        if job_json.is_file():
            try:
                job_meta = json.loads(job_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                job_meta = None
        pdf_path = generate_invoice_pdf(job_root, job_meta)
        payload["invoice_pdf_path"] = f"{COUNTS_DIR}/{pdf_path.name}"
        (out_dir / "result.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        logger.exception("Invoice PDF generation failed for %s", job_id)

    logger.info(
        "Symbol count complete: %d matched units across %d legend rows (grand_total=$%.2f)",
        total,
        len(rows),
        float(invoice.get("grand_total_usd") or 0),
    )
    return payload
