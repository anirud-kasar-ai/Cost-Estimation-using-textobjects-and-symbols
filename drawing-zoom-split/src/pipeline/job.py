"""Single-PDF job orchestrator: page → diagram crop → wing split."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from pipeline import config
from pipeline.diagram_crop import PixelBox, crop_drawing
from pipeline.llama_vision import get_llama_client
from pipeline.page_render import render_pages
from pipeline.sheet_scan import PageScan, WingHint, _in_title_block, scan_pdf_pages
from pipeline.roi_zoom import export_wing_roi_and_zooms
from pipeline.extraction.stages import finalize_sheet_notes, run_extraction_stages
from pipeline.wing_crop import assign_wing_instance_ids, crop_wings, slugify
from pipeline.count_stage import COUNTS_DIR, run_symbol_count_stage

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str], None]

SOURCE_DIR = "01_source"
PAGES_DIR = "02_pages"
DRAWINGS_DIR = "03_drawings"
WINGS_DIR = "04_wings"
METADATA_DIR = "05_metadata"

FOLDER_GUIDE = """Dual-pathway drawing split output
=================================

01_source/    Original PDF, unchanged.
02_pages/     Full sheet renders — only plan/diagram pages are kept.
03_drawings/  One diagram crop per page (title block, notes removed).
04_wings/     One image per wing, grouped by wing name.
              Folder: B-WING-EAST/     all B-WING-EAST crops in this job
              Name:   page_004.jpg     source page number
              rois/ + zooms/           ink-tight ROI and overlapping zoom tiles
05_metadata/  Per-page JSON: wing map, label points, CV pixel boxes.
06_counts/    YOLO symbol counts + invoice.json / invoice.csv
symbol_table.json  Extracted symbol legend rows.
summary.json  Flat index + skipped pages.
job.json      Job status for the batch runner / UI.

End-to-end: PDF upload → legend extract → diagram/wing crop → ROI zoom
→ YOLO symbol detect → pricing CSV → cost invoice.

Reading order: 02_pages -> 03_drawings -> 04_wings -> 05_metadata -> 06_counts.
"""

_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# Rough per-unit costs (seconds) used for the upfront time estimate shown in
# the UI. They only seed the first guess — once the job is running, the real
# per-page and per-wing timings observed so far take over.
EST_PAGE_SEC = 12.0        # LLM classify + wing map + CV crop, per page
EST_WING_COUNT_SEC = 20.0  # YOLO across one wing's zoom tiles
EST_WINGS_PER_PAGE = 1.6   # typical wings per page until the pages tell us


def _pdf_page_count(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        import fitz

        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:  # noqa: BLE001
        return None


def initial_estimate_seconds(page_count: int) -> int:
    """First guess before we know how many wings the PDF has."""
    per_page = EST_PAGE_SEC + (
        EST_WINGS_PER_PAGE * EST_WING_COUNT_SEC if config.SYMBOL_COUNT_ENABLED else 0.0
    )
    return int(round(max(1, page_count) * per_page))


def _apply_eta(job: dict[str, Any], elapsed: float, eta: float) -> None:
    eta = max(0.0, eta)
    job["eta_seconds"] = int(round(eta))
    job["estimated_total_seconds"] = int(round(elapsed + eta))
    total = elapsed + eta
    job["progress"] = round(elapsed / total, 4) if total > 0 else 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_id_from_filename(filename: str) -> str:
    stem = Path(filename or "upload").stem.strip()
    stem = _INVALID_FOLDER_CHARS.sub("_", stem).strip(" .")
    stem = re.sub(r"_+", "_", stem)
    if not stem:
        stem = "upload"
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    return stem[:120]


def job_dir(job_id: str) -> Path:
    return config.STORAGE_DIR / job_id


def prepare_job_folder(job_id: str) -> Path:
    root = job_dir(job_id)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _source_pdf(job_id: str) -> Path | None:
    source = job_dir(job_id) / SOURCE_DIR
    if not source.is_dir():
        return None
    return next(iter(sorted(source.glob("*.pdf"))), None)


# Client info extracted from cover sheets of jobs created before extraction
# existed — cached per process so each PDF is only parsed once.
_client_cache: dict[str, dict[str, str]] = {}


def ensure_client_info(data: dict[str, Any]) -> None:
    """Fill data['client'] from the drawing's cover sheet when not set yet."""
    if data.get("client"):
        return
    job_id = str(data.get("id") or "")
    if not job_id:
        return
    cached = _client_cache.get(job_id)
    if cached is None:
        pdf = _source_pdf(job_id)
        client = None
        if pdf is not None:
            from pipeline.client_info import extract_client_info

            client = extract_client_info(pdf)
        cached = client or {}
        _client_cache[job_id] = cached
    if cached:
        data["client"] = dict(cached)


def read_job(job_id: str) -> dict[str, Any]:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        raise FileNotFoundError(job_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    # Jobs created before size tracking existed: fill in from the stored PDF.
    if not data.get("size_bytes"):
        pdf = _source_pdf(job_id)
        if pdf is not None:
            data["size_bytes"] = pdf.stat().st_size
    ensure_client_info(data)
    return data


def write_job(job_id: str, data: dict[str, Any]) -> None:
    dest = job_dir(job_id)
    dest.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now()
    (dest / "job.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_job(job_id: str, filename: str, source: Path) -> dict[str, Any]:
    root = prepare_job_folder(job_id)
    original = root / SOURCE_DIR
    original.mkdir(parents=True, exist_ok=True)
    stored = original / source.name
    shutil.copy2(source, stored)
    (root / "READ_ME_FIRST.txt").write_text(FOLDER_GUIDE, encoding="utf-8")
    data = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "message": "Waiting to start",
        "filename": filename,
        "size_bytes": stored.stat().st_size,
        "created_at": _now(),
        "updated_at": _now(),
        "pages": [],
        "error": None,
    }
    # Best-effort Bill To details from the cover sheet's title block.
    try:
        from pipeline.client_info import extract_client_info

        client = extract_client_info(stored)
        if client:
            data["client"] = client
    except Exception:  # noqa: BLE001
        logger.warning("Could not extract client info for %s", job_id, exc_info=True)
    # Upfront time estimate for the UI: wings are unknown until pages are
    # processed, so start from the page count and typical wings-per-page.
    page_count = _pdf_page_count(stored)
    if page_count:
        data["page_count"] = page_count
        estimate = initial_estimate_seconds(page_count)
        data["estimated_total_seconds"] = estimate
        data["eta_seconds"] = estimate
        data["progress"] = 0.0
        data["message"] = (
            f"Waiting to start — {page_count} page(s), "
            f"estimated ~{max(1, round(estimate / 60))} min"
        )
    write_job(job_id, data)
    return data


def _set_stage(job: dict[str, Any], stage: str, message: str) -> None:
    job["status"] = "processing"
    job["stage"] = stage
    job["message"] = message
    write_job(job["id"], job)


def _scan_hints_on_diagram(
    scan: PageScan | None,
    *,
    page_size: tuple[int, int],
    diagram_box: PixelBox,
) -> list[dict[str, Any]]:
    if scan is None or not scan.wing_hints:
        return []
    page_w, page_h = page_size
    diagram_w = max(1.0, float(diagram_box.x2 - diagram_box.x1))
    diagram_h = max(1.0, float(diagram_box.y2 - diagram_box.y1))
    mapped: list[dict[str, Any]] = []
    for hint in scan.wing_hints:
        x1, y1, x2, y2 = hint.box
        local = [
            (x1 * page_w - diagram_box.x1) / diagram_w,
            (y1 * page_h - diagram_box.y1) / diagram_h,
            (x2 * page_w - diagram_box.x1) / diagram_w,
            (y2 * page_h - diagram_box.y1) / diagram_h,
        ]
        local = [max(0.0, min(1.0, value)) for value in local]
        if local[2] - local[0] < 0.04 or local[3] - local[1] < 0.04:
            continue
        mapped.append(
            {"name": slugify(hint.name), "box": local, "source": "pdf_text"}
        )
    return mapped


def _merge_wing_maps(
    llm_wings: list[dict[str, Any]],
    pdf_hints: list[dict[str, Any]],
    label_points: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge PDF hints, label points, and vision wings — keep duplicate names."""
    merged: list[dict[str, Any]] = []

    for hint in pdf_hints:
        name = slugify(str(hint.get("name") or ""))
        if name:
            merged.append({**hint, "name": name})

    labels = label_points or []
    for point in labels:
        name = slugify(str(point.get("name") or ""))
        px, py = point.get("point") or [None, None]
        if not name or px is None or py is None:
            continue
        covered = False
        for item in merged:
            if item["name"] != name:
                continue
            box = item.get("box")
            if box and box[0] <= float(px) <= box[2] and box[1] <= float(py) <= box[3]:
                covered = True
                break
            item_point = item.get("point")
            if item_point and abs(float(item_point[0]) - float(px)) < 0.02:
                if abs(float(item_point[1]) - float(py)) < 0.02:
                    covered = True
                    break
        if not covered:
            merged.append(
                {
                    "name": name,
                    "source": "pdf_text",
                    "point": point.get("point"),
                }
            )

    for wing in llm_wings:
        name = slugify(str(wing.get("name") or ""))
        box = wing.get("box")
        if not name:
            continue
        attached = False
        if box:
            for item in merged:
                if item["name"] != name or item.get("box"):
                    continue
                item_point = item.get("point")
                if item_point:
                    cx = (float(box[0]) + float(box[2])) / 2.0
                    cy = (float(box[1]) + float(box[3])) / 2.0
                    px, py = float(item_point[0]), float(item_point[1])
                    if abs(px - cx) > 0.08 or abs(py - cy) > 0.08:
                        continue
                item["box"] = box
                attached = True
                break
        if not attached:
            merged.append({**wing, "name": name})

    return assign_wing_instance_ids(merged)


def _points_on_diagram(
    hints: list[WingHint],
    *,
    page_size: tuple[int, int],
    diagram_box: PixelBox,
) -> list[dict[str, Any]]:
    page_w, page_h = page_size
    diagram_w = max(1.0, float(diagram_box.x2 - diagram_box.x1))
    diagram_h = max(1.0, float(diagram_box.y2 - diagram_box.y1))
    points: list[dict[str, Any]] = []
    for hint in hints:
        x1, y1, x2, y2 = hint.box
        cx = ((x1 + x2) / 2.0 * page_w - diagram_box.x1) / diagram_w
        cy = ((y1 + y2) / 2.0 * page_h - diagram_box.y1) / diagram_h
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        points.append({"name": slugify(hint.name), "point": [cx, cy]})
    return points


def _write_summary(
    root: Path, job: dict[str, Any], pages_out: list[dict[str, Any]]
) -> None:
    entries: list[dict[str, Any]] = []
    for page in pages_out:
        if not page.get("kept"):
            continue
        for wing in page.get("wings") or []:
            entries.append(
                {
                    "page": page["page"],
                    "sheet_no": page.get("sheet_no"),
                    "title": page.get("title"),
                    "page_image": page.get("page_image"),
                    "diagram_image": page.get("diagram"),
                    "wing": wing["name"],
                    "wing_image": wing["image"],
                    "metadata": f"{METADATA_DIR}/{page['page_key']}.json",
                }
            )
    summary = {
        "file": job.get("filename"),
        "folder": root.name,
        "pages_total": job.get("page_count"),
        "pages_kept": sum(1 for p in pages_out if p.get("kept")),
        "diagrams": job.get("drawing_count"),
        "wings": job.get("wing_count"),
        "skipped_pages": [
            {"page": page["page"], "reason": page.get("reason") or "no diagram"}
            for page in pages_out
            if not page.get("kept")
        ],
        "items": entries,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def process_job(job_id: str, *, on_progress: ProgressFn | None = None) -> dict[str, Any]:
    """Process one PDF job: classify pages, crop diagrams, split wings."""

    def progress(stage: str, message: str) -> None:
        if on_progress:
            on_progress(stage, message)

    job = read_job(job_id)
    root = job_dir(job_id)
    started = time.monotonic()
    try:
        originals = list((root / SOURCE_DIR).iterdir())
        if not originals:
            raise FileNotFoundError("uploaded file missing")
        source = originals[0]

        sheet_notes_info = None
        if source.suffix.lower() == ".pdf":
            extract_msg = (
                "Extracting requirements, symbols, and sheet notes"
                if config.SHEET_NOTES_ENABLED
                else "Extracting requirements and symbols"
            )
            _set_stage(job, "extracting", extract_msg)
            progress("extracting", extract_msg)

            def extraction_progress(message: str) -> None:
                _set_stage(job, "extracting", message)
                progress("extracting", message)

            extraction, sheet_notes_info = run_extraction_stages(
                source,
                job.get("filename") or source.name,
                root,
                on_progress=extraction_progress,
            )
            job.update(extraction)
            write_job(job_id, job)

        _set_stage(job, "rendering", "Rendering pages")
        progress("rendering", f"Rendering {source.name}")
        page_paths = render_pages(source, root / PAGES_DIR)
        job["page_count"] = len(page_paths)
        _apply_eta(
            job,
            time.monotonic() - started,
            initial_estimate_seconds(len(page_paths)),
        )

        scans: dict[int, PageScan] = {}
        if source.suffix.lower() == ".pdf":
            scans = {scan.page: scan for scan in scan_pdf_pages(source)}
        write_job(job_id, job)

        _set_stage(job, "loading_model", "Connecting to vision model")
        progress("loading_model", "Connecting to vision model")
        client = get_llama_client()
        if not client.ready:
            raise RuntimeError(client._load_error or "Vision model is unavailable")
        job["vision_model"] = {
            "model": client.model_id,
            "runtime": config.VISION_LLM_RUNTIME,
            "endpoint": config.HF_INFERENCE_ENDPOINT or None,
            "ready": True,
            "required": config.VISION_REQUIRE_LLM,
        }
        write_job(job_id, job)

        pages_out: list[dict[str, Any]] = []
        diagram_count = 0
        wing_count = 0
        loop_started = time.monotonic()
        for page_index, page_path in enumerate(page_paths, start=1):
            page_key = page_path.stem
            scan = scans.get(page_index)

            # Refine the ETA from real timings: pages processed so far give the
            # per-page speed, and the wings found so far predict how many wings
            # the remaining pages will add to the counting stage.
            pages_done = page_index - 1
            per_page = (
                (time.monotonic() - loop_started) / pages_done
                if pages_done
                else EST_PAGE_SEC
            )
            wings_per_page = (
                wing_count / pages_done if pages_done else EST_WINGS_PER_PAGE
            )
            pages_left = len(page_paths) - pages_done
            predicted_wings = wing_count + pages_left * wings_per_page
            count_eta = (
                predicted_wings * EST_WING_COUNT_SEC
                if config.SYMBOL_COUNT_ENABLED
                else 0.0
            )
            _apply_eta(
                job,
                time.monotonic() - started,
                pages_left * per_page + count_eta,
            )

            _set_stage(
                job,
                "classifying",
                f"Checking {page_key} ({page_index}/{len(page_paths)})",
            )
            progress("classifying", f"{page_key} ({page_index}/{len(page_paths)})")
            page_image = Image.open(page_path).convert("RGB")
            classification = client.classify_page(page_image)
            has_diagram = bool(classification.get("has_diagram"))
            plan_area = classification.get("plan_area")
            reason = classification.get("reason") or ""

            page_record: dict[str, Any] = {
                "page": page_index,
                "page_key": page_key,
                "sheet_no": scan.sheet_no if scan else None,
                "title": scan.title if scan else page_key,
                "classification_source": "vision_model",
                "kept": has_diagram,
                "has_diagram": has_diagram,
                "reason": reason,
                "page_image": None,
                "diagram": None,
                "wings": [],
            }

            if not has_diagram or not plan_area:
                page_path.unlink(missing_ok=True)
                pages_out.append(page_record)
                job["pages"] = pages_out
                write_job(job_id, job)
                continue

            page_record["page_image"] = f"{PAGES_DIR}/{page_path.name}"

            _set_stage(job, "drawing_crop", f"Cropping diagram on {page_key}")
            page_label_points = [
                (
                    (label.box[0] + label.box[2]) / 2.0,
                    (label.box[1] + label.box[3]) / 2.0,
                )
                for label in (scan.wing_labels if scan else [])
                if not _in_title_block(label.box)
            ]
            diagram, abs_box = crop_drawing(page_image, plan_area, page_label_points)

            _set_stage(job, "wing_map", f"Mapping wings on {page_key}")
            llm_wings = client.locate_wings(diagram)
            pdf_hints = _scan_hints_on_diagram(
                scan,
                page_size=page_image.size,
                diagram_box=abs_box,
            )
            label_points = _points_on_diagram(
                scan.wing_labels if scan else [],
                page_size=page_image.size,
                diagram_box=abs_box,
            )
            label_points = [
                point
                for point in label_points
                if not _in_title_block(
                    [point["point"][0], point["point"][1], point["point"][0], point["point"][1]]
                )
            ]
            label_points = assign_wing_instance_ids(label_points)
            on_diagram = {point["name"] for point in label_points}
            pdf_hints = [hint for hint in pdf_hints if hint["name"] in on_diagram]
            wing_map = _merge_wing_maps(llm_wings, pdf_hints, label_points)
            if not wing_map:
                page_path.unlink(missing_ok=True)
                page_record["kept"] = False
                page_record["reason"] = "vision model returned no wing regions"
                pages_out.append(page_record)
                job["pages"] = pages_out
                write_job(job_id, job)
                logger.warning("Skipping %s — no wing regions from vision/PDF text", page_key)
                continue
            tag_points = _points_on_diagram(
                scan.room_tags if scan else [],
                page_size=page_image.size,
                diagram_box=abs_box,
            )

            _set_stage(job, "wing_crop", f"Cutting wing images for {page_key}")
            regions = crop_wings(diagram, wing_map, label_points, tag_points)
            if not regions:
                page_path.unlink(missing_ok=True)
                page_record["kept"] = False
                page_record["reason"] = "no usable wing regions after CV crop"
                pages_out.append(page_record)
                job["pages"] = pages_out
                write_job(job_id, job)
                continue

            diagram_count += 1
            drawings_dir = root / DRAWINGS_DIR
            drawings_dir.mkdir(parents=True, exist_ok=True)
            diagram_name = f"{page_key}_diagram.jpg"
            diagram_path = drawings_dir / diagram_name
            diagram.save(diagram_path, format="JPEG", quality=94, optimize=True)
            diagram_rel = f"{DRAWINGS_DIR}/{diagram_name}"

            wings_out: list[dict[str, Any]] = []
            for wing_index, region in enumerate(regions, start=1):
                wing_count += 1
                wing_label = region.name
                wing_folder = root / WINGS_DIR / wing_label
                wing_folder.mkdir(parents=True, exist_ok=True)
                file_name = f"{page_key}.jpg"
                duplicate = 2
                while (wing_folder / file_name).exists():
                    file_name = f"{page_key}_{duplicate}.jpg"
                    duplicate += 1
                region.image.save(
                    wing_folder / file_name,
                    format="JPEG",
                    quality=94,
                    optimize=True,
                )

                # Unique subfolder per duplicate (page_004, page_004_2, …) so ROI/zoom
                # artifacts are not overwritten when the same wing name appears twice.
                instance_dir_name = Path(file_name).stem

                roi_payload = export_wing_roi_and_zooms(
                    wing_image=region.image,
                    wing_folder=wing_folder,
                    wing_rel_prefix=f"{WINGS_DIR}/{wing_label}",
                    instance_dir_name=instance_dir_name,
                    wing_name=wing_label,
                    tile_size=config.ROI_ZOOM_TILE_SIZE,
                    overlap_pct=config.ROI_ZOOM_OVERLAP_PCT,
                )
                wing_w, wing_h = region.image.size
                wings_out.append(
                    {
                        "index": wing_index,
                        "name": wing_label,
                        "instance_id": region.instance_id or region.name,
                        "slug": region.slug,
                        "source": region.source,
                        "image": f"{WINGS_DIR}/{wing_label}/{file_name}",
                        "vision_box_normalized": region.vision_box,
                        "cv_box_on_drawing": {
                            "x1": region.box.x1,
                            "y1": region.box.y1,
                            "x2": region.box.x2,
                            "y2": region.box.y2,
                        },
                        "cv_box_on_page": {
                            "x1": abs_box.x1 + region.box.x1,
                            "y1": abs_box.y1 + region.box.y1,
                            "x2": abs_box.x1 + region.box.x2,
                            "y2": abs_box.y1 + region.box.y2,
                        },
                        "size_pixels": {"width": wing_w, "height": wing_h},
                        "roi_box_on_wing": roi_payload.get("roi_box_on_wing"),
                        "roi_image": roi_payload.get("roi_image"),
                        "roi_overlay_image": roi_payload.get("roi_overlay_image"),
                        "zooms_manifest": roi_payload.get("zooms_manifest"),
                        "zooms_full_image": roi_payload.get("zooms_full_image"),
                    }
                )

            page_meta = {
                "page": page_index,
                "page_key": page_key,
                "sheet_no": page_record["sheet_no"],
                "title": page_record["title"],
                "page_image": page_record["page_image"],
                "diagram_image": diagram_rel,
                "vision_plan_area": plan_area,
                "cv_box_on_page": {
                    "x1": abs_box.x1,
                    "y1": abs_box.y1,
                    "x2": abs_box.x2,
                    "y2": abs_box.y2,
                },
                "wing_count": len(wings_out),
                "wings_without_image": sorted(
                    {
                        w.get("instance_id") or slugify(w["name"])
                        for w in wing_map
                    }
                    - {
                        w.get("instance_id") or slugify(w["name"])
                        for w in wings_out
                    }
                    - {"FULL-PLAN"}
                ),
                "vision_wing_map": [
                    {
                        "name": w["name"],
                        "box": w.get("box"),
                        "source": w.get("source"),
                    }
                    for w in llm_wings
                ],
                "wing_map": [
                    {
                        "name": w["name"],
                        "instance_id": w.get("instance_id"),
                        "box": w.get("box"),
                        "source": w.get("source"),
                    }
                    for w in wing_map
                ],
                "label_points": label_points,
                "tag_points": tag_points,
                "wings": wings_out,
            }
            meta_dir = root / METADATA_DIR
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"{page_key}.json").write_text(
                json.dumps(page_meta, indent=2),
                encoding="utf-8",
            )

            page_record["diagram"] = diagram_rel
            page_record["wings"] = wings_out
            pages_out.append(page_record)
            job["pages"] = pages_out
            write_job(job_id, job)

        kept_pages = sum(1 for page in pages_out if page["kept"])
        job["drawing_count"] = diagram_count
        job["wing_count"] = wing_count
        job["error"] = None
        if sheet_notes_info is not None:
            job["diagram_notes_written"] = finalize_sheet_notes(sheet_notes_info, root)

        # YOLO symbol count on zoom tiles → pricing invoice
        if config.SYMBOL_COUNT_ENABLED:
            _set_stage(job, "counting", "Running YOLO symbol detection on zoom tiles")

            # Every wing folder about to be scanned reports in through the
            # progress callback, so the ETA can follow the real per-wing speed.
            folders_total = sum(
                1
                for page in pages_out
                if page.get("kept")
                for wing in page.get("wings") or []
                if wing.get("zooms_manifest") or wing.get("zooms_full_image")
            )
            count_started = time.monotonic()
            count_state = {"started": 0}

            def _count_progress(stage: str, msg: str) -> None:
                if stage == "counting" and msg.startswith("YOLO on"):
                    count_state["started"] += 1
                    finished = count_state["started"] - 1
                    per_folder = (
                        (time.monotonic() - count_started) / finished
                        if finished > 0
                        else EST_WING_COUNT_SEC
                    )
                    left = max(0, folders_total - finished)
                    _apply_eta(job, time.monotonic() - started, left * per_folder)
                _set_stage(job, stage, msg)

            count_payload = run_symbol_count_stage(
                root,
                pages_out,
                job_id=job_id,
                progress=_count_progress,
            )
            invoice = count_payload.get("invoice") or {}
            job["symbol_count_total"] = count_payload.get("total_count", 0)
            job["symbol_count_rows"] = len(count_payload.get("rows") or [])
            job["counts_path"] = f"{COUNTS_DIR}/result.json"
            job["invoice_path"] = f"{COUNTS_DIR}/invoice.json"
            job["invoice_csv_path"] = f"{COUNTS_DIR}/invoice.csv"
            job["invoice_grand_total_usd"] = invoice.get("grand_total_usd")
            job["invoice_subtotal_usd"] = invoice.get("subtotal_usd")
            job["yolo_enabled"] = bool(count_payload.get("yolo_enabled"))
            job["yolo_model"] = count_payload.get("yolo_model")
        else:
            count_payload = None

        job["status"] = "done"
        job["stage"] = "done"
        job["progress"] = 1.0
        job["eta_seconds"] = 0
        job["actual_seconds"] = int(round(time.monotonic() - started))
        if count_payload and count_payload.get("method") == "yolo":
            job["message"] = (
                f"Done — {diagram_count} diagram(s), {wing_count} wing(s), "
                f"{job.get('symbol_count_total', 0)} symbols counted, "
                f"invoice ${job.get('invoice_grand_total_usd', 0):.2f}"
            )
        else:
            job["message"] = (
                f"Done — {diagram_count} diagram(s) and {wing_count} wing image(s) "
                f"from {kept_pages} of {len(page_paths)} page(s)"
            )

        write_job(job_id, job)
        _write_summary(root, job, pages_out)
        return job
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        job["status"] = "failed"
        job["stage"] = "failed"
        job["message"] = str(exc)
        job["error"] = str(exc)
        write_job(job_id, job)
        return job
