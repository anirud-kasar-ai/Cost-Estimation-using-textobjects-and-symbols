"""Single-PDF job orchestrator: page → diagram crop → wing split."""

from __future__ import annotations

import json
import logging
import re
import shutil
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
from pipeline.symbol_count import run_symbol_count
from pipeline.wing_crop import assign_wing_instance_ids, crop_wings, slugify

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
05_metadata/  Per-page JSON: wing map, label points, CV pixel boxes.
06_symbol_counts/  Symbol count pivot CSV/PDF per wing instance (when enabled).
summary.json  Flat index + skipped pages.
job.json      Job status for the batch runner.

Reading order: 02_pages -> 03_drawings -> 04_wings -> 05_metadata.
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


def read_job(job_id: str) -> dict[str, Any]:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        raise FileNotFoundError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


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
        "created_at": _now(),
        "updated_at": _now(),
        "pages": [],
        "error": None,
    }
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
    try:
        originals = list((root / SOURCE_DIR).iterdir())
        if not originals:
            raise FileNotFoundError("uploaded file missing")
        source = originals[0]

        sheet_notes_info = None
        if source.suffix.lower() == ".pdf":
            _set_stage(job, "extracting", "Extracting requirements, symbols, and sheet notes")
            progress("extracting", "Extracting requirements, symbols, and sheet notes")

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
        for page_index, page_path in enumerate(page_paths, start=1):
            page_key = page_path.stem
            scan = scans.get(page_index)
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
            if not wing_map and config.VISION_REQUIRE_LLM:
                raise RuntimeError(f"Vision model returned no wing regions for {page_key}")
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

        symbol_instances = 0
        if config.SYMBOL_COUNT_ENABLED and source.suffix.lower() == ".pdf":
            _set_stage(job, "counting_symbols", "Counting symbols per wing")
            progress("counting_symbols", "Counting symbols on wing images with vision model")
            try:
                count_result = run_symbol_count(root, on_progress=progress)
                symbol_instances = int(count_result.get("instances") or 0)
                job.update(
                    {
                        "symbol_count_status": count_result.get("status"),
                        "symbol_count_instances": symbol_instances,
                        "symbol_count_detail_path": count_result.get(
                            "symbol_count_detail_path"
                        ),
                        "symbol_count_report_csv": count_result.get(
                            "symbol_count_report_csv"
                        ),
                        "symbol_count_report_pdf": count_result.get(
                            "symbol_count_report_pdf"
                        ),
                        "symbol_count_report_json": count_result.get(
                            "symbol_count_report_json"
                        ),
                    }
                )
                if count_result.get("reason"):
                    job["symbol_count_reason"] = count_result["reason"]
            except Exception as exc:  # noqa: BLE001
                logger.exception("Symbol count failed for job %s", job_id)
                job["symbol_count_status"] = "failed"
                job["symbol_count_error"] = str(exc)

        job["status"] = "done"
        job["stage"] = "done"
        base_message = (
            f"Done — {diagram_count} diagram(s) and {wing_count} wing image(s) "
            f"from {kept_pages} of {len(page_paths)} page(s)"
        )
        if job.get("symbol_count_status") == "done" and symbol_instances:
            job["message"] = f"{base_message}; symbol counts for {symbol_instances} wing instance(s)"
        elif job.get("symbol_count_status") == "failed":
            job["message"] = f"{base_message}; symbol count failed (see symbol_count_error)"
        else:
            job["message"] = base_message

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
