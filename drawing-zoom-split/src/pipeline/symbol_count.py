"""Count legend symbols per wing using vision LLM on direct wing crop images."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from pipeline import config
from pipeline.extraction.symbol_table_extractor import (
    SymbolTableInfo,
    catalog_id_for_entry,
    load_symbol_table_json,
)
from pipeline.llama_vision import LlamaVisionClient, downscale_for_llama, get_llama_client
from pipeline.roi_zoom import DEFAULT_TILE_SIZE
from pipeline.symbol_count_cv import (
    _load_wing_regions,
    aggregate_results_by_wing_name,
    select_wings_for_counting,
)
from pipeline.symbol_count_report import write_symbol_count_reports

logger = logging.getLogger(__name__)

METADATA_DIR = "05_metadata"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )


@dataclass
class SymbolDetection:
    symbol: str
    box: BoundingBox
    score: float
    tile_file: str | None = None
    source: str = "llm"


@dataclass
class WingInstance:
    column_key: str
    wing_name: str
    page_key: str
    instance_dir_name: str
    zooms_dir: Path
    manifest_path: Path
    full_wing_path: Path | None
    manifest: dict[str, Any] = field(default_factory=dict)


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_by_class(
    detections: list[SymbolDetection],
    iou_threshold: float,
) -> list[SymbolDetection]:
    """Greedy class-wise NMS in wing-image pixel coordinates."""
    iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
    by_label: dict[str, list[SymbolDetection]] = {}
    for det in detections:
        by_label.setdefault(det.symbol, []).append(det)

    kept: list[SymbolDetection] = []
    for group in by_label.values():
        ordered = sorted(group, key=lambda d: d.score, reverse=True)
        suppressed = [False] * len(ordered)
        for i, det in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(det)
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                if box_iou(det.box, ordered[j].box) >= iou_threshold:
                    suppressed[j] = True
    return kept


def translate_norm_box_to_roi(
    norm_box: list[float],
    *,
    tile_bbox_roi: dict[str, int],
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Map a normalized 0-1 box on a tile image to ROI pixel coordinates."""
    x1n, y1n, x2n, y2n = [float(v) for v in norm_box]
    tx1 = float(tile_bbox_roi["x1"])
    ty1 = float(tile_bbox_roi["y1"])
    tw = max(1.0, float(image_width))
    th = max(1.0, float(image_height))
    return BoundingBox(
        x1=tx1 + x1n * tw,
        y1=ty1 + y1n * th,
        x2=tx1 + x2n * tw,
        y2=ty1 + y2n * th,
    )


def translate_full_image_norm_box_to_roi(
    norm_box: list[float],
    *,
    roi_width: int,
    roi_height: int,
) -> BoundingBox:
    x1n, y1n, x2n, y2n = [float(v) for v in norm_box]
    rw = max(1.0, float(roi_width))
    rh = max(1.0, float(roi_height))
    return BoundingBox(
        x1=x1n * rw,
        y1=y1n * rh,
        x2=x2n * rw,
        y2=y2n * rh,
    )


def discover_wing_instances(job_root: Path) -> list[WingInstance]:
    """Collect wing instances from per-page metadata JSON files (legacy zoom path)."""
    meta_dir = job_root / METADATA_DIR
    if not meta_dir.is_dir():
        return []

    instances: list[WingInstance] = []
    for meta_path in sorted(meta_dir.glob("page_*.json")):
        try:
            page_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        page_key = str(page_meta.get("page_key") or meta_path.stem)
        for wing in page_meta.get("wings") or []:
            if not isinstance(wing, dict):
                continue
            wing_name = str(wing.get("name") or "")
            image_rel = str(wing.get("image") or "")
            manifest_rel = str(wing.get("zooms_manifest") or "")
            if not wing_name or not image_rel or not manifest_rel:
                continue
            instance_dir_name = Path(image_rel).stem
            manifest_path = job_root / manifest_rel
            zooms_dir = manifest_path.parent
            full_rel = wing.get("zooms_full_image")
            full_wing_path = job_root / str(full_rel) if full_rel else None
            manifest: dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = {}
            column_key = f"{wing_name} ({instance_dir_name})"
            instances.append(
                WingInstance(
                    column_key=column_key,
                    wing_name=wing_name,
                    page_key=page_key,
                    instance_dir_name=instance_dir_name,
                    zooms_dir=zooms_dir,
                    manifest_path=manifest_path,
                    full_wing_path=full_wing_path,
                    manifest=manifest,
                )
            )
    instances.sort(key=lambda item: (item.wing_name, item.page_key, item.instance_dir_name))
    return instances


def _legend_label_map(legend: SymbolTableInfo) -> dict[str, str]:
    """Map LLM catalog ids to report row keys."""
    mapping: dict[str, str] = {}
    desc_ids: dict[str, list[str]] = defaultdict(list)
    for entry in legend.entries:
        catalog_id = catalog_id_for_entry(entry)
        if not catalog_id:
            continue
        mapping[catalog_id] = catalog_id
        mapping[catalog_id.upper()] = catalog_id
        desc = str(entry.description or "").strip()
        if desc:
            desc_ids[desc].append(catalog_id)
    for desc, ids in desc_ids.items():
        if len(ids) == 1:
            mapping[desc] = ids[0]
            mapping[desc.upper()] = ids[0]
    return mapping


def _resolve_catalog_id(symbol: str, legend: SymbolTableInfo) -> str:
    """Map model output to a unique legend catalog id."""
    label = str(symbol or "").strip()
    if not label:
        return label
    label_map = _legend_label_map(legend)
    if label in label_map:
        return label_map[label]
    upper = label.upper()
    if upper in label_map:
        return label_map[upper]
    matches = [
        entry
        for entry in legend.entries
        if str(entry.description or "").strip().upper() == upper
    ]
    if len(matches) == 1:
        return catalog_id_for_entry(matches[0])
    if len(matches) > 1:
        for entry in matches:
            if str(entry.part_number or "").upper() == "WM2300":
                return catalog_id_for_entry(entry)
        return catalog_id_for_entry(matches[0])
    return label


def _load_zoom_manifest(
    job_root: Path,
    wing: dict[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    rel = str(wing.get("zooms_manifest") or "").strip()
    if not rel:
        return None
    manifest_path = job_root / rel
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or not manifest.get("tiles"):
        return None
    return manifest, manifest_path.parent


def _roi_dimensions(
    manifest: dict[str, Any] | None,
    wing_image_path: Path | None,
) -> tuple[int, int]:
    if manifest:
        roi_box = manifest.get("roi_box_px") or {}
        if all(k in roi_box for k in ("x1", "y1", "x2", "y2")):
            return (
                max(1, int(roi_box["x2"]) - int(roi_box["x1"])),
                max(1, int(roi_box["y2"]) - int(roi_box["y1"])),
            )
    if wing_image_path and wing_image_path.is_file():
        with Image.open(wing_image_path) as img:
            return img.size
    tile_size = int((manifest or {}).get("tile_size") or DEFAULT_TILE_SIZE)
    return tile_size, tile_size


def _detections_from_llm(
    client: LlamaVisionClient,
    image: Image.Image,
    legend: SymbolTableInfo,
    *,
    tile_file: str | None,
) -> list[SymbolDetection]:
    raw = client.detect_symbols_in_tile(image, legend.entries)
    min_conf = config.SYMBOL_COUNT_MIN_CONFIDENCE
    label_map = _legend_label_map(legend)
    out: list[SymbolDetection] = []
    w, h = image.size
    for item in raw:
        symbol = str(item.get("symbol") or "").strip()
        box = item.get("box")
        if not symbol or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            score = float(item.get("confidence", item.get("score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        if score < min_conf:
            continue
        norm = [max(0.0, min(1.0, float(v))) for v in box]
        x1, y1, x2, y2 = norm
        if x2 <= x1 or y2 <= y1:
            continue
        display = _resolve_catalog_id(symbol, legend)
        pixel_box = BoundingBox(
            x1=x1 * w,
            y1=y1 * h,
            x2=x2 * w,
            y2=y2 * h,
        )
        out.append(
            SymbolDetection(
                symbol=display,
                box=pixel_box,
                score=score,
                tile_file=tile_file,
            )
        )
    return out


def count_symbols_on_wing_image(
    job_root: Path,
    wing: dict[str, Any],
    legend: SymbolTableInfo,
    client: LlamaVisionClient,
) -> dict[str, Any]:
    """Run vision LLM on wing zoom tiles (preferred) or the full crop; NMS in ROI space."""
    image_rel = str(wing.get("image") or "")
    image_path = job_root / image_rel
    empty = {
        "column_key": wing["wing_name"],
        "wing_name": wing["wing_name"],
        "sheet_no": wing.get("sheet_no"),
        "page_key": wing.get("page_key"),
        "instance_dir_name": wing.get("instance_dir"),
        "wing_image": image_rel,
        "tiles_processed": 0,
        "raw_detection_count": 0,
        "post_nms_count": 0,
        "nms_removed": 0,
        "counts": {},
        "detections": [],
    }
    if not image_path.is_file():
        return empty

    zoom = _load_zoom_manifest(job_root, wing)
    raw_detections: list[SymbolDetection] = []
    tile_jobs: list[tuple[str, Path, dict[str, int] | None]] = []

    if zoom is not None:
        manifest, zooms_dir = zoom
        for tile in manifest.get("tiles") or []:
            if not isinstance(tile, dict):
                continue
            file_name = str(tile.get("file") or "")
            bbox = tile.get("bbox_roi")
            if not file_name or not isinstance(bbox, dict):
                continue
            tile_path = zooms_dir / file_name
            if tile_path.is_file():
                tile_jobs.append((file_name, tile_path, bbox))
    else:
        tile_jobs.append((Path(image_rel).name, image_path, None))

    roi_w, roi_h = _roi_dimensions(zoom[0] if zoom else None, image_path)

    for tile_file, tile_path, bbox_roi in tile_jobs:
        with Image.open(tile_path) as img:
            image = downscale_for_llama(img.convert("RGB"))
        local = _detections_from_llm(
            client, image, legend, tile_file=tile_file
        )
        for det in local:
            if bbox_roi is None:
                roi_box = translate_full_image_norm_box_to_roi(
                    [
                        det.box.x1 / max(1, image.width),
                        det.box.y1 / max(1, image.height),
                        det.box.x2 / max(1, image.width),
                        det.box.y2 / max(1, image.height),
                    ],
                    roi_width=roi_w,
                    roi_height=roi_h,
                )
            else:
                roi_box = translate_norm_box_to_roi(
                    [
                        det.box.x1 / max(1, image.width),
                        det.box.y1 / max(1, image.height),
                        det.box.x2 / max(1, image.width),
                        det.box.y2 / max(1, image.height),
                    ],
                    tile_bbox_roi=bbox_roi,
                    image_width=image.width,
                    image_height=image.height,
                )
            raw_detections.append(
                SymbolDetection(
                    symbol=det.symbol,
                    box=roi_box,
                    score=det.score,
                    tile_file=tile_file,
                )
            )

    pre_nms = len(raw_detections)
    kept = nms_by_class(raw_detections, config.SYMBOL_COUNT_NMS_IOU)
    counts: dict[str, int] = {}
    for det in kept:
        counts[det.symbol] = counts.get(det.symbol, 0) + 1

    return {
        **empty,
        "tiles_processed": len(tile_jobs),
        "raw_detection_count": pre_nms,
        "post_nms_count": len(kept),
        "nms_removed": pre_nms - len(kept),
        "counts": counts,
        "detections": [
            {
                "symbol": det.symbol,
                "score": det.score,
                "tile_file": det.tile_file,
                "box_roi": {
                    "x1": det.box.x1,
                    "y1": det.box.y1,
                    "x2": det.box.x2,
                    "y2": det.box.y2,
                },
            }
            for det in kept
        ],
    }


def _legacy_roi_dimensions(
    manifest: dict[str, Any], full_wing_path: Path | None
) -> tuple[int, int]:
    roi_box = manifest.get("roi_box_px") or {}
    if all(k in roi_box for k in ("x1", "y1", "x2", "y2")):
        return (
            max(1, int(roi_box["x2"]) - int(roi_box["x1"])),
            max(1, int(roi_box["y2"]) - int(roi_box["y1"])),
        )
    if full_wing_path and full_wing_path.is_file():
        with Image.open(full_wing_path) as img:
            return img.size
    tile_size = int(manifest.get("tile_size") or DEFAULT_TILE_SIZE)
    return tile_size, tile_size


def count_symbols_for_instance(
    instance: WingInstance,
    legend: SymbolTableInfo,
    client: LlamaVisionClient,
) -> dict[str, Any]:
    """Legacy zoom-tile path kept for unit tests."""
    manifest = instance.manifest
    roi_w, roi_h = _legacy_roi_dimensions(manifest, instance.full_wing_path)
    raw_detections: list[SymbolDetection] = []
    tile_jobs: list[tuple[str, Path, dict[str, int] | None]] = []

    full_wing_path = instance.full_wing_path
    if full_wing_path and full_wing_path.is_file():
        tile_size = int(manifest.get("tile_size") or DEFAULT_TILE_SIZE)
        threshold = config.SYMBOL_COUNT_USE_FULL_WING_THRESHOLD
        if roi_w <= tile_size * threshold and roi_h <= tile_size * threshold:
            tile_jobs.append(("full_wing.jpg", full_wing_path, None))
    if not tile_jobs:
        for tile in manifest.get("tiles") or []:
            if not isinstance(tile, dict):
                continue
            file_name = str(tile.get("file") or "")
            bbox = tile.get("bbox_roi")
            if not file_name or not isinstance(bbox, dict):
                continue
            tile_path = instance.zooms_dir / file_name
            if tile_path.is_file():
                tile_jobs.append((file_name, tile_path, bbox))

    for tile_file, tile_path, bbox_roi in tile_jobs:
        with Image.open(tile_path) as img:
            image = downscale_for_llama(img.convert("RGB"))
        local = _detections_from_llm(client, image, legend, tile_file=tile_file)
        for det in local:
            if bbox_roi is None:
                roi_box = translate_full_image_norm_box_to_roi(
                    [
                        det.box.x1 / max(1, image.width),
                        det.box.y1 / max(1, image.height),
                        det.box.x2 / max(1, image.width),
                        det.box.y2 / max(1, image.height),
                    ],
                    roi_width=roi_w,
                    roi_height=roi_h,
                )
            else:
                roi_box = translate_norm_box_to_roi(
                    [
                        det.box.x1 / max(1, image.width),
                        det.box.y1 / max(1, image.height),
                        det.box.x2 / max(1, image.width),
                        det.box.y2 / max(1, image.height),
                    ],
                    tile_bbox_roi=bbox_roi,
                    image_width=image.width,
                    image_height=image.height,
                )
            raw_detections.append(
                SymbolDetection(
                    symbol=det.symbol,
                    box=roi_box,
                    score=det.score,
                    tile_file=tile_file,
                )
            )

    pre_nms = len(raw_detections)
    kept = nms_by_class(raw_detections, config.SYMBOL_COUNT_NMS_IOU)
    counts: dict[str, int] = {}
    for det in kept:
        counts[det.symbol] = counts.get(det.symbol, 0) + 1

    return {
        "column_key": instance.column_key,
        "wing_name": instance.wing_name,
        "page_key": instance.page_key,
        "instance_dir_name": instance.instance_dir_name,
        "tiles_processed": len(tile_jobs),
        "raw_detection_count": pre_nms,
        "post_nms_count": len(kept),
        "nms_removed": pre_nms - len(kept),
        "counts": counts,
        "detections": [
            {
                "symbol": det.symbol,
                "score": det.score,
                "tile_file": det.tile_file,
                "box_roi": {
                    "x1": det.box.x1,
                    "y1": det.box.y1,
                    "x2": det.box.x2,
                    "y2": det.box.y2,
                },
            }
            for det in kept
        ],
    }


def run_symbol_count(
    job_root: Path,
    *,
    client: LlamaVisionClient | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Count symbols on T-200 floor-plan wing crops using the vision LLM."""
    job_root = Path(job_root)
    symbol_path = job_root / "symbol_table.json"
    legend = load_symbol_table_json(symbol_path)
    if legend is None or not legend.entries:
        return {
            "status": "skipped",
            "reason": "symbol_table.json missing or empty",
            "instances": 0,
        }

    wings = _load_wing_regions(job_root)
    if not wings:
        return {
            "status": "skipped",
            "reason": "no T-200 floor-plan wings with cv_box_on_page",
            "instances": 0,
        }

    llm = client or get_llama_client()
    if not llm.ready:
        raise RuntimeError(llm._load_error or "Vision LLM is unavailable")

    out_dir = job_root / config.SYMBOL_COUNT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def _progress(message: str) -> None:
        if on_progress:
            on_progress("counting_symbols", message)

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for wing in wings:
        by_page[wing["page_idx"]].append(wing)

    wing_jobs: list[dict[str, Any]] = []
    for _page_idx, page_wings in sorted(by_page.items()):
        wing_jobs.extend(select_wings_for_counting(page_wings))

    per_sheet: list[dict[str, Any]] = []
    workers = max(1, config.SYMBOL_COUNT_LLM_WORKERS)
    total = len(wing_jobs)

    if workers == 1:
        for index, wing in enumerate(wing_jobs, start=1):
            image_name = Path(str(wing.get("image") or "")).name
            _progress(
                f"Vision count {wing['wing_name']} ({image_name}) "
                f"({index}/{total})"
            )
            per_sheet.append(count_symbols_on_wing_image(job_root, wing, legend, llm))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    count_symbols_on_wing_image, job_root, wing, legend, llm
                ): wing
                for wing in wing_jobs
            }
            done = 0
            for future in as_completed(futures):
                wing = futures[future]
                done += 1
                image_name = Path(str(wing.get("image") or "")).name
                _progress(
                    f"Vision count {wing['wing_name']} ({image_name}) "
                    f"({done}/{total})"
                )
                per_sheet.append(future.result())

    per_sheet.sort(key=lambda row: (str(row.get("page_key")), str(row.get("wing_name"))))
    instance_results = aggregate_results_by_wing_name(per_sheet)

    detail = {
        "status": "done",
        "method": "wing_zoom_tiles_vision_by_wing_name",
        "instances": len(instance_results),
        "sheets_scanned": len(per_sheet),
        "legend_entry_count": legend.entry_count,
        "sheets": "T-200–T-299",
        "nms_iou": config.SYMBOL_COUNT_NMS_IOU,
        "min_confidence": config.SYMBOL_COUNT_MIN_CONFIDENCE,
        "instance_results": instance_results,
        "per_sheet_results": per_sheet,
    }
    detail_path = out_dir / "symbol_count_detail.json"
    detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    report_paths = write_symbol_count_reports(
        legend=legend,
        instance_results=instance_results,
        out_dir=out_dir,
        source_filename=job_root.name,
        project_title=None,
    )

    return {
        "status": "done",
        "instances": len(instance_results),
        "symbol_count_detail_path": f"{config.SYMBOL_COUNT_DIR}/symbol_count_detail.json",
        "symbol_count_report_csv": report_paths["csv"],
        "symbol_count_report_pdf": report_paths.get("pdf"),
        "symbol_count_report_json": report_paths.get("json"),
    }
