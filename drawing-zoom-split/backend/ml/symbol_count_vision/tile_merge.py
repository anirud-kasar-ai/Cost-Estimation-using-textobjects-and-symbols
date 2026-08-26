"""Merge symbol detections across overlapping ROI zoom tiles.

Mirrors dual-pathway ``roi_zoom`` conventions:
  - tile size 653, overlap 10% (step 588) when inferring from filenames
  - class-wise NMS at IoU 0.5 in ROI pixel space (count once in overlap)
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
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
from pipeline.overlay import draw_detections_overlay
from pipeline.symbol_count import (
    _finalize_counts,
    detect_symbols_raw,
    filter_to_legend,
    legend_count_keys,
)
from pipeline.symbol_legend import parse_symbol_file

DEFAULT_TILE_SIZE = 653
DEFAULT_OVERLAP_PCT = 0.10
TILE_NAME_RE = re.compile(
    r"plan_zoom_r(\d+)_c(\d+)\.(jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class TileSpec:
    path: Path
    r: int
    c: int
    bbox_roi: dict[str, int]  # x1,y1,x2,y2 in ROI pixels


def tile_step(tile_size: int = DEFAULT_TILE_SIZE, overlap_pct: float = DEFAULT_OVERLAP_PCT) -> int:
    tile_size = max(64, int(tile_size))
    overlap_pct = max(0.0, min(0.7, float(overlap_pct)))
    return max(1, int(round(tile_size * (1.0 - overlap_pct))))


def translate_pixel_box_to_roi(
    box: BoundingBox,
    tile_bbox_roi: dict[str, int],
) -> BoundingBox:
    """Map a tile-local pixel box into ROI pixel coordinates."""
    tx1 = float(tile_bbox_roi["x1"])
    ty1 = float(tile_bbox_roi["y1"])
    return BoundingBox(
        x1=tx1 + box.x1,
        y1=ty1 + box.y1,
        x2=tx1 + box.x2,
        y2=ty1 + box.y2,
    )


def infer_tiles_from_files(
    tile_paths: list[Path],
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap_pct: float = DEFAULT_OVERLAP_PCT,
) -> list[TileSpec]:
    """Build ``bbox_roi`` from ``plan_zoom_r##_c##`` names (step = tile*(1-overlap))."""
    step = tile_step(tile_size, overlap_pct)
    out: list[TileSpec] = []
    for path in sorted(tile_paths):
        match = TILE_NAME_RE.search(path.name)
        if not match:
            continue
        r = int(match.group(1))
        c = int(match.group(2))
        x1 = c * step
        y1 = r * step
        out.append(
            TileSpec(
                path=path,
                r=r,
                c=c,
                bbox_roi={
                    "x1": x1,
                    "y1": y1,
                    "x2": x1 + tile_size,
                    "y2": y1 + tile_size,
                },
            )
        )
    return out


def load_tiles_from_manifest(
    manifest_path: Path,
    tiles_dir: Path,
) -> tuple[list[TileSpec], dict[str, Any]]:
    """Load tile specs from ``zooms_manifest.json`` (dual-pathway format)."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tile_size = int(data.get("tile_size") or DEFAULT_TILE_SIZE)
    overlap_pct = float(data.get("overlap_pct") if data.get("overlap_pct") is not None else DEFAULT_OVERLAP_PCT)
    specs: list[TileSpec] = []
    for entry in data.get("tiles") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("file") or "").strip()
        if not name:
            continue
        path = tiles_dir / name
        if not path.is_file():
            # Manifest may store just the basename; also try nested.
            alt = tiles_dir / Path(name).name
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue
        bbox = entry.get("bbox_roi") or {}
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            bbox_roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif isinstance(bbox, dict):
            bbox_roi = {
                "x1": int(bbox["x1"]),
                "y1": int(bbox["y1"]),
                "x2": int(bbox["x2"]),
                "y2": int(bbox["y2"]),
            }
        else:
            continue
        r = int(entry.get("r", 0))
        c = int(entry.get("c", 0))
        specs.append(TileSpec(path=path, r=r, c=c, bbox_roi=bbox_roi))
    meta = {
        "tile_size": tile_size,
        "overlap_pct": overlap_pct,
        "source": "manifest",
        "manifest": str(manifest_path),
    }
    return specs, meta


def resolve_tiles(
    tiles_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> tuple[list[TileSpec], dict[str, Any]]:
    """Prefer manifest; else infer from plan_zoom filenames in ``tiles_dir``."""
    tiles_dir = Path(tiles_dir)
    if manifest_path and manifest_path.is_file():
        specs, meta = load_tiles_from_manifest(manifest_path, tiles_dir)
        if specs:
            return specs, meta

    auto_manifest = tiles_dir / "zooms_manifest.json"
    if auto_manifest.is_file():
        specs, meta = load_tiles_from_manifest(auto_manifest, tiles_dir)
        if specs:
            return specs, meta

    image_files = [
        p
        for p in sorted(tiles_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and TILE_NAME_RE.search(p.name)
    ]
    if not image_files:
        # Fall back to any images in the folder (single-column grid by sort order).
        image_files = [
            p
            for p in sorted(tiles_dir.iterdir())
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and p.name.lower() not in {"full_wing.jpg", "full_wing.png", "overlay.png"}
        ]
        specs: list[TileSpec] = []
        step = tile_step()
        for i, path in enumerate(image_files):
            x1 = 0
            y1 = i * step
            specs.append(
                TileSpec(
                    path=path,
                    r=i,
                    c=0,
                    bbox_roi={
                        "x1": x1,
                        "y1": y1,
                        "x2": x1 + DEFAULT_TILE_SIZE,
                        "y2": y1 + DEFAULT_TILE_SIZE,
                    },
                )
            )
        return specs, {
            "tile_size": DEFAULT_TILE_SIZE,
            "overlap_pct": DEFAULT_OVERLAP_PCT,
            "source": "flat_list",
        }

    specs = infer_tiles_from_files(image_files)
    return specs, {
        "tile_size": DEFAULT_TILE_SIZE,
        "overlap_pct": DEFAULT_OVERLAP_PCT,
        "source": "inferred_grid",
    }


def build_roi_canvas(
    tiles: list[TileSpec],
    *,
    full_wing_path: Path | None = None,
) -> Image.Image:
    """ROI background for the merged overlay."""
    if full_wing_path and full_wing_path.is_file():
        return Image.open(full_wing_path).convert("RGB")

    if not tiles:
        return Image.new("RGB", (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), (255, 255, 255))

    roi_w = max(t.bbox_roi["x2"] for t in tiles)
    roi_h = max(t.bbox_roi["y2"] for t in tiles)
    canvas = Image.new("RGB", (max(1, roi_w), max(1, roi_h)), (255, 255, 255))
    for tile in tiles:
        try:
            img = Image.open(tile.path).convert("RGB")
        except OSError:
            continue
        canvas.paste(img, (tile.bbox_roi["x1"], tile.bbox_roi["y1"]))
    return canvas


def count_symbols_on_zoom_folder(
    *,
    legend: Any,
    glyph_dir: Path | None,
    tiles_dir: Path,
    symbol_file_suffix: str = ".pdf",
    manifest_path: Path | None = None,
    nms_iou: float | None = None,
) -> tuple[dict[str, Any], Image.Image]:
    """Detect on each zoom tile, merge in ROI space, return payload + ROI image."""
    nms_iou = float(nms_iou if nms_iou is not None else config.SYMBOL_COUNT_NMS_IOU)
    tiles, tile_meta = resolve_tiles(tiles_dir, manifest_path=manifest_path)
    if not tiles:
        raise RuntimeError("No zoom tile images found in the uploaded folder.")

    valid_keys = legend_count_keys(legend)
    pooled: list[SymbolDetection] = []
    notes: list[str] = [
        f"Folder merge: {len(tiles)} tile(s), source={tile_meta.get('source')}, "
        f"overlap={tile_meta.get('overlap_pct')}, NMS IoU={nms_iou}."
    ]
    meta: dict[str, Any] = {
        "linear_keys": set(),
        "hash_key": None,
        "camera_key": None,
        "glyphs_available": False,
    }
    raw_total = 0

    for tile in tiles:
        image = Image.open(tile.path).convert("RGB")
        dets, tile_notes, tile_meta_det = detect_symbols_raw(
            image=image,
            legend=legend,
            glyph_dir=glyph_dir,
            symbol_file_suffix=symbol_file_suffix,
            nms_iou=nms_iou,
            apply_local_nms=False,
        )
        raw_total += len(dets)
        for note in tile_notes:
            if note not in notes and not note.startswith("Linear/run"):
                # Keep linear note once from first tile that emits it.
                pass
        if tile_meta_det.get("linear_keys"):
            meta["linear_keys"] = set(tile_meta_det["linear_keys"]) | set(
                meta.get("linear_keys") or set()
            )
        meta["hash_key"] = meta.get("hash_key") or tile_meta_det.get("hash_key")
        meta["camera_key"] = meta.get("camera_key") or tile_meta_det.get("camera_key")
        meta["glyphs_available"] = meta.get("glyphs_available") or tile_meta_det.get(
            "glyphs_available"
        )
        for det in dets:
            pooled.append(
                SymbolDetection(
                    symbol=det.symbol,
                    box=translate_pixel_box_to_roi(det.box, tile.bbox_roi),
                    score=det.score,
                    source=det.source,
                    qty=max(1, int(det.qty)),
                )
            )

    if meta.get("linear_keys"):
        notes.append(
            "Linear/run items are not counted by glyph template matching. "
            "Part labels on the plan are still counted via text/OCR: "
            + ", ".join(sorted(meta["linear_keys"]))
        )

    pooled = filter_to_legend(pooled, valid_keys)
    kept = nms_by_class(pooled, iou_threshold=nms_iou)
    kept = suppress_cross_label(kept, iou_threshold=0.6, source="template")
    kept = suppress_nearby_same_label(kept, source="ocr", max_center_dist=80.0)
    kept = filter_to_legend(kept, valid_keys)

    full_wing = tiles_dir / "full_wing.jpg"
    if not full_wing.is_file():
        full_wing = tiles_dir / "full_wing.png"
    roi_image = build_roi_canvas(
        tiles,
        full_wing_path=full_wing if full_wing.is_file() else None,
    )

    payload = _finalize_counts(
        legend=legend,
        kept=kept,
        notes=notes,
        meta=meta,
        image=None,  # skip per-tile LLM; folder jobs use CV merge only
        method="folder_cv_merged",
        raw_detection_count=raw_total,
        nms_iou=nms_iou,
        extra={
            "tile_count": len(tiles),
            "tile_merge": tile_meta,
            "overlap_deduped": True,
        },
    )
    return payload, roi_image


def run_symbol_count_folder_job(
    symbol_file: Path,
    tiles_dir: Path,
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Parse legend, merge zoom-folder detections, write result + overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    legend, glyph_dir = parse_symbol_file(symbol_file, output_dir)
    payload, roi_image = count_symbols_on_zoom_folder(
        legend=legend,
        glyph_dir=glyph_dir,
        tiles_dir=tiles_dir,
        symbol_file_suffix=symbol_file.suffix.lower(),
        manifest_path=manifest_path,
    )
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_detections_overlay(
        image=roi_image,
        detections=payload.get("detections") or [],
        output_path=output_dir / "overlay.png",
    )
    try:
        roi_image.save(output_dir / "plan_image.png", format="PNG")
    except Exception:  # noqa: BLE001
        pass
    return payload


def collect_tile_uploads(
    files: list[tuple[str, bytes]],
    dest_dir: Path,
) -> Path | None:
    """Write uploaded zoom files into ``dest_dir``; return manifest path if any.

    ``files`` is a list of (filename, bytes). Nested paths from webkitdirectory
    are flattened to basenames (last path segment).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous tiles
    if dest_dir.exists():
        for child in dest_dir.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    manifest_path: Path | None = None
    for name, data in files:
        if not data:
            continue
        base = Path(name.replace("\\", "/")).name
        if not base:
            continue
        lower = base.lower()
        out = dest_dir / base
        out.write_bytes(data)
        if lower == "zooms_manifest.json":
            manifest_path = out
    return manifest_path


__all__ = [
    "DEFAULT_OVERLAP_PCT",
    "DEFAULT_TILE_SIZE",
    "TileSpec",
    "build_roi_canvas",
    "collect_tile_uploads",
    "count_symbols_on_zoom_folder",
    "infer_tiles_from_files",
    "load_tiles_from_manifest",
    "resolve_tiles",
    "run_symbol_count_folder_job",
    "tile_step",
    "translate_pixel_box_to_roi",
]
