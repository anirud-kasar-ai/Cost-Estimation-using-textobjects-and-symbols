"""Per-wing ROI + overlapping zoom tiling.

Goal: given an already-cropped wing image, create:
  - an ink-tight ROI crop (rois/roi.png)
  - overlapping zoom tiles covering the ROI (zooms/plan_zoom_r##_c##.jpg)
  - a manifest (zooms/zooms_manifest.json)
  - an overlay preview showing ROI + tile coverage (rois/roi_overlay.png)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from pipeline.diagram_crop import _content_bbox, ink_ratio, plan_mask


DEFAULT_TILE_SIZE = 653
DEFAULT_OVERLAP_PCT = 0.10
# If a tile is nearly blank, skip it to avoid noisy crops.
DEFAULT_MIN_TILE_INK = 0.002


@dataclass(frozen=True)
class TileInfo:
    r: int
    c: int
    bbox_roi: tuple[int, int, int, int]  # (x1,y1,x2,y2) in ROI pixel coords
    file: str
    ink_ratio: float


def _safe_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip().replace(" ", "_"))
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned[:80] or "roi"


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    outline: tuple[int, int, int] = (220, 55, 55),
    width: int = 2,
    dash: int = 10,
    gap: int = 6,
) -> None:
    """Draw a dashed rectangle via segments (PIL has no native dashed rectangle)."""
    x1, y1, x2, y2 = xy
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in segments:
        length = max(abs(bx - ax), abs(by - ay), 1)
        steps = max(1, length // max(dash + gap, 1))
        for i in range(steps + 1):
            t0 = i * (dash + gap) / length
            t1 = min(1.0, (i * (dash + gap) + dash) / length)
            if t0 >= 1.0:
                break
            sx = int(ax + (bx - ax) * t0)
            sy = int(ay + (by - ay) * t0)
            ex = int(ax + (bx - ax) * t1)
            ey = int(ay + (by - ay) * t1)
            draw.line([(sx, sy), (ex, ey)], fill=outline, width=width)


def compute_wing_roi(wing_image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Ink-tight ROI within a wing crop.

    Uses the same plan linework mask as the diagram/wing split.
    """
    if wing_image.mode != "RGB":
        wing_image = wing_image.convert("RGB")
    mask = plan_mask(wing_image)
    inner = _content_bbox(mask, pad_pct=0.015)
    if inner is None:
        w, h = wing_image.size
        return wing_image, (0, 0, w, h)
    return wing_image.crop(inner.to_tuple()).convert("RGB"), inner.to_tuple()


def iter_overlapping_tiles(
    roi_w: int,
    roi_h: int,
    *,
    tile_size: int,
    overlap_pct: float,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """Create a deterministic overlapping grid over ROI."""
    tile_size = max(64, int(tile_size))
    overlap_pct = float(overlap_pct)
    overlap_pct = max(0.0, min(0.7, overlap_pct))
    step = int(round(tile_size * (1.0 - overlap_pct)))
    step = max(1, step)

    def positions(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        out: list[int] = []
        pos = 0
        while pos + tile_size <= total:
            out.append(pos)
            pos += step
        # ensure right/bottom edge coverage
        if out:
            last = out[-1]
            end = total - tile_size
            if end > last:
                out.append(end)
        return sorted(set(out))

    xs = positions(roi_w)
    ys = positions(roi_h)

    tiles: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for ri, y in enumerate(ys):
        for ci, x in enumerate(xs):
            bbox = (x, y, min(roi_w, x + tile_size), min(roi_h, y + tile_size))
            tiles.append((ri, ci, bbox))
    return tiles


def export_wing_roi_and_zooms(
    *,
    wing_image: Image.Image,
    wing_folder: Path,
    wing_rel_prefix: str,
    instance_dir_name: str,
    wing_name: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap_pct: float = DEFAULT_OVERLAP_PCT,
    min_tile_ink: float = DEFAULT_MIN_TILE_INK,
    min_tile_side: int = 64,
) -> dict[str, Any]:
    """Write ROI + zoom tiles under a given wing folder.

    Returns a JSON-serializable summary for storing into metadata.
    """
    wing_folder = Path(wing_folder)
    rois_dir = wing_folder / "rois" / _safe_slug(instance_dir_name)
    zooms_dir = wing_folder / "zooms" / _safe_slug(instance_dir_name)
    rois_dir.mkdir(parents=True, exist_ok=True)
    zooms_dir.mkdir(parents=True, exist_ok=True)

    roi_image, roi_box = compute_wing_roi(wing_image)
    roi_w, roi_h = roi_image.size

    # 1) ROI exports
    roi_png_path = rois_dir / "roi.png"
    roi_image.save(roi_png_path, format="PNG", optimize=True)

    # 2) Zoom exports
    full_wing_path = zooms_dir / "full_wing.jpg"
    roi_image.save(full_wing_path, format="JPEG", quality=90, optimize=True)

    tiles = iter_overlapping_tiles(
        roi_w,
        roi_h,
        tile_size=tile_size,
        overlap_pct=overlap_pct,
    )
    tiles_kept: list[TileInfo] = []
    tiles_skipped_blank = 0

    # overlay image is on the ROI image, since tiles are in ROI coords.
    overlay = roi_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Stable outline color per wing.
    # (We keep it simple here; hvac-cost-estimator uses more nuanced palettes.)
    outline = (220, 55, 55)

    for r, c, bbox in tiles:
        x1, y1, x2, y2 = bbox
        tw, th = x2 - x1, y2 - y1
        if tw < min_tile_side or th < min_tile_side:
            continue
        tile = roi_image.crop(bbox)

        tile_ink = ink_ratio(tile)
        if tile_ink < min_tile_ink:
            tiles_skipped_blank += 1
            continue

        tile_name = f"plan_zoom_r{r:02d}_c{c:02d}.jpg"
        tile_path = zooms_dir / tile_name
        tile.save(tile_path, format="JPEG", quality=94, optimize=True)

        tiles_kept.append(
            TileInfo(
                r=r,
                c=c,
                bbox_roi=(x1, y1, x2, y2),
                file=str(tile_name),
                ink_ratio=tile_ink,
            )
        )
        _draw_dashed_rect(draw, (x1, y1, x2, y2), outline=outline, width=2)

    overlay_path = rois_dir / "roi_overlay.png"
    overlay.convert("RGB").save(overlay_path, format="PNG", optimize=True)

    manifest: dict[str, Any] = {
        "wing_name": wing_name,
        "tile_size": int(tile_size),
        "overlap_pct": float(overlap_pct),
        "count": len(tiles_kept),
        "skipped_blank": int(tiles_skipped_blank),
        "roi_box_px": {
            # ROI bbox in wing crop pixel coords, for audit.
            "x1": int(roi_box[0]),
            "y1": int(roi_box[1]),
            "x2": int(roi_box[2]),
            "y2": int(roi_box[3]),
        },
        "tiles": [
            {
                "r": t.r,
                "c": t.c,
                "bbox_roi": {
                    "x1": t.bbox_roi[0],
                    "y1": t.bbox_roi[1],
                    "x2": t.bbox_roi[2],
                    "y2": t.bbox_roi[3],
                },
                "file": t.file,
                "ink_ratio": float(t.ink_ratio),
            }
            for t in tiles_kept
        ],
    }

    manifest_path = zooms_dir / "zooms_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "roi_box_on_wing": {
            "x1": roi_box[0],
            "y1": roi_box[1],
            "x2": roi_box[2],
            "y2": roi_box[3],
        },
        # Paths are relative to the job root (so the FastAPI UI can serve them).
        "roi_image": f"{wing_rel_prefix}/rois/{_safe_slug(instance_dir_name)}/roi.png",
        "roi_overlay_image": f"{wing_rel_prefix}/rois/{_safe_slug(instance_dir_name)}/roi_overlay.png",
        "zooms_manifest": f"{wing_rel_prefix}/zooms/{_safe_slug(instance_dir_name)}/zooms_manifest.json",
        "zooms_full_image": f"{wing_rel_prefix}/zooms/{_safe_slug(instance_dir_name)}/full_wing.jpg",
        "zooms_manifest_payload": manifest,
    }

