"""Map tile-local detections into a shared ROI so overlapping zooms count once."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline import config
from pipeline.nms import nms_yolo_hits
from pipeline.yolo_detect import YoloHit

logger = logging.getLogger(__name__)

_ZOOM_RE = re.compile(r"plan_zoom_r(\d+)_c(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


@dataclass(frozen=True)
class TileRoi:
    file: str
    r: int
    c: int
    x1: float
    y1: float
    x2: float
    y2: float


def _basename(name: str) -> str:
    return Path(name.replace("\\", "/")).name


def load_tile_rois(
    folder: Path,
    *,
    default_overlap_pct: float = 0.10,
) -> dict[str, TileRoi] | None:
    """Return filename → ROI crop from zooms_manifest.json or r/c filenames."""
    manifest = folder / "zooms_manifest.json"
    if not manifest.is_file():
        nested = list(folder.rglob("zooms_manifest.json"))
        manifest = nested[0] if nested else manifest
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            tiles = data.get("tiles")
            if isinstance(tiles, list) and tiles:
                out: dict[str, TileRoi] = {}
                for row in tiles:
                    if not isinstance(row, dict):
                        continue
                    bbox = row.get("bbox_roi") or {}
                    file = _basename(str(row.get("file") or ""))
                    if not file or not isinstance(bbox, dict):
                        continue
                    try:
                        out[file.lower()] = TileRoi(
                            file=file,
                            r=int(row.get("r") or 0),
                            c=int(row.get("c") or 0),
                            x1=float(bbox["x1"]),
                            y1=float(bbox["y1"]),
                            x2=float(bbox["x2"]),
                            y2=float(bbox["y2"]),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                if out:
                    return out

    # Infer dual-pathway grid: plan_zoom_r##_c## + 10% overlap
    inferred: dict[str, TileRoi] = {}
    sizes: list[tuple[int, int]] = []
    for path in sorted(folder.rglob("*")):
        m = _ZOOM_RE.search(path.name)
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        try:
            from PIL import Image

            with Image.open(path) as im:
                w, h = im.size
        except OSError:
            continue
        sizes.append((w, h))
        inferred[path.name.lower()] = TileRoi(
            file=path.name, r=r, c=c, x1=0, y1=0, x2=float(w), y2=float(h)
        )
    if len(inferred) < 2:
        return None

    # Typical zoom tiles are square-ish; use median width as tile size
    widths = sorted(s[0] for s in sizes)
    tile_w = widths[len(widths) // 2]
    overlap = max(0.0, min(0.4, float(default_overlap_pct)))
    step = max(1, int(round(tile_w * (1.0 - overlap))))
    placed: dict[str, TileRoi] = {}
    for key, t in inferred.items():
        tw = t.x2 - t.x1
        th = t.y2 - t.y1
        x1 = float(t.c * step)
        y1 = float(t.r * step)
        placed[key] = TileRoi(
            file=t.file,
            r=t.r,
            c=t.c,
            x1=x1,
            y1=y1,
            x2=x1 + tw,
            y2=y1 + th,
        )
    logger.info(
        "Inferred %d overlapping zoom tiles (step=%d, overlap=%.0f%%)",
        len(placed),
        step,
        overlap * 100,
    )
    return placed


def map_hit_to_roi(hit: YoloHit, tile: TileRoi) -> YoloHit:
    return YoloHit(
        class_id=hit.class_id,
        class_name=hit.class_name,
        confidence=hit.confidence,
        x1=tile.x1 + hit.x1,
        y1=tile.y1 + hit.y1,
        x2=tile.x1 + hit.x2,
        y2=tile.y1 + hit.y2,
        image_name=hit.image_name,
        legend_number=hit.legend_number,
        count_weight=hit.count_weight,
        rescued=hit.rescued,
    )


def _rescue_consensus_filter(
    mapped: list[YoloHit],
    tiles: list[TileRoi],
) -> tuple[list[YoloHit], int]:
    """Two-tile consensus for rescue-band hits (all boxes in ROI space).

    A rescued (low-confidence) hit whose center is covered by 2+ zoom tiles
    must be corroborated by a same-class hit from a *different* source tile at
    roughly the same spot. Real symbols re-detect across the 25% overlap;
    one-off smudges and hatching noise don't. Hits in single-tile territory
    (no overlap available) are kept — consensus is impossible there.
    """
    if not mapped or not tiles:
        return mapped, 0

    def covering_tiles(cx: float, cy: float) -> int:
        return sum(1 for t in tiles if t.x1 <= cx <= t.x2 and t.y1 <= cy <= t.y2)

    kept: list[YoloHit] = []
    dropped = 0
    for h in mapped:
        if not h.rescued:
            kept.append(h)
            continue
        cx, cy = (h.x1 + h.x2) / 2.0, (h.y1 + h.y2) / 2.0
        if covering_tiles(cx, cy) < 2:
            kept.append(h)  # only one tile sees this spot; nothing to agree with
            continue
        supported = False
        for o in mapped:
            if o is h or o.class_id != h.class_id:
                continue
            if _basename(o.image_name or "").lower() == _basename(h.image_name or "").lower():
                continue  # support must come from a different tile
            ocx, ocy = (o.x1 + o.x2) / 2.0, (o.y1 + o.y2) / 2.0
            reach = 0.75 * max(h.x2 - h.x1, h.y2 - h.y1, 8.0)
            if abs(ocx - cx) <= reach and abs(ocy - cy) <= reach:
                supported = True
                break
        if supported:
            kept.append(h)
        else:
            dropped += 1
    return kept, dropped


def merge_overlapping_tile_hits(
    hits: list[YoloHit],
    rois: dict[str, TileRoi],
    *,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> list[YoloHit]:
    """Translate tile-local boxes into ROI space and de-dup real overlaps.

    Every detection survives unless it geometrically overlaps a
    higher-confidence detection of the same symbol in shared ROI space.

    (The old "exclusive region" rule deleted every hit whose center fell in
    a neighbouring tile's territory, assuming the neighbour re-detected the
    same symbol at its own tile edge — which YOLO often doesn't, so real,
    confidently-detected symbols were silently erased.)
    """
    if not hits or not rois:
        return hits

    image_sizes = image_sizes or {}
    mapped: list[YoloHit] = []

    for hit in hits:
        key = _basename(hit.image_name or "").lower()
        tile = rois.get(key)
        if tile is None:
            mapped.append(hit)
            continue

        size = image_sizes.get(key)
        sx = sy = 1.0
        if size:
            iw, ih = float(size[0]), float(size[1])
            tw = max(1.0, tile.x2 - tile.x1)
            th = max(1.0, tile.y2 - tile.y1)
            if iw > 0:
                sx = tw / iw
            if ih > 0:
                sy = th / ih
        roi_hit = YoloHit(
            class_id=hit.class_id,
            class_name=hit.class_name,
            confidence=hit.confidence,
            x1=tile.x1 + hit.x1 * sx,
            y1=tile.y1 + hit.y1 * sy,
            x2=tile.x1 + hit.x2 * sx,
            y2=tile.y1 + hit.y2 * sy,
            image_name=hit.image_name,
            legend_number=hit.legend_number,
            count_weight=hit.count_weight,
            rescued=hit.rescued,
        )
        mapped.append(roi_hit)

    consensus_dropped = 0
    if config.YOLO_RESCUE_CONSENSUS:
        mapped, consensus_dropped = _rescue_consensus_filter(mapped, list(rois.values()))

    merged = nms_yolo_hits(
        mapped,
        same_class_iou=config.YOLO_NMS_IOU,
        cross_class_iou=config.YOLO_CROSS_NMS_IOU,
        compact_center_px=config.YOLO_NMS_CENTER_PX,
    )
    logger.info(
        "Tile overlap merge: kept %d / %d YOLO hits (%d cross-tile duplicates, "
        "%d unconfirmed rescues)",
        len(merged),
        len(hits),
        len(hits) - consensus_dropped - len(merged),
        consensus_dropped,
    )
    return merged
