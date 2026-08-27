"""Merge symbol detections across overlapping ROI zoom tiles.

Mirrors dual-pathway ``roi_zoom.iter_overlapping_tiles``:
  - tile size 653 (dual-pathway default), overlap 10% (step ~588)
  - last row/column snap to the ROI edge (extra overlap, still full tile crops)
  - class-wise NMS in ROI pixel space (count each physical mark once)

Order is always: detect each zoom tile → map to ROI → merge → count.
Never detect on a stitched mosaic.
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
    suppress_cross_source_duplicates,
    suppress_nearby_same_label,
)
from pipeline.merge_evaluator import evaluate_merged_detections
from pipeline.overlay import draw_detections_on_image, draw_detections_overlay
from pipeline.symbol_count import (
    _finalize_counts,
    detect_symbols_raw,
    filter_to_legend,
    legend_count_keys,
)
from pipeline.symbol_legend import parse_symbol_file

# Match dual-pathway ``ROI_ZOOM_TILE_SIZE`` / ``ROI_ZOOM_OVERLAP_PCT`` (653 @ 10%).
DEFAULT_TILE_SIZE = 653
DEFAULT_OVERLAP_PCT = 0.10
TILE_NAME_RE = re.compile(
    r"plan_zoom_r(\d+)_c(\d+)\.(jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _median(values: list[int]) -> int:
    if not values:
        return 0
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return int(vals[mid])
    return int((vals[mid - 1] + vals[mid]) / 2)


def _index_tiles_by_rc(image_files: list[Path]) -> dict[tuple[int, int], Path]:
    by_rc: dict[tuple[int, int], Path] = {}
    for path in image_files:
        match = TILE_NAME_RE.search(path.name)
        if not match:
            continue
        by_rc[(int(match.group(1)), int(match.group(2)))] = path
    return by_rc


def _load_gray(path: Path):
    import numpy as np

    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("L"))
    except OSError:
        return None


def _match_origin_offset(
    prev,
    curr,
    *,
    axis: str,
    max_offset: int,
    templ_span: int,
) -> tuple[int, float]:
    """Locate ``curr``'s leading strip inside ``prev``.

    Dual-pathway adjacent tiles share at least ``tile * overlap_pct`` pixels.
    Last row/col share *more* because the crop origin snaps to ``roi - tile``.
    The returned offset is ``curr_origin - prev_origin`` in that axis.
    """
    import cv2

    if prev is None or curr is None or prev.size == 0 or curr.size == 0:
        return max_offset, 0.0
    max_offset = max(1, int(max_offset))
    templ_span = max(16, int(templ_span))
    ph, pw = int(prev.shape[0]), int(prev.shape[1])
    ch, cw = int(curr.shape[0]), int(curr.shape[1])
    if axis == "x":
        tw = min(templ_span, cw, pw - 1)
        th = min(ph, ch)
        if tw < 16 or th < 16:
            return max_offset, 0.0
        hay = prev[:th, :]
        templ = curr[:th, :tw]
        if hay.shape[1] <= tw or hay.shape[0] < th:
            return max_offset, 0.0
        result = cv2.matchTemplate(hay, templ, cv2.TM_CCOEFF_NORMED)
        limit = min(max_offset, result.shape[1] - 1)
        region = result[:, : limit + 1]
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(region)
        return int(max_loc[0]), float(max_val)
    th = min(templ_span, ch, ph - 1)
    tw = min(pw, cw)
    if th < 16 or tw < 16:
        return max_offset, 0.0
    hay = prev[:, :tw]
    templ = curr[:th, :tw]
    if hay.shape[0] <= th or hay.shape[1] < tw:
        return max_offset, 0.0
    result = cv2.matchTemplate(hay, templ, cv2.TM_CCOEFF_NORMED)
    limit = min(max_offset, result.shape[0] - 1)
    region = result[: limit + 1, :]
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(region)
    return int(max_loc[1]), float(max_val)


def _collect_pair_offsets(
    by_rc: dict[tuple[int, int], Path],
    *,
    axis: str,
    max_a: int,
    max_b: int,
    max_offset: int,
    templ_span: int,
    last_only: bool,
) -> list[tuple[int, float]]:
    """Measure origin offsets between consecutive tiles along one axis.

    ``axis='x'`` walks columns (a=row, b=col). ``axis='y'`` walks rows
    (a=col, b=row). ``last_only`` keeps just the last-edge pair (the snapped one).
    """
    gray_cache: dict[Path, object] = {}
    out: list[tuple[int, float]] = []

    def gray(path: Path):
        cached = gray_cache.get(path)
        if cached is None and path not in gray_cache:
            cached = _load_gray(path)
            gray_cache[path] = cached
        return cached

    if axis == "x":
        b_values = [max_b - 1] if last_only else list(range(max_b))
        for b in b_values:
            if b < 0:
                continue
            for a in range(max_a + 1):
                prev_p = by_rc.get((a, b))
                curr_p = by_rc.get((a, b + 1))
                if prev_p is None or curr_p is None:
                    continue
                dx, score = _match_origin_offset(
                    gray(prev_p),
                    gray(curr_p),
                    axis="x",
                    max_offset=max_offset,
                    templ_span=templ_span,
                )
                out.append((dx, score))
        return out
    b_values = [max_b - 1] if last_only else list(range(max_b))
    for b in b_values:
        if b < 0:
            continue
        for a in range(max_a + 1):
            prev_p = by_rc.get((b, a))
            curr_p = by_rc.get((b + 1, a))
            if prev_p is None or curr_p is None:
                continue
            dy, score = _match_origin_offset(
                gray(prev_p),
                gray(curr_p),
                axis="y",
                max_offset=max_offset,
                templ_span=templ_span,
            )
            out.append((dy, score))
    return out


def _best_offset(pairs: list[tuple[int, float]], *, fallback: int, min_score: float = 0.55) -> tuple[int, float]:
    good = [(off, score) for off, score in pairs if score >= min_score]
    if not good:
        return int(fallback), 0.0
    good.sort(key=lambda item: item[1], reverse=True)
    top = good[: max(1, min(5, len(good)))]
    offset = _median([off for off, _score in top])
    score = max(s for _off, s in top)
    return int(offset), float(score)


def estimate_roi_size_from_overlap(
    image_files: list[Path],
    *,
    tile_size: int,
    overlap_pct: float,
) -> tuple[tuple[int, int], dict[str, float | int]] | None:
    """Recover dual-pathway ROI size from overlapping tile *content*.

    Dual-pathway last row/col are shifted so the crop still fills ``tile_size``
    (no blank padding). Matching the shared overlap strip recovers that snap
    when ``zooms_manifest.json`` / ``full_wing.jpg`` were not uploaded.
    """
    by_rc = _index_tiles_by_rc(image_files)
    if not by_rc:
        return None
    rows = [r for r, _c in by_rc]
    cols = [c for _r, c in by_rc]
    max_r = max(rows)
    max_c = max(cols)
    default_step = tile_step(tile_size, overlap_pct)
    templ_span = max(64, int(round(tile_size * overlap_pct)))

    # Interior pairs are everything except the last column / last row.
    interior_x_only = _collect_pair_offsets(
        by_rc,
        axis="x",
        max_a=max_r,
        max_b=max(0, max_c - 1),
        max_offset=default_step,
        templ_span=templ_span,
        last_only=False,
    )
    step_x, step_x_score = _best_offset(interior_x_only, fallback=default_step)
    if step_x < int(tile_size * 0.2):
        step_x = default_step
        step_x_score = 0.0
    last_dx, last_dx_score = _best_offset(
        _collect_pair_offsets(
            by_rc,
            axis="x",
            max_a=max_r,
            max_b=max_c,
            max_offset=step_x,
            templ_span=templ_span,
            last_only=True,
        ),
        fallback=step_x,
    )

    interior_y_only = _collect_pair_offsets(
        by_rc,
        axis="y",
        max_a=max_c,
        max_b=max(0, max_r - 1),
        max_offset=default_step,
        templ_span=templ_span,
        last_only=False,
    )
    step_y, step_y_score = _best_offset(interior_y_only, fallback=default_step)
    if step_y < int(tile_size * 0.2):
        step_y = default_step
        step_y_score = 0.0
    last_dy, last_dy_score = _best_offset(
        _collect_pair_offsets(
            by_rc,
            axis="y",
            max_a=max_c,
            max_b=max_r,
            max_offset=step_y,
            templ_span=templ_span,
            last_only=True,
        ),
        fallback=step_y,
    )

    last_path = by_rc.get((max_r, max_c)) or next(iter(by_rc.values()))
    last_w = last_h = int(tile_size)
    try:
        with Image.open(last_path) as img:
            last_w, last_h = img.size
    except OSError:
        pass

    last_x1 = max_c * step_x if max_c == 0 else (max_c - 1) * step_x + last_dx
    last_y1 = max_r * step_y if max_r == 0 else (max_r - 1) * step_y + last_dy
    roi_w = int(last_x1 + last_w)
    roi_h = int(last_y1 + last_h)
    roi_w = max(tile_size, roi_w)
    roi_h = max(tile_size, roi_h)
    meta = {
        "step_x": int(step_x),
        "step_y": int(step_y),
        "last_x1": int(last_x1),
        "last_y1": int(last_y1),
        "last_dx": int(last_dx),
        "last_dy": int(last_dy),
        "last_dx_score": round(float(last_dx_score), 4),
        "last_dy_score": round(float(last_dy_score), 4),
        "step_x_score": round(float(step_x_score), 4),
        "step_y_score": round(float(step_y_score), 4),
    }
    return (roi_w, roi_h), meta


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
    *,
    image_size: tuple[int, int] | None = None,
) -> BoundingBox:
    """Map a tile-local pixel box into ROI pixel coordinates."""
    tx1 = float(tile_bbox_roi["x1"])
    ty1 = float(tile_bbox_roi["y1"])
    sx = sy = 1.0
    if image_size:
        iw, ih = float(image_size[0]), float(image_size[1])
        tw = max(1.0, float(tile_bbox_roi["x2"]) - tx1)
        th = max(1.0, float(tile_bbox_roi["y2"]) - ty1)
        if iw > 0:
            sx = tw / iw
        if ih > 0:
            sy = th / ih
    return BoundingBox(
        x1=tx1 + box.x1 * sx,
        y1=ty1 + box.y1 * sy,
        x2=tx1 + box.x2 * sx,
        y2=ty1 + box.y2 * sy,
    )


def iter_overlapping_tiles(
    roi_w: int,
    roi_h: int,
    *,
    tile_size: int,
    overlap_pct: float,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """Same grid as dual-pathway ``roi_zoom.iter_overlapping_tiles`` (edge snap)."""
    tile_size = max(64, int(tile_size))
    overlap_pct = max(0.0, min(0.7, float(overlap_pct)))
    step = tile_step(tile_size, overlap_pct)

    def positions(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        out: list[int] = []
        pos = 0
        while pos + tile_size <= total:
            out.append(pos)
            pos += step
        if out:
            end = total - tile_size
            if end > out[-1]:
                out.append(end)
        return sorted(set(out))

    tiles: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for ri, y in enumerate(positions(roi_h)):
        for ci, x in enumerate(positions(roi_w)):
            bbox = (x, y, min(roi_w, x + tile_size), min(roi_h, y + tile_size))
            tiles.append((ri, ci, bbox))
    return tiles


def _measure_tile_size(tile_paths: list[Path], fallback: int = DEFAULT_TILE_SIZE) -> int:
    for path in tile_paths:
        try:
            with Image.open(path) as img:
                return max(int(img.size[0]), int(img.size[1]), 64)
        except OSError:
            continue
    return fallback


def infer_tiles_from_files(
    tile_paths: list[Path],
    *,
    tile_size: int | None = None,
    overlap_pct: float = DEFAULT_OVERLAP_PCT,
    roi_size: tuple[int, int] | None = None,
) -> list[TileSpec]:
    """Build ``bbox_roi`` from ``plan_zoom_r##_c##`` using dual-pathway step/overlap.

    Uses the actual tile pixel size (these zooms are 768, not 1152). When
    ``roi_size`` is known (``full_wing.jpg``), last row/column are snapped to
    the ROI edge exactly as dual-pathway does.
    """
    paths = [p for p in tile_paths if TILE_NAME_RE.search(p.name)]
    if not paths:
        return []
    tile_size = int(tile_size or _measure_tile_size(paths))
    overlap_pct = float(overlap_pct)
    step = tile_step(tile_size, overlap_pct)

    by_rc: dict[tuple[int, int], Path] = {}
    for path in sorted(paths):
        match = TILE_NAME_RE.search(path.name)
        if not match:
            continue
        by_rc[(int(match.group(1)), int(match.group(2)))] = path

    out: list[TileSpec] = []
    if roi_size is not None:
        roi_w, roi_h = int(roi_size[0]), int(roi_size[1])
        expected = {
            (r, c): bbox
            for r, c, bbox in iter_overlapping_tiles(
                roi_w, roi_h, tile_size=tile_size, overlap_pct=overlap_pct
            )
        }
        for (r, c), path in sorted(by_rc.items()):
            bbox = expected.get((r, c))
            if bbox is None:
                x1, y1 = c * step, r * step
                bbox = (x1, y1, x1 + tile_size, y1 + tile_size)
            x1, y1, x2, y2 = bbox
            out.append(
                TileSpec(path=path, r=r, c=c, bbox_roi={"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            )
        return out

    for (r, c), path in sorted(by_rc.items()):
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


def infer_tiles_from_snap(
    tile_paths: list[Path],
    *,
    snap_meta: dict[str, float | int],
    roi_size: tuple[int, int],
    tile_size: int,
) -> list[TileSpec]:
    """Build ``bbox_roi`` from recovered snap offsets (step_x/step_y/last_dx/last_dy)."""
    by_rc: dict[tuple[int, int], Path] = {}
    for path in tile_paths:
        match = TILE_NAME_RE.search(path.name)
        if not match:
            continue
        by_rc[(int(match.group(1)), int(match.group(2)))] = path
    if not by_rc:
        return []

    roi_w, roi_h = int(roi_size[0]), int(roi_size[1])
    tile_size = max(64, int(tile_size))
    step_x = max(1, int(snap_meta.get("step_x") or tile_step(tile_size)))
    step_y = max(1, int(snap_meta.get("step_y") or step_x))
    last_dx = max(1, int(snap_meta.get("last_dx") or step_x))
    last_dy = max(1, int(snap_meta.get("last_dy") or step_y))
    max_r = max(r for r, _c in by_rc)
    max_c = max(c for _r, c in by_rc)

    def x1_for_c(c: int) -> int:
        if c == max_c and max_c > 0:
            return (max_c - 1) * step_x + last_dx
        return c * step_x

    def y1_for_r(r: int) -> int:
        if r == max_r and max_r > 0:
            return (max_r - 1) * step_y + last_dy
        return r * step_y

    out: list[TileSpec] = []
    for (r, c), path in sorted(by_rc.items()):
        x1 = x1_for_c(c)
        y1 = y1_for_r(r)
        out.append(
            TileSpec(
                path=path,
                r=r,
                c=c,
                bbox_roi={
                    "x1": x1,
                    "y1": y1,
                    "x2": min(roi_w, x1 + tile_size),
                    "y2": min(roi_h, y1 + tile_size),
                },
            )
        )
    return out


def validate_tile_bboxes(
    specs: list[TileSpec],
    roi_size: tuple[int, int] | None,
) -> tuple[list[str], bool]:
    """Return warnings and whether any tile extends outside the ROI canvas."""
    if not specs or roi_size is None:
        return [], False
    roi_w, roi_h = int(roi_size[0]), int(roi_size[1])
    warnings: list[str] = []
    outside = False
    for spec in specs:
        b = spec.bbox_roi
        if (
            b["x1"] < -1
            or b["y1"] < -1
            or b["x2"] > roi_w + 1
            or b["y2"] > roi_h + 1
        ):
            outside = True
            warnings.append(
                f"Tile {spec.path.name} bbox {b} extends outside ROI ({roi_w}×{roi_h}). "
                "Overlap/step may be wrong — upload zooms_manifest.json or re-export at 10%."
            )
    return warnings, outside


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
        "alignment_confidence": "high",
        "manifest": str(manifest_path),
    }
    return specs, meta


def resolve_effective_overlap(
    tile_meta: dict[str, Any],
    *,
    tile_size: int,
    configured_overlap: float = DEFAULT_OVERLAP_PCT,
) -> tuple[float, int, float, list[str]]:
    """Return (overlap_pct, overlap_px, center_dist, warning_notes)."""
    overlap_pct = float(tile_meta.get("overlap_pct") or configured_overlap)
    snap = tile_meta.get("overlap_snap") or {}
    step_x = snap.get("step_x")
    if step_x is not None and int(step_x) > 0:
        recovered = max(0.0, min(0.7, 1.0 - int(step_x) / max(1, tile_size)))
        if abs(recovered - overlap_pct) > 0.05:
            overlap_pct = recovered
    overlap_px = max(24.0, tile_size * overlap_pct)
    center_dist = max(36.0, overlap_px / 3.0)
    warnings: list[str] = []
    if step_x is not None and abs(float(configured_overlap) - overlap_pct) > 0.05:
        pct = int(round(overlap_pct * 100))
        cfg_pct = int(round(configured_overlap * 100))
        warnings.append(
            f"Tiles appear generated at ~{pct}% overlap (step {step_x}px); "
            f"configured default is {cfg_pct}%. Merge uses recovered step for NMS. "
            "Re-export zooms from dual-pathway or upload zooms_manifest.json."
        )
    return overlap_pct, int(overlap_px), center_dist, warnings


def alignment_confidence(tile_meta: dict[str, Any], *, has_full_wing: bool) -> str:
    source = str(tile_meta.get("source") or "")
    if source == "manifest":
        return "high"
    if has_full_wing and tile_meta.get("overlap_snap"):
        return "medium"
    if has_full_wing:
        return "medium"
    return "low"


def _find_full_wing(tiles_dir: Path) -> Path | None:
    for name in ("full_wing.jpg", "full_wing.jpeg", "full_wing.png"):
        path = tiles_dir / name
        if path.is_file():
            return path
    return None


def resolve_tiles(
    tiles_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> tuple[list[TileSpec], dict[str, Any]]:
    """Prefer manifest; else infer from plan_zoom filenames + dual-pathway grid."""
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
    full_wing = _find_full_wing(tiles_dir)
    roi_size: tuple[int, int] | None = None
    if full_wing is not None:
        try:
            with Image.open(full_wing) as wing:
                roi_size = wing.size
        except OSError:
            roi_size = None

    if image_files:
        tile_size = _measure_tile_size(image_files)
        snap_meta: dict[str, float | int] = {}
        estimated = estimate_roi_size_from_overlap(
            image_files, tile_size=tile_size, overlap_pct=DEFAULT_OVERLAP_PCT
        )
        if estimated is not None:
            est_roi, snap_meta = estimated
            if roi_size is None:
                roi_size = est_roi

        effective_overlap = DEFAULT_OVERLAP_PCT
        if snap_meta.get("step_x"):
            effective_overlap = max(
                0.0, min(0.7, 1.0 - int(snap_meta["step_x"]) / max(1, tile_size))
            )

        if snap_meta and roi_size is not None:
            specs = infer_tiles_from_snap(
                image_files,
                snap_meta=snap_meta,
                roi_size=roi_size,
                tile_size=tile_size,
            )
        else:
            specs = infer_tiles_from_files(
                image_files,
                overlap_pct=effective_overlap,
                roi_size=roi_size,
                tile_size=tile_size,
            )

        bbox_warnings, bbox_outside = validate_tile_bboxes(specs, roi_size)
        source = "inferred_grid"
        if roi_size is not None:
            source = "inferred_grid_overlap_snap" if snap_meta else "inferred_grid_with_roi"

        align_input = {"source": source, "overlap_snap": snap_meta}
        alignment = alignment_confidence(
            align_input, has_full_wing=full_wing is not None
        )
        if bbox_outside:
            alignment = "low"

        return specs, {
            "tile_size": tile_size,
            "overlap_pct": effective_overlap,
            "configured_overlap_pct": DEFAULT_OVERLAP_PCT,
            "source": source,
            "alignment_confidence": alignment,
            "roi_size": list(roi_size) if roi_size else None,
            "overlap_snap": snap_meta or None,
            "bbox_warnings": bbox_warnings or None,
            "tiles": [
                {"file": t.path.name, "r": t.r, "c": t.c, "bbox_roi": t.bbox_roi}
                for t in specs
            ],
        }

    image_files = [
        p
        for p in sorted(tiles_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and p.name.lower() not in {"full_wing.jpg", "full_wing.jpeg", "full_wing.png", "overlay.png"}
    ]
    specs: list[TileSpec] = []
    step = tile_step()
    for i, path in enumerate(image_files):
        size = _measure_tile_size([path])
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
                    "x2": x1 + size,
                    "y2": y1 + size,
                },
            )
        )
    return specs, {
        "tile_size": DEFAULT_TILE_SIZE,
        "overlap_pct": DEFAULT_OVERLAP_PCT,
        "source": "flat_list",
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
    """Detect on each zoom tile first, then merge in ROI space, then count.

    Pipeline order (never detect on a stitched mosaic):
      1. resolve tile metadata (manifest / full_wing / overlap snap)
      2. detect symbols on every ``plan_zoom_*`` tile independently
      3. map local boxes → ROI coords using each tile's ``bbox_roi``
      4. NMS merge (count once in overlap)
      5. evaluate on full_wing crops (glyph + vision judge)
      6. finalize counts; overlay drawn on ``full_wing`` when present
    """
    nms_iou = float(nms_iou if nms_iou is not None else config.SYMBOL_COUNT_NMS_IOU)
    tiles, tile_meta = resolve_tiles(tiles_dir, manifest_path=manifest_path)
    if not tiles:
        raise RuntimeError("No zoom tile images found in the uploaded folder.")

    full_wing = _find_full_wing(tiles_dir)
    if full_wing is not None and not tile_meta.get("roi_size"):
        try:
            with Image.open(full_wing) as wing:
                tile_meta = {**tile_meta, "roi_size": list(wing.size)}
        except OSError:
            pass

    valid_keys = legend_count_keys(legend)
    tile_size = int(tile_meta.get("tile_size") or DEFAULT_TILE_SIZE)
    overlap_pct, overlap_px, center_dist, overlap_warnings = resolve_effective_overlap(
        tile_meta, tile_size=tile_size, configured_overlap=DEFAULT_OVERLAP_PCT
    )
    tile_meta = {
        **tile_meta,
        "overlap_pct": overlap_pct,
        "effective_overlap_pct": overlap_pct,
        "overlap_px": overlap_px,
        "center_dist_px": center_dist,
        "alignment_confidence": tile_meta.get("alignment_confidence")
        or alignment_confidence(tile_meta, has_full_wing=full_wing is not None),
    }
    notes: list[str] = [
        "Pipeline: detect each zoom tile → map via bbox_roi metadata → "
        "NMS merge → evaluate on full_wing → count "
        f"({len(tiles)} tile(s), source={tile_meta.get('source')}, "
        f"tile={tile_size}px, overlap={overlap_pct:.0%} ({overlap_px}px), "
        f"ROI={tile_meta.get('roi_size')}, align={tile_meta.get('alignment_confidence')}, "
        f"NMS IoU={nms_iou} + center {center_dist:.0f}px)."
    ]
    notes.extend(overlap_warnings)
    notes.extend(tile_meta.get("bbox_warnings") or [])
    if config.SYMBOL_COUNT_REQUIRE_MANIFEST_WARN and str(
        tile_meta.get("source") or ""
    ).startswith("inferred"):
        notes.append(
            "WARNING: zooms_manifest.json missing — merge alignment is low/guessed. "
            "Re-upload the dual-pathway zooms folder including zooms_manifest.json "
            "+ full_wing.jpg for correct bbox_roi placement."
        )
        tile_meta = {
            **tile_meta,
            "manifest_missing": True,
            "manifest_warning": True,
        }
    elif str(tile_meta.get("source") or "") == "manifest":
        tile_meta = {**tile_meta, "manifest_missing": False}
    if full_wing is not None:
        notes.append(
            f"Using full_wing as ROI frame for merge/count: {full_wing.name}."
        )

    meta: dict[str, Any] = {
        "linear_keys": set(),
        "hash_key": None,
        "camera_key": None,
        "glyphs_available": False,
    }

    # --- Phase 1: DETECT (per tile; no stitch) ---
    tile_detection_records: list[dict[str, Any]] = []
    pooled: list[SymbolDetection] = []
    raw_total = 0

    for tile_idx, tile in enumerate(tiles):
        image = Image.open(tile.path).convert("RGB")
        dets, tile_notes, tile_meta_det = detect_symbols_raw(
            image=image,
            legend=legend,
            glyph_dir=glyph_dir,
            symbol_file_suffix=symbol_file_suffix,
            nms_iou=nms_iou,
            apply_local_nms=False,
            # Per-tile Gemini vision OCR + symbol locate when FOLDER_USE_VISION_OCR
            # (costly; user-requested for max recall).
            use_vision_ocr=bool(config.FOLDER_USE_VISION_OCR),
            exclude_top_pct=0.0,
        )
        if config.FOLDER_USE_VISION_OCR and tile_idx == 0:
            notes.append(
                "Folder detect: per-tile Gemini vision OCR + symbol locate ENABLED "
                f"({config.llm_model()}) — slower/costlier, higher recall."
            )
            for n in tile_notes:
                if "Vision symbol locate skipped" in n or "Vision OCR" in n:
                    notes.append(f"{tile.path.name}: {n}")
                    break
        raw_total += len(dets)
        if tile_meta_det.get("linear_keys"):
            meta["linear_keys"] = set(tile_meta_det["linear_keys"]) | set(
                meta.get("linear_keys") or set()
            )
        meta["hash_key"] = meta.get("hash_key") or tile_meta_det.get("hash_key")
        meta["camera_key"] = meta.get("camera_key") or tile_meta_det.get("camera_key")
        meta["glyphs_available"] = meta.get("glyphs_available") or tile_meta_det.get(
            "glyphs_available"
        )

        local_items: list[dict[str, Any]] = []
        for det in dets:
            local_box = {
                "x1": det.box.x1,
                "y1": det.box.y1,
                "x2": det.box.x2,
                "y2": det.box.y2,
            }
            roi_box = translate_pixel_box_to_roi(
                det.box, tile.bbox_roi, image_size=image.size
            )
            local_items.append(
                {
                    "symbol": det.symbol,
                    "score": det.score,
                    "source": det.source,
                    "qty": max(1, int(det.qty)),
                    "box_local": local_box,
                    "box_roi": {
                        "x1": roi_box.x1,
                        "y1": roi_box.y1,
                        "x2": roi_box.x2,
                        "y2": roi_box.y2,
                    },
                }
            )
            pooled.append(
                SymbolDetection(
                    symbol=det.symbol,
                    box=roi_box,
                    score=det.score,
                    source=det.source,
                    qty=max(1, int(det.qty)),
                    tile_file=tile.path.name,
                )
            )

        tile_detection_records.append(
            {
                "file": tile.path.name,
                "r": tile.r,
                "c": tile.c,
                "bbox_roi": dict(tile.bbox_roi),
                "image_size": [int(image.size[0]), int(image.size[1])],
                "detection_count": len(local_items),
                "detections": local_items,
            }
        )

    empty_tiles = [
        rec["file"]
        for rec in tile_detection_records
        if int(rec.get("detection_count") or 0) == 0
    ]
    if empty_tiles:
        notes.append(
            f"Tiles with 0 detections: {len(empty_tiles)}/{len(tiles)} "
            f"({', '.join(empty_tiles[:8])}"
            + ("…" if len(empty_tiles) > 8 else "")
            + "). Empty tiles usually mean CV found nothing on that zoom crop."
        )
    else:
        notes.append(f"All {len(tiles)} zoom tile(s) produced at least one detection.")

    if meta.get("linear_keys"):
        notes.append(
            "Linear/run items are not counted by glyph template matching. "
            "Part labels on the plan are still counted via text/OCR: "
            + ", ".join(sorted(meta["linear_keys"]))
        )

    # --- Phase 2: MERGE in ROI space (metadata-aligned) ---
    pooled = filter_to_legend(pooled, valid_keys)
    kept = nms_by_class(pooled, iou_threshold=nms_iou, center_dist=center_dist)
    kept = suppress_cross_label(kept, iou_threshold=0.5, source="template")
    kept = suppress_nearby_same_label(
        kept, source="ocr", max_center_dist=max(80.0, center_dist)
    )
    kept = suppress_nearby_same_label(
        kept, source="vision_ocr", max_center_dist=max(80.0, center_dist)
    )
    kept = suppress_nearby_same_label(
        kept, source="vision_detect", max_center_dist=max(80.0, center_dist)
    )
    kept = suppress_cross_source_duplicates(
        kept,
        sources={"ocr", "vision_ocr", "vision_detect", "template"},
        max_center_dist=max(36.0, center_dist * 0.75),
        iou_threshold=0.15,
    )
    kept = filter_to_legend(kept, valid_keys)
    post_merge_count = len(kept)

    # --- Phase 3: EVALUATE on full_wing (glyph + vision judge) ---
    roi_image = build_roi_canvas(
        tiles,
        full_wing_path=full_wing if full_wing is not None else None,
    )
    evaluated, eval_summary = evaluate_merged_detections(
        roi_image=roi_image,
        legend=legend,
        glyph_dir=glyph_dir,
        detections=kept,
    )
    validated = [ev.detection for ev in evaluated if ev.verdict == "accepted"]
    rejected_items = [ev.to_dict() for ev in evaluated if ev.verdict != "accepted"]
    accepted_items = [ev.to_dict() for ev in evaluated if ev.verdict == "accepted"]
    notes.append(
        f"Evaluation: {post_merge_count} merged → {eval_summary.get('accepted', 0)} accepted / "
        f"{eval_summary.get('rejected', 0)} rejected "
        f"({eval_summary.get('judged', 0)} vision-judged)."
    )

    use_folder_vision = bool(
        config.FOLDER_VISION_COUNT_VERIFY and config.llm_enabled()
    )
    pipeline_steps = [
        "detect_tiles",
        "map_bbox_roi",
        "nms_merge",
        "evaluate",
        "count",
    ]
    if use_folder_vision:
        pipeline_steps.insert(-1, "folder_vision_fill")
        notes.append(
            "Post-merge capped vision count verify on full_wing "
            "(per-tile vision OCR remains off)."
        )

    payload = _finalize_counts(
        legend=legend,
        kept=validated,
        notes=notes,
        meta=meta,
        image=roi_image if use_folder_vision else None,
        method="detect_merge_evaluate_count",
        raw_detection_count=raw_total,
        nms_iou=nms_iou,
        pre_eval_detections=kept,
        detection_items=accepted_items,
        rejected_detection_items=rejected_items,
        extra={
            "pipeline": pipeline_steps,
            "tile_count": len(tiles),
            "tile_merge": tile_meta,
            "overlap_deduped": True,
            "overlap_px": overlap_px,
            "center_dist_px": center_dist,
            "full_wing": full_wing.name if full_wing is not None else None,
            "tile_detections": tile_detection_records,
            "pre_merge_detection_count": raw_total,
            "post_merge_mark_count": post_merge_count,
            "evaluation_summary": eval_summary,
            "alignment_confidence": tile_meta.get("alignment_confidence"),
            "merge_source": tile_meta.get("source"),
            "manifest_missing": bool(tile_meta.get("manifest_missing")),
            "manifest_warning": bool(tile_meta.get("manifest_warning")),
            "empty_tiles": empty_tiles,
            "empty_tile_count": len(empty_tiles),
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
    """Detect per zoom tile, merge with ROI metadata, write result + overlay."""
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
    eval_summary = payload.get("evaluation_summary") or {}
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "pipeline": payload.get("pipeline"),
                "pre_merge": payload.get("pre_merge_detection_count"),
                "post_merge": payload.get("post_merge_mark_count"),
                "accepted": eval_summary.get("accepted"),
                "rejected": eval_summary.get("rejected"),
                "judged": eval_summary.get("judged"),
                "rejected_by_reason": eval_summary.get("rejected_by_reason"),
                "rejected_detections": payload.get("rejected_detections") or [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Compact per-tile audit (same data as result.tile_detections).
    tile_dets = payload.get("tile_detections") or []
    (output_dir / "tile_detections.json").write_text(
        json.dumps(
            {
                "pipeline": payload.get("pipeline"),
                "tile_count": payload.get("tile_count"),
                "pre_merge_detection_count": payload.get("pre_merge_detection_count"),
                "post_merge_mark_count": payload.get("post_merge_mark_count"),
                "tiles": tile_dets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_detections_overlay(
        image=roi_image,
        detections=payload.get("detections") or [],
        output_path=output_dir / "overlay.png",
    )
    rejected = payload.get("rejected_detections") or []
    if rejected:
        draw_detections_overlay(
            image=roi_image,
            detections=rejected,
            output_path=output_dir / "overlay_rejected.png",
            dashed=True,
        )
        combined = draw_detections_on_image(
            roi_image.copy().convert("RGB"),
            payload.get("detections") or [],
        )
        combined = draw_detections_on_image(combined, rejected, dashed=True)
        combined.save(output_dir / "overlay_with_rejected.png", format="PNG")
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
    "estimate_roi_size_from_overlap",
    "resolve_effective_overlap",
    "alignment_confidence",
    "infer_tiles_from_files",
    "infer_tiles_from_snap",
    "validate_tile_bboxes",
    "iter_overlapping_tiles",
    "load_tiles_from_manifest",
    "resolve_tiles",
    "run_symbol_count_folder_job",
    "tile_step",
    "translate_pixel_box_to_roi",
]
