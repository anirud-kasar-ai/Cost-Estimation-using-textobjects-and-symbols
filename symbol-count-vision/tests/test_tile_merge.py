from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.nms import (
    BoundingBox,
    SymbolDetection,
    nms_by_class,
)
from pipeline.cv.template_match import _resolve_glyph_label, detect_symbols_from_glyphs
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.symbol_count import filter_to_legend, legend_count_keys
from pipeline.tile_merge import (
    DEFAULT_OVERLAP_PCT,
    DEFAULT_TILE_SIZE,
    TileSpec,
    estimate_roi_size_from_overlap,
    infer_tiles_from_files,
    infer_tiles_from_snap,
    iter_overlapping_tiles,
    merge_exclusive_overlap,
    resolve_effective_overlap,
    resolve_tiles,
    tile_exclusive_bbox,
    tile_step,
    translate_pixel_box_to_roi,
    validate_tile_bboxes,
)


def test_tile_step_matches_dual_pathway():
    assert tile_step(1152, 0.20) == 922
    assert tile_step(768, 0.10) == 691
    assert tile_step(653, 0.10) == 588
    assert tile_step(588, 0.10) == 529
    assert tile_step() == 529
    assert DEFAULT_OVERLAP_PCT == 0.10
    assert DEFAULT_TILE_SIZE == 588


def test_infer_tiles_bbox_roi_for_adjacent_columns(tmp_path: Path):
    p0 = tmp_path / "plan_zoom_r00_c00.jpg"
    p1 = tmp_path / "plan_zoom_r00_c01.jpg"
    Image.new("RGB", (768, 768), (255, 255, 255)).save(p0)
    Image.new("RGB", (768, 768), (255, 255, 255)).save(p1)
    specs = infer_tiles_from_files([p0, p1])
    assert len(specs) == 2
    step = tile_step(768, 0.10)
    assert specs[0].bbox_roi == {"x1": 0, "y1": 0, "x2": 768, "y2": 768}
    assert specs[1].bbox_roi == {
        "x1": step,
        "y1": 0,
        "x2": step + 768,
        "y2": 768,
    }
    assert specs[1].bbox_roi["x1"] == 691


def test_last_column_snaps_to_roi_edge_like_dual_pathway(tmp_path: Path):
    """page_020_3 last column is 3849 (edge snap), not 6*691=4146."""
    tiles = []
    for c in range(7):
        p = tmp_path / f"plan_zoom_r00_c{c:02d}.jpg"
        Image.new("RGB", (768, 768), (255, 255, 255)).save(p)
        tiles.append(p)
    specs = infer_tiles_from_files(tiles, roi_size=(4617, 1791))
    last = next(s for s in specs if s.c == 6)
    assert last.bbox_roi["x1"] == 3849
    assert last.bbox_roi["x2"] == 4617


def test_iter_overlapping_tiles_matches_awing_manifest():
    grid = iter_overlapping_tiles(4617, 1791, tile_size=768, overlap_pct=0.10)
    by_rc = {(r, c): bbox for r, c, bbox in grid}
    assert by_rc[(0, 0)] == (0, 0, 768, 768)
    assert by_rc[(0, 1)] == (691, 0, 1459, 768)
    assert by_rc[(0, 6)] == (3849, 0, 4617, 768)
    assert by_rc[(2, 0)][1] == 1023


def _write_dual_pathway_zoom_folder(
    dest: Path,
    *,
    roi_w: int = 4617,
    roi_h: int = 1791,
    tile_size: int = 768,
    overlap_pct: float = 0.10,
) -> list[Path]:
    """Crop unique-content tiles the same way dual-pathway ``roi_zoom`` does."""
    import numpy as np

    rng = np.random.default_rng(20)
    canvas = rng.integers(0, 256, size=(roi_h, roi_w), dtype=np.uint8)
    # Extra structure so overlap strips are distinctive.
    yy, xx = np.ogrid[:roi_h, :roi_w]
    canvas = np.bitwise_xor(canvas, ((xx * 13 + yy * 7) % 256).astype(np.uint8))
    roi = Image.fromarray(canvas).convert("RGB")
    dest.mkdir(parents=True, exist_ok=True)
    roi.save(dest / "full_wing.jpg", format="JPEG", quality=90)
    paths: list[Path] = []
    for r, c, bbox in iter_overlapping_tiles(
        roi_w, roi_h, tile_size=tile_size, overlap_pct=overlap_pct
    ):
        tile = roi.crop(bbox)
        path = dest / f"plan_zoom_r{r:02d}_c{c:02d}.jpg"
        tile.save(path, format="JPEG", quality=94)
        paths.append(path)
    return paths


def test_overlap_match_recovers_awing_edge_snap(tmp_path: Path):
    """Last col x1=3849 / last row y1=1023 even without manifest or full_wing."""
    paths = _write_dual_pathway_zoom_folder(tmp_path)
    est = estimate_roi_size_from_overlap(paths, tile_size=768, overlap_pct=0.10)
    assert est is not None
    (roi_w, roi_h), meta = est
    assert roi_w == 4617
    assert roi_h == 1791
    assert meta["last_x1"] == 3849
    assert meta["last_y1"] == 1023

    specs, info = resolve_tiles(tmp_path)
    last_col = next(s for s in specs if s.r == 0 and s.c == 6)
    last_row = next(s for s in specs if s.r == 2 and s.c == 0)
    assert last_col.bbox_roi["x1"] == 3849
    assert last_col.bbox_roi["x2"] == 4617
    assert last_row.bbox_roi["y1"] == 1023
    assert last_row.bbox_roi["y2"] == 1791
    assert info["source"] == "inferred_grid_overlap_snap"
    assert info["roi_size"] == [4617, 1791]


def test_merge_overlap_uses_recovered_step():
    overlap_pct, overlap_px, center_dist, _warnings = resolve_effective_overlap(
        {"overlap_pct": 0.10, "overlap_snap": {"step_x": 576}},
        tile_size=768,
        configured_overlap=0.10,
    )
    assert overlap_pct == 0.25
    assert overlap_px == 192
    assert center_dist == 64.0


def test_full_wing_with_25pct_tiles_recovers_step_and_bbox(tmp_path: Path):
    """full_wing present must not skip overlap snap — c01 x1=576 not 691."""
    _write_dual_pathway_zoom_folder(tmp_path, overlap_pct=0.25)
    specs, info = resolve_tiles(tmp_path)
    assert info["overlap_snap"]["step_x"] == 576
    assert info["overlap_pct"] == 0.25
    c1 = next(s for s in specs if s.c == 1 and s.r == 0)
    assert c1.bbox_roi["x1"] == 576
    assert c1.bbox_roi["x2"] == 576 + 768
    last_col = max(specs, key=lambda s: s.c)
    assert last_col.bbox_roi["x2"] <= info["roi_size"][0]
    warnings, outside = validate_tile_bboxes(specs, tuple(info["roi_size"]))
    assert not outside
    assert not warnings


def test_full_wing_with_10pct_tiles_uses_step_691(tmp_path: Path):
    _write_dual_pathway_zoom_folder(tmp_path, overlap_pct=0.10)
    specs, info = resolve_tiles(tmp_path)
    assert info["overlap_snap"]["step_x"] == 691
    assert abs(info["overlap_pct"] - 0.10) < 0.01
    c1 = next(s for s in specs if s.c == 1 and s.r == 0)
    assert c1.bbox_roi["x1"] == 691
    last_col = next(s for s in specs if s.r == 0 and s.c == 6)
    assert last_col.bbox_roi["x1"] == 3849
    assert last_col.bbox_roi["x2"] == 4617


def test_no_tile_bbox_outside_roi(tmp_path: Path):
    _write_dual_pathway_zoom_folder(tmp_path, overlap_pct=0.10)
    specs, info = resolve_tiles(tmp_path)
    warnings, outside = validate_tile_bboxes(specs, tuple(info["roi_size"]))
    assert not outside
    assert not warnings


def test_manifest_overrides_inference(tmp_path: Path):
    _write_dual_pathway_zoom_folder(tmp_path, overlap_pct=0.10)
    manifest = {
        "tile_size": 768,
        "overlap_pct": 0.10,
        "tiles": [
            {
                "r": 0,
                "c": 1,
                "file": "plan_zoom_r00_c01.jpg",
                "bbox_roi": {"x1": 999, "y1": 0, "x2": 1767, "y2": 768},
            }
        ],
    }
    (tmp_path / "zooms_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    specs, info = resolve_tiles(tmp_path)
    assert info["source"] == "manifest"
    c1 = next(s for s in specs if s.c == 1)
    assert c1.bbox_roi["x1"] == 999


def test_missing_manifest_warns_in_folder_count(tmp_path: Path, monkeypatch):
    """Inferred grid path should hard-warn about missing zooms_manifest.json."""
    from pipeline import tile_merge as tm
    from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo

    t0 = tmp_path / "plan_zoom_r00_c00.png"
    Image.new("RGB", (120, 120), color=(255, 255, 255)).save(t0)
    forced_specs = [
        tm.TileSpec(
            path=t0,
            r=0,
            c=0,
            bbox_roi={"x1": 0, "y1": 0, "x2": 120, "y2": 120},
        )
    ]
    monkeypatch.setattr(
        tm,
        "resolve_tiles",
        lambda tiles_dir, manifest_path=None: (
            forced_specs,
            {
                "tile_size": 120,
                "overlap_pct": 0.10,
                "source": "inferred_grid_overlap_snap",
                "alignment_confidence": "low",
                "roi_size": [120, 120],
            },
        ),
    )
    monkeypatch.setattr(tm.config, "FOLDER_VISION_COUNT_VERIFY", False)
    monkeypatch.setattr(tm.config, "FOLDER_USE_VISION_OCR", False)
    monkeypatch.setattr(tm.config, "SYMBOL_COUNT_REQUIRE_MANIFEST_WARN", True)

    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK, CAT6A JACK / CABLE",
            )
        ]
    )
    payload, _ = tm.count_symbols_on_zoom_folder(
        legend=legend,
        glyph_dir=None,
        tiles_dir=tmp_path,
        symbol_file_suffix=".json",
    )
    assert payload.get("manifest_missing") is True
    assert payload.get("manifest_warning") is True
    assert any("zooms_manifest.json missing" in n for n in payload.get("notes") or [])
    assert payload.get("merge_source") == "inferred_grid_overlap_snap"
    assert payload.get("empty_tile_count") == 1
    assert payload.get("empty_tiles") == ["plan_zoom_r00_c00.png"]
    assert any("Tiles with 0 detections" in n for n in payload.get("notes") or [])


def test_infer_tiles_from_snap_matches_grid(tmp_path: Path):
    paths = _write_dual_pathway_zoom_folder(tmp_path, overlap_pct=0.25)
    est = estimate_roi_size_from_overlap(paths, tile_size=768, overlap_pct=0.10)
    assert est is not None
    roi_size, snap = est
    from_snap = infer_tiles_from_snap(paths, snap_meta=snap, roi_size=roi_size, tile_size=768)
    from_grid = infer_tiles_from_files(paths, overlap_pct=0.25, roi_size=roi_size, tile_size=768)
    by_rc_snap = {(s.r, s.c): s.bbox_roi for s in from_snap}
    by_rc_grid = {(s.r, s.c): s.bbox_roi for s in from_grid}
    assert by_rc_snap == by_rc_grid


def test_overlap_pair_maps_to_same_roi_coord():
    """A mark in the extra last-column overlap maps to one ROI x."""
    prev = {"x1": 3455, "y1": 1023, "x2": 4223, "y2": 1791}
    last = {"x1": 3849, "y1": 1023, "x2": 4617, "y2": 1791}
    # Same physical x=4000: local x is 4000-3455=545 on prev, 4000-3849=151 on last.
    a = translate_pixel_box_to_roi(BoundingBox(545, 200, 575, 230), prev)
    b = translate_pixel_box_to_roi(BoundingBox(151, 200, 181, 230), last)
    assert abs(a.x1 - b.x1) < 1
    assert abs(a.y1 - b.y1) < 1


def test_translate_pixel_box_to_roi():
    box = BoundingBox(x1=10, y1=20, x2=30, y2=40)
    mapped = translate_pixel_box_to_roi(box, {"x1": 922, "y1": 0, "x2": 2074, "y2": 1152})
    assert mapped.x1 == 932
    assert mapped.y1 == 20
    assert mapped.x2 == 952
    assert mapped.y2 == 40


def test_overlap_nms_keeps_one_and_prefers_higher_qty():
    a = SymbolDetection(
        symbol="#",
        box=BoundingBox(x1=0, y1=0, x2=40, y2=40),
        score=0.8,
        source="triangle",
        qty=2,
    )
    b = SymbolDetection(
        symbol="#",
        box=BoundingBox(x1=5, y1=5, x2=45, y2=45),
        score=0.8,
        source="triangle",
        qty=4,
    )
    kept = nms_by_class([a, b], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].qty == 4


def test_overlap_nms_center_distance_merges_small_triangles():
    """Small drop marks in the overlap band have low IoU but share a center."""
    a = SymbolDetection(
        symbol="#",
        box=BoundingBox(x1=0, y1=0, x2=30, y2=30),
        score=0.85,
        source="triangle",
        qty=3,
    )
    b = SymbolDetection(
        symbol="#",
        box=BoundingBox(x1=20, y1=4, x2=50, y2=34),
        score=0.85,
        source="triangle",
        qty=3,
    )
    kept_iou_only = nms_by_class([a, b], iou_threshold=0.5)
    kept = nms_by_class([a, b], iou_threshold=0.5, center_dist=40.0)
    assert len(kept_iou_only) == 2
    assert len(kept) == 1


def test_template_skips_glyph_not_in_legend(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("pipeline.cv.template_match.config.CV_TEMPLATE_THRESHOLD", 0.5)
    import cv2
    import numpy as np

    glyph = Image.new("L", (24, 24), color=255)
    arr = np.array(glyph)
    cv2.rectangle(arr, (6, 6), (18, 18), 0, -1)
    # Filename does not match any legend key.
    Image.fromarray(arr).save(tmp_path / "UNKNOWN_XYZ.png")

    plan = Image.new("RGB", (200, 200), color=(255, 255, 255))
    plan_arr = np.array(plan)
    plan_arr[80:104, 80:104] = 0
    plan = Image.fromarray(plan_arr)

    legend = SymbolTableInfo(entries=[SymbolEntry(symbol="AP", description="ACCESS POINT")])
    hits = detect_symbols_from_glyphs(plan, legend, tmp_path)
    assert hits == []


def test_resolve_glyph_label_no_stem_fallback():
    assert _resolve_glyph_label("ORPHAN", {"AP": "AP"}) is None
    assert _resolve_glyph_label("AP_12", {"AP": "AP"}) == "AP"


def test_filter_to_legend_drops_unknown_keys():
    legend = SymbolTableInfo(entries=[SymbolEntry(symbol="#", description="DATA JACK")])
    valid = legend_count_keys(legend)
    dets = [
        SymbolDetection(
            symbol="#",
            box=BoundingBox(0, 0, 10, 10),
            score=0.9,
            source="triangle",
        ),
        SymbolDetection(
            symbol="NOT_IN_LEGEND",
            box=BoundingBox(20, 20, 30, 30),
            score=0.9,
            source="template",
        ),
    ]
    kept = filter_to_legend(dets, valid)
    assert len(kept) == 1
    assert kept[0].symbol == "#"


def test_manifest_load(tmp_path: Path):
    from pipeline.tile_merge import load_tiles_from_manifest

    img = tmp_path / "plan_zoom_r00_c00.jpg"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(img)
    manifest = {
        "tile_size": 1152,
        "overlap_pct": 0.2,
        "tiles": [
            {
                "r": 0,
                "c": 0,
                "file": "plan_zoom_r00_c00.jpg",
                "bbox_roi": {"x1": 0, "y1": 0, "x2": 1152, "y2": 1152},
            }
        ],
    }
    man_path = tmp_path / "zooms_manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    specs, meta = load_tiles_from_manifest(man_path, tmp_path)
    assert len(specs) == 1
    assert specs[0].bbox_roi["x2"] == 1152
    assert meta["source"] == "manifest"


def test_folder_merge_smoke(tmp_path: Path, monkeypatch):
    """Two overlapping tiles with the same triangle → one mark after NMS."""
    monkeypatch.setattr("pipeline.symbol_count.config.llm_enabled", lambda: False)
    monkeypatch.setattr("pipeline.symbol_count.tesseract_available", lambda: False)
    monkeypatch.setattr("pipeline.config.MERGE_EVAL_ENABLED", False)

    def make_tile(path: Path) -> None:
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.polygon([(100, 90), (85, 120), (115, 120)], fill=(0, 0, 0))
        img.save(path)

    t0 = tmp_path / "plan_zoom_r00_c00.png"
    t1 = tmp_path / "plan_zoom_r00_c01.png"
    make_tile(t0)
    make_tile(t1)

    # Force tiles to share ROI space so the same mark overlaps after translate.
    # Put both at x1=0 so detections land on top of each other.
    from pipeline import tile_merge as tm

    forced_specs = [
        tm.TileSpec(
            path=t0,
            r=0,
            c=0,
            bbox_roi={"x1": 0, "y1": 0, "x2": 200, "y2": 200},
        ),
        tm.TileSpec(
            path=t1,
            r=0,
            c=1,
            bbox_roi={"x1": 0, "y1": 0, "x2": 200, "y2": 200},
        ),
    ]
    monkeypatch.setattr(
        tm,
        "resolve_tiles",
        lambda tiles_dir, manifest_path=None: (
            forced_specs,
            {
                "tile_size": 200,
                "overlap_pct": 0.10,
                "source": "test",
                "overlap_snap": {"step_x": 180},
                "roi_size": [200, 200],
            },
        ),
    )
    monkeypatch.setattr(tm.config, "FOLDER_VISION_COUNT_VERIFY", False)
    monkeypatch.setattr(tm.config, "FOLDER_USE_VISION_OCR", False)

    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK, CAT6A JACK / CABLE",
            )
        ]
    )
    payload, _roi = tm.count_symbols_on_zoom_folder(
        legend=legend,
        glyph_dir=None,
        tiles_dir=tmp_path,
        symbol_file_suffix=".json",
    )
    assert payload["status"] == "done"
    assert payload["method"] == "detect_merge_evaluate_count"
    assert payload["pipeline"] == [
        "detect_tiles",
        "map_bbox_roi",
        "nms_merge",
        "evaluate",
        "count",
    ]
    assert payload["tile_count"] == 2
    assert "tile_detections" in payload
    assert len(payload["tile_detections"]) == 2
    # One physical mark across both tiles.
    assert payload["mark_counts"].get("#", 0) == 1
    assert payload["counts"].get("#", 0) >= 1


def _two_col_tiles():
    t00 = TileSpec(
        path=Path("plan_zoom_r00_c00.jpg"),
        r=0,
        c=0,
        bbox_roi={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
    )
    t01 = TileSpec(
        path=Path("plan_zoom_r00_c01.jpg"),
        r=0,
        c=1,
        bbox_roi={"x1": 90, "y1": 0, "x2": 190, "y2": 100},
    )
    t10 = TileSpec(
        path=Path("plan_zoom_r01_c00.jpg"),
        r=1,
        c=0,
        bbox_roi={"x1": 0, "y1": 90, "x2": 100, "y2": 190},
    )
    t11 = TileSpec(
        path=Path("plan_zoom_r01_c01.jpg"),
        r=1,
        c=1,
        bbox_roi={"x1": 90, "y1": 90, "x2": 190, "y2": 190},
    )
    by_rc = {(0, 0): t00, (0, 1): t01, (1, 0): t10, (1, 1): t11}
    return t00, t01, by_rc, t10, t11


def test_tile_exclusive_bbox_skips_prior_overlap():
    t00, t01, by_rc, t10, t11 = _two_col_tiles()
    e00 = tile_exclusive_bbox(t00, by_rc)
    assert (e00.x1, e00.y1, e00.x2, e00.y2) == (0, 0, 100, 100)
    e01 = tile_exclusive_bbox(t01, by_rc)
    assert e01.x1 == 100
    assert e01.y1 == 0
    e10 = tile_exclusive_bbox(t10, by_rc)
    assert e10.x1 == 0
    assert e10.y1 == 100
    e11 = tile_exclusive_bbox(t11, by_rc)
    assert e11.x1 == 100
    assert e11.y1 == 100


def test_overlap_duplicate_counted_once_from_primary_tile():
    t00, t01, _by_rc, _t10, _t11 = _two_col_tiles()
    primary = SymbolDetection(
        symbol="#",
        box=BoundingBox(50, 40, 70, 60),
        score=0.85,
        source="triangle",
        qty=2,
        tile_file="plan_zoom_r00_c00.jpg",
    )
    overlap_copy = SymbolDetection(
        symbol="#",
        box=BoundingBox(52, 41, 72, 61),
        score=0.84,
        source="triangle",
        qty=2,
        tile_file="plan_zoom_r00_c01.jpg",
    )
    kept, stats = merge_exclusive_overlap(
        [primary, overlap_copy],
        [t00, t01],
        nms_iou=0.3,
        primary_center_dist=20.0,
        salvage_center_dist=24.0,
    )
    assert len(kept) == 1
    assert kept[0].tile_file == "plan_zoom_r00_c00.jpg"
    assert stats["salvage_dropped"] == 1
    assert stats["salvage_kept"] == 0


def test_salvage_keeps_mark_cropped_off_primary_tile():
    t00, t01, _by_rc, _t10, _t11 = _two_col_tiles()
    # Only the next zoom sees the glyph (left 10% of tile 2 / right edge of tile 1).
    salvage = SymbolDetection(
        symbol="#",
        box=BoundingBox(91, 40, 99, 56),
        score=0.86,
        source="triangle",
        qty=1,
        tile_file="plan_zoom_r00_c01.jpg",
    )
    kept, stats = merge_exclusive_overlap(
        [salvage],
        [t00, t01],
        nms_iou=0.3,
        primary_center_dist=20.0,
        salvage_center_dist=24.0,
    )
    assert len(kept) == 1
    assert kept[0].tile_file == "plan_zoom_r00_c01.jpg"
    assert stats["salvage_kept"] == 1


def test_salvage_replaces_fragment_with_complete_box():
    t00, t01, _by_rc, _t10, _t11 = _two_col_tiles()
    fragment = SymbolDetection(
        symbol="#",
        box=BoundingBox(88, 42, 99, 54),
        score=0.70,
        source="triangle",
        qty=1,
        tile_file="plan_zoom_r00_c00.jpg",
    )
    complete = SymbolDetection(
        symbol="#",
        box=BoundingBox(90, 40, 108, 58),
        score=0.88,
        source="triangle",
        qty=1,
        tile_file="plan_zoom_r00_c01.jpg",
    )
    kept, _stats = merge_exclusive_overlap(
        [fragment, complete],
        [t00, t01],
        nms_iou=0.3,
        primary_center_dist=20.0,
        salvage_center_dist=24.0,
    )
    assert len(kept) == 1
    assert kept[0].tile_file == "plan_zoom_r00_c01.jpg"
    assert kept[0].score == 0.88


def test_close_distinct_marks_in_primary_tile_both_kept():
    t00, t01, _by_rc, _t10, _t11 = _two_col_tiles()
    a = SymbolDetection(
        symbol="#",
        box=BoundingBox(20, 40, 36, 56),
        score=0.85,
        source="triangle",
        qty=1,
        tile_file="plan_zoom_r00_c00.jpg",
    )
    b = SymbolDetection(
        symbol="#",
        box=BoundingBox(52, 40, 68, 56),
        score=0.85,
        source="triangle",
        qty=1,
        tile_file="plan_zoom_r00_c00.jpg",
    )
    kept, _stats = merge_exclusive_overlap(
        [a, b],
        [t00, t01],
        nms_iou=0.3,
        primary_center_dist=20.0,
        salvage_center_dist=24.0,
    )
    assert len(kept) == 2
