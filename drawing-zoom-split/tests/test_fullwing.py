"""Full-wing pass: symbols too large for any tile are caught, never doubled."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.count_stage import merge_fullwing_large_hits  # noqa: E402
from pipeline.yolo_detect import YoloHit  # noqa: E402


def _hit(x1, y1, x2, y2, conf=0.9, cls=0, name="AHU", rescued=False, img="t.jpg") -> YoloHit:
    return YoloHit(
        class_id=cls,
        class_name=name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_name=img,
        rescued=rescued,
    )


def test_large_wing_hit_added_and_partial_tile_boxes_absorbed():
    """A 200px symbol cut by tile boundaries: tiles saw two partial slivers,
    the wing pass saw the whole shape → one final box."""
    partial_a = _hit(100, 100, 160, 290, conf=0.42, img="plan_zoom_r00_c00.jpg")
    partial_b = _hit(210, 100, 300, 290, conf=0.38, img="plan_zoom_r00_c01.jpg")
    whole = _hit(95, 95, 300, 295, conf=0.88, img="full_wing.jpg")

    merged = merge_fullwing_large_hits([partial_a, partial_b], [whole], min_px=120)
    assert len(merged) == 1
    assert merged[0].confidence == 0.88
    assert merged[0].image_name == "full_wing.jpg"


def test_small_wing_hits_are_ignored():
    """Small symbols are the tile pass's job — the downscaled wing view of
    them is lower quality and must not add boxes."""
    tile_hit = _hit(100, 100, 130, 130, conf=0.80, img="plan_zoom_r00_c00.jpg")
    small_wing = _hit(500, 500, 540, 540, conf=0.90, img="full_wing.jpg")
    merged = merge_fullwing_large_hits([tile_hit], [small_wing], min_px=120)
    assert merged == [tile_hit]


def test_lowconf_and_rescued_wing_hits_are_ignored():
    """Only confident primary wing boxes qualify — no new FP source."""
    lowconf = _hit(100, 100, 320, 320, conf=0.20, img="full_wing.jpg")
    rescued = _hit(400, 400, 620, 620, conf=0.50, img="full_wing.jpg", rescued=True)
    merged = merge_fullwing_large_hits([], [lowconf, rescued], min_px=120)
    assert merged == []


def test_distinct_small_symbol_near_large_one_survives():
    """A small tile-detected symbol overlapping the large box only slightly
    is a different device, not a partial view — it stays."""
    small = _hit(80, 80, 110, 110, conf=0.85, cls=1, name="CAM",
                 img="plan_zoom_r00_c00.jpg")
    whole = _hit(95, 95, 300, 295, conf=0.88, img="full_wing.jpg")
    merged = merge_fullwing_large_hits([small], [whole], min_px=120)
    assert len(merged) == 2
