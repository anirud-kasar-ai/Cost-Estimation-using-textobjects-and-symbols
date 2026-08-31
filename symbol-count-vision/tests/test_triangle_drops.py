from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from pipeline.cv.triangle_drops import count_drop_triangles, detect_drop_marks, triangle_is_noise


def test_count_drop_triangles_on_synthetic_image():
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Solid upward triangle
    draw.polygon([(200, 210), (185, 235), (215, 235)], fill=(0, 0, 0))
    count, boxes = count_drop_triangles(img, exclude_top_pct=0.0)
    assert count >= 1
    assert len(boxes) >= 1


def test_triangle_inside_circle_is_not_drop_mark():
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # AP-style circle with upward triangle inside.
    draw.ellipse((170, 170, 230, 230), outline=(0, 0, 0), width=2)
    draw.polygon([(200, 188), (188, 212), (212, 212)], fill=(0, 0, 0))
    drops, _cams = detect_drop_marks(img, exclude_top_pct=0.0)
    assert len(drops) == 0


def test_qty_digit_read_next_to_triangle():
    import shutil

    from PIL import ImageFont

    if not shutil.which("tesseract"):
        import pipeline.config as config

        if not Path(getattr(config, "TESSERACT_CMD", "") or "").exists():
            pytest.skip("tesseract not available")
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        pytest.skip("arial.ttf not available")

    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Right-pointing solid triangle with the quantity "3" on its base side.
    draw.polygon([(200, 180), (200, 220), (235, 200)], fill=(0, 0, 0))
    draw.text((160, 182), "3", fill=(0, 0, 0), font=font)
    drops, _cams = detect_drop_marks(img, exclude_top_pct=0.0)
    assert len(drops) == 1
    assert drops[0]["qty"] == 3


def test_raceway_wall_tile_keeps_hash_drops_despite_false_circles():
    """Real 653 tile: false circularity must not wipe wall # drops (qty 2 and 4)."""
    from pipeline.cv.triangle_drops import device_glyph_skip_centers
    from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
    from pipeline.symbol_count import detect_symbols_raw

    fixture = Path(__file__).parent / "fixtures" / "raceway_wall_drops_653.png"
    if not fixture.is_file():
        pytest.skip("fixture missing")
    img = Image.open(fixture).convert("RGB")
    assert device_glyph_skip_centers(img, exclude_top_pct=0.0) == []
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK, CAT6A JACK / CABLE",
            ),
            SymbolEntry(
                symbol="",
                description="SURFACE RACEWAY (WM5400)",
                part_number="WM5400",
            ),
        ]
    )
    dets, _notes, _meta = detect_symbols_raw(
        image=img,
        legend=legend,
        glyph_dir=None,
        symbol_file_suffix=".json",
        apply_local_nms=True,
        use_vision_ocr=False,
        exclude_top_pct=0.0,
    )
    hashes = [d for d in dets if d.source == "triangle"]
    assert len(hashes) >= 2
    race = [d for d in dets if "5400" in d.symbol or "RACEWAY" in d.symbol.upper()]
    assert len(race) >= 1


def test_arrowhead_sliver_is_noise():
    sliver = {"w": 10.0, "h": 19.0, "area": 70.0, "dy": 0.9}
    assert triangle_is_noise(sliver, dense_zoom=True, near_callout=False)
    real = {"w": 27.0, "h": 31.0, "area": 280.0, "dy": 0.4}
    assert not triangle_is_noise(real, dense_zoom=True, near_callout=False)
