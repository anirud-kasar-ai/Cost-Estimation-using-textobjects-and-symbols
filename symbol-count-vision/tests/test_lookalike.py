from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.lookalike import (
    detect_conduit_stubs,
    reclassify_j_hook_clusters,
)
from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.cv.template_match import LINEAR_GLYPH_RE
from pipeline.cv.triangle_drops import detect_data_pole_boxes, detect_drop_marks
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo


def _legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="J", description="JUNCTION BOX"),
            SymbolEntry(symbol="", description="J-HOOK (SINGLE/STACKED)"),
            SymbolEntry(symbol="", description="CONDUIT STUB"),
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
            SymbolEntry(symbol="", description="DATA POLE", part_number="30TC-4**V"),
        ]
    )


def test_linear_regex_excludes_stub_pole_and_jhook():
    assert LINEAR_GLYPH_RE.search("CONDUIT (EMT & GRC)")
    assert LINEAR_GLYPH_RE.search("UNDERGROUND CONDUIT")
    assert not LINEAR_GLYPH_RE.search("CONDUIT STUB")
    assert not LINEAR_GLYPH_RE.search("DATA POLE (30TC-4**V)")
    assert not LINEAR_GLYPH_RE.search("J-HOOK (SINGLE/STACKED)")
    assert LINEAR_GLYPH_RE.search("SURFACE RACEWAY (WM5400)")


def test_three_j_on_a_line_become_one_jhook_not_three_jboxes():
    legend = _legend()
    dets = [
        SymbolDetection(
            symbol="J",
            box=BoundingBox(x1=10, y1=20, x2=22, y2=36),
            score=0.9,
            source="ocr",
        ),
        SymbolDetection(
            symbol="J",
            box=BoundingBox(x1=40, y1=21, x2=52, y2=37),
            score=0.88,
            source="ocr",
        ),
        SymbolDetection(
            symbol="J",
            box=BoundingBox(x1=70, y1=19, x2=82, y2=35),
            score=0.91,
            source="ocr",
        ),
    ]
    out = reclassify_j_hook_clusters(dets, legend)
    hooks = [d for d in out if "HOOK" in d.symbol.upper()]
    jboxes = [d for d in out if d.symbol == "J"]
    assert len(hooks) == 1
    assert jboxes == []


def test_isolated_j_stays_junction_box():
    legend = _legend()
    dets = [
        SymbolDetection(
            symbol="J",
            box=BoundingBox(x1=10, y1=20, x2=22, y2=36),
            score=0.9,
            source="ocr",
        )
    ]
    out = reclassify_j_hook_clusters(dets, legend)
    assert len(out) == 1
    assert out[0].symbol == "J"


def test_conduit_stub_long_middle_bar_detected_plain_e_not():
    stub_img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(stub_img)
    draw.rectangle((80, 60, 90, 140), fill=(0, 0, 0))
    draw.rectangle((80, 60, 120, 70), fill=(0, 0, 0))
    draw.rectangle((80, 96, 170, 108), fill=(0, 0, 0))
    draw.rectangle((80, 130, 120, 140), fill=(0, 0, 0))
    stubs = detect_conduit_stubs(stub_img, exclude_top_pct=0.0)
    assert len(stubs) >= 1

    plain = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(plain)
    draw.rectangle((80, 60, 90, 140), fill=(0, 0, 0))
    draw.rectangle((80, 60, 120, 70), fill=(0, 0, 0))
    draw.rectangle((80, 96, 120, 108), fill=(0, 0, 0))
    draw.rectangle((80, 130, 120, 140), fill=(0, 0, 0))
    assert detect_conduit_stubs(plain, exclude_top_pct=0.0) == []


def test_filled_triangle_is_not_conduit_stub():
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(150, 70), (130, 110), (170, 110)], fill=(0, 0, 0))
    assert detect_conduit_stubs(img, exclude_top_pct=0.0) == []


def test_bowtie_data_pole_triangles_are_not_hash_drops():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((160, 160, 240, 240), outline=(0, 0, 0), width=3)
    draw.polygon([(162, 162), (162, 238), (200, 200)], fill=(0, 0, 0))
    draw.polygon([(238, 162), (238, 238), (200, 200)], fill=(0, 0, 0))
    drops, _cams = detect_drop_marks(img, exclude_top_pct=0.0)
    poles = detect_data_pole_boxes(img, exclude_top_pct=0.0)
    assert len(poles) >= 1
    assert drops == []


def test_data_pole_keeps_flanking_hash_drops():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((160, 160, 240, 240), outline=(0, 0, 0), width=3)
    draw.polygon([(162, 162), (162, 238), (200, 200)], fill=(0, 0, 0))
    draw.polygon([(238, 162), (238, 238), (200, 200)], fill=(0, 0, 0))
    draw.polygon([(110, 180), (110, 220), (148, 200)], fill=(0, 0, 0))
    draw.polygon([(290, 180), (290, 220), (252, 200)], fill=(0, 0, 0))
    drops, cams = detect_drop_marks(img, exclude_top_pct=0.0, read_quantities=False)
    poles = detect_data_pole_boxes(img, exclude_top_pct=0.0)
    assert len(poles) >= 1
    assert len(drops) >= 2
    assert cams == []


def test_lone_triangle_still_counts_as_drop():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(200, 180), (185, 220), (215, 220)], fill=(0, 0, 0))
    drops, _cams = detect_drop_marks(img, exclude_top_pct=0.0)
    assert len(drops) >= 1
    assert detect_data_pole_boxes(img, exclude_top_pct=0.0) == []


def test_hourglass_camera_triangles_are_not_hash_drops():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(70, 100), (70, 124), (125, 112)], fill=(0, 0, 0))
    draw.rectangle((130, 104, 150, 120), outline=(0, 0, 0), width=2)
    draw.line((130, 104, 150, 120), fill=(0, 0, 0), width=2)
    draw.line((150, 104, 130, 120), fill=(0, 0, 0), width=2)
    draw.polygon([(210, 100), (210, 124), (155, 112)], fill=(0, 0, 0))
    drops, cams = detect_drop_marks(img, exclude_top_pct=0.0, read_quantities=False)
    assert len(cams) >= 1
    assert drops == []


def test_title_east_letter_e_is_not_a_plan_tag():
    from pipeline.cv.ocr_tags import _is_title_or_keyplan_ocr_span

    assert _is_title_or_keyplan_ocr_span(
        text_upper="E",
        height=12,
        neighbor_blob="A-WING (EAST)",
        max_tag_h=40,
    )
    assert not _is_title_or_keyplan_ocr_span(
        text_upper="J",
        height=12,
        neighbor_blob="WALL JBOX",
        max_tag_h=40,
    )
