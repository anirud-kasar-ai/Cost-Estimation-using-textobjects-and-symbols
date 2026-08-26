"""Tests for PDF-text symbol counting inside wing pixel boxes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo  # noqa: E402
from pipeline.symbol_count_cv import (  # noqa: E402
    aggregate_results_by_wing_name,
    assign_hits_to_wings,
    count_drop_triangles,
    count_spans_for_wing,
    count_symbols_on_wing_image,
    dedupe_overlapping_wings,
    expand_wing_bands,
    is_floor_plan_sheet,
    match_span_to_tag,
    point_in_cv_box,
    prefer_labeled_wings,
    select_wings_for_counting,
    tagged_legend_keys,
    wing_named_on_sheet,
)


def _legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(description="DATA RACK", symbol="R"),
            SymbolEntry(description="ACCESS POINT", symbol="AP"),
            SymbolEntry(description="DATA LINK", symbol="#"),
            SymbolEntry(description="SURFACE RACEWAY", symbol=None, part_number="WM2300"),
            SymbolEntry(description="MOUNT", symbol=None, mfg_model="ARUBA"),
            SymbolEntry(description="MOUNT", symbol=None, mfg_model="AXIS"),
        ]
    )


def test_tagged_legend_keys_skip_untagged_and_duplicates():
    keys = tagged_legend_keys(_legend())
    tags = [tag for tag, _ in keys]
    assert tags[0] == "AP"
    assert set(tags) == {"AP", "R", "#"}


def test_match_span_requires_whole_tag():
    tags = tagged_legend_keys(_legend())
    assert match_span_to_tag("R", tags) == "R"
    assert match_span_to_tag("AP", tags) == "AP"
    assert match_span_to_tag("AP-1", tags) == "AP"
    assert match_span_to_tag("#", tags) == "#"
    assert match_span_to_tag("#12", tags) == "#"
    assert match_span_to_tag("ROOM", tags) is None
    assert match_span_to_tag("APPROVED", tags) is None
    assert match_span_to_tag("MOUNT", tags) is None
    assert match_span_to_tag("SURFACE RACEWAY", tags) is None
    assert match_span_to_tag("DATA RACK", tags) is None


def test_match_span_rejects_title_font_not_large_callouts():
    tags = tagged_legend_keys(_legend())
    assert match_span_to_tag("R", tags, size=9.5) == "R"
    assert match_span_to_tag("R", tags, size=23.6, font="Arial") == "R"
    assert match_span_to_tag("R", tags, size=10.0, font="Romantic") is None
    assert match_span_to_tag("AP", tags, size=9.5) == "AP"


def test_assign_hit_to_smallest_overlapping_wing():
    tags = tagged_legend_keys(_legend())
    wings = [
        {
            "column_key": "ADMIN-BLDG",
            "cv_box_on_page": {"x1": 0, "y1": 0, "x2": 500, "y2": 500},
        },
        {
            "column_key": "B-WING-EAST",
            "cv_box_on_page": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        },
    ]
    spans = [{"text": "AP", "px": 150, "py": 150, "size": 9.5}]
    counts = assign_hits_to_wings(spans, wings, tags)
    assert counts["B-WING-EAST"] == {"AP": 1}
    assert counts["ADMIN-BLDG"] == {}


def test_dedupe_overlapping_same_name():
    wings = [
        {
            "wing_name": "A-WING-EAST",
            "cv_box_on_page": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        },
        {
            "wing_name": "A-WING-EAST",
            "cv_box_on_page": {"x1": 1, "y1": 1, "x2": 99, "y2": 99},
        },
    ]
    kept = dedupe_overlapping_wings(wings)
    assert len(kept) == 1


def test_floor_plan_sheet_filter():
    assert is_floor_plan_sheet("T-207") is True
    assert is_floor_plan_sheet("T-023") is False
    assert is_floor_plan_sheet("T-100") is False
    assert is_floor_plan_sheet("T-701") is False
    assert wing_named_on_sheet("A-WING-EAST", "T-207 A-WING-WEST, A-WING-EAST")
    assert not wing_named_on_sheet("EST", "T-210 C-WING-WEST, C-WING-WEST-CTR")


def test_point_in_cv_box_false_without_box():
    assert point_in_cv_box(10, 10, None) is False
    assert point_in_cv_box(10, 10, {"x1": 0, "y1": 0, "x2": 5, "y2": 5}) is False
    assert point_in_cv_box(10, 10, {"x1": 0, "y1": 0, "x2": 20, "y2": 20}) is True


def test_counts_only_spans_inside_wing_box():
    tags = tagged_legend_keys(_legend())
    box = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    spans = [
        {"text": "R", "px": 150, "py": 150, "size": 23.0},
        {"text": "R", "px": 10, "py": 10, "size": 23.0},
        {"text": "AP", "px": 180, "py": 120, "size": 9.5},
        {"text": "MOUNT", "px": 150, "py": 150, "size": 9.5},
    ]
    counts = count_spans_for_wing(spans, box, tags)
    assert counts == {"R": 1, "AP": 1}


def test_exclude_title_band_on_wing():
    tags = tagged_legend_keys(_legend())
    box = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    spans = [
        {"text": "AP", "px": 50, "py": 5, "size": 9.5},
        {"text": "AP", "px": 50, "py": 50, "size": 9.5},
    ]
    counts = count_spans_for_wing(spans, box, tags, exclude_top_pct=0.12)
    assert counts == {"AP": 1}


def test_prefer_pdf_text_over_vision_duplicates():
    wings = [
        {
            "wing_name": "A-WING-EAST",
            "source": "vision_model",
            "cv_box_on_page": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        },
        {
            "wing_name": "A-WING-WEST",
            "source": "pdf_text",
            "cv_box_on_page": {"x1": 0, "y1": 0, "x2": 200, "y2": 80},
        },
        {
            "wing_name": "A-WING-EAST",
            "source": "pdf_text",
            "cv_box_on_page": {"x1": 50, "y1": 90, "x2": 200, "y2": 180},
        },
    ]
    kept = prefer_labeled_wings(wings)
    assert [w["source"] for w in kept] == ["pdf_text", "pdf_text"]


def test_select_wings_keeps_all_names_and_prefers_pdf_text():
    wings = [
        {
            "wing_name": "A-WING-EAST",
            "source": "vision_model",
            "cv_box_on_page": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        },
        {
            "wing_name": "E-WING-SOUTH",
            "source": "vision_model",
            "cv_box_on_page": {"x1": 0, "y1": 100, "x2": 100, "y2": 200},
        },
        {
            "wing_name": "A-WING-EAST",
            "source": "pdf_text",
            "cv_box_on_page": {"x1": 50, "y1": 90, "x2": 200, "y2": 180},
        },
    ]
    kept = select_wings_for_counting(wings)
    names = {w["wing_name"] for w in kept}
    assert names == {"A-WING-EAST", "E-WING-SOUTH"}
    east = next(w for w in kept if w["wing_name"] == "A-WING-EAST")
    assert east["source"] == "pdf_text"


def test_expand_bands_cover_full_diagram_width():
    wings = [
        {
            "column_key": "A-WING-WEST",
            "wing_name": "A-WING-WEST",
            "cv_box_on_page": {"x1": 500, "y1": 1000, "x2": 5000, "y2": 2800},
        },
        {
            "column_key": "A-WING-EAST",
            "wing_name": "A-WING-EAST",
            "cv_box_on_page": {"x1": 2800, "y1": 2850, "x2": 7400, "y2": 4500},
        },
    ]
    diagram = {"x1": 460, "y1": 1000, "x2": 7400, "y2": 4600}
    bands = expand_wing_bands(wings, diagram, pad_y2=4800)
    west, east = bands
    assert west["cv_box_on_page"]["x1"] == 460
    assert east["cv_box_on_page"]["x2"] == 7400
    assert east["cv_box_on_page"]["y2"] == 4800
    tags = tagged_legend_keys(_legend())
    spans = [
        {"text": "AP", "px": 4300, "py": 1200, "size": 9.5},
        {"text": "AP", "px": 1760, "py": 4700, "size": 9.5},
        {"text": "AP", "px": 2180, "py": 4100, "size": 9.5},
        {"text": "R", "px": 3970, "py": 3975, "size": 23.5},
    ]
    counts = assign_hits_to_wings(spans, bands, tags)
    assert counts["A-WING-WEST"] == {"AP": 1}
    assert counts["A-WING-EAST"] == {"AP": 2, "R": 1}


def test_aggregate_by_wing_name_sums_across_sheets():
    aggregated = aggregate_results_by_wing_name(
        [
            {
                "wing_name": "A-WING-EAST",
                "wing_image": "a/page_020.jpg",
                "counts": {"AP": 1, "R": 1},
            },
            {
                "wing_name": "A-WING-EAST",
                "wing_image": "a/page_021.jpg",
                "counts": {"AP": 2, "J": 1, "#": 5},
            },
            {
                "wing_name": "B-WING-WEST",
                "wing_image": "b/page_022.jpg",
                "counts": {"AP": 3},
            },
        ]
    )
    assert len(aggregated) == 2
    east = next(r for r in aggregated if r["column_key"] == "A-WING-EAST")
    assert east["counts"] == {"AP": 3, "R": 1, "J": 1, "#": 5}
    assert east["sheets_processed"] == 2
    assert len(east["wing_images"]) == 2


def test_count_symbols_on_wing_image_uses_direct_crop(tmp_path):
    import numpy as np

    try:
        import cv2
    except ImportError:
        return

    tags = tagged_legend_keys(_legend())
    wing_jpg = tmp_path / "wing.jpg"
    canvas = np.full((200, 400, 3), 255, dtype=np.uint8)
    tri = np.array([[180, 150], [200, 190], [160, 190]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, tri, (0, 0, 0))
    cv2.imwrite(str(wing_jpg), canvas)

    wing = {
        "image": "wing.jpg",
        "cv_box_on_page": {"x1": 1000, "y1": 500, "x2": 1400, "y2": 700},
        "column_key": "A-WING-EAST",
    }
    spans = [
        {"text": "AP", "px": 1200, "py": 600, "size": 9.5, "font": "Arial"},
        {"text": "R", "px": 1300, "py": 650, "size": 23.0, "font": "Arial"},
    ]
    counts = count_symbols_on_wing_image(tmp_path, wing, spans, tags)
    assert counts == {"AP": 1, "R": 1}
