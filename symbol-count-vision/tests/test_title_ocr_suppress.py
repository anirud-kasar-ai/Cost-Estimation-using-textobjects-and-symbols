from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.ocr_tags import _is_title_or_keyplan_ocr_span


def test_title_wing_blob_is_suppressed():
    assert _is_title_or_keyplan_ocr_span(
        text_upper="G",
        height=12.0,
        neighbor_blob="A-WING EAST",
        max_tag_h=40.0,
    )


def test_key_plan_blob_is_suppressed():
    assert _is_title_or_keyplan_ocr_span(
        text_upper="AP",
        height=10.0,
        neighbor_blob="KEY PLAN",
        max_tag_h=40.0,
    )


def test_normal_wall_tag_not_suppressed():
    assert not _is_title_or_keyplan_ocr_span(
        text_upper="G",
        height=14.0,
        neighbor_blob="RM 101",
        max_tag_h=40.0,
    )


def test_oversized_single_letter_suppressed():
    assert _is_title_or_keyplan_ocr_span(
        text_upper="G",
        height=36.0,
        neighbor_blob="",
        max_tag_h=40.0,
    )


def test_keyplan_regions_from_words_covers_mini_plan():
    from pipeline.cv.nms import BoundingBox, SymbolDetection, suppress_in_boxes
    from pipeline.cv.ocr_tags import keyplan_regions_from_words

    regions = keyplan_regions_from_words(
        [("KEY", 3500.0, 1720.0, 3560.0, 1755.0), ("PLAN", 3570.0, 1720.0, 3680.0, 1755.0)]
    )
    assert len(regions) == 1
    box = regions[0]
    assert box.x1 < 3500.0
    assert box.y1 < 1656.0
    fake = SymbolDetection(
        symbol="#",
        box=BoundingBox(3511, 1656, 3535, 1706),
        score=0.85,
        source="triangle",
    )
    assert suppress_in_boxes([fake], regions) == []


def test_title_regions_from_words_covers_wing_label():
    from pipeline.cv.nms import BoundingBox, SymbolDetection, suppress_in_boxes
    from pipeline.cv.ocr_tags import title_regions_from_words

    regions = title_regions_from_words(
        [("A-WING", 1800.0, 380.0, 2100.0, 430.0), ("EAST", 2140.0, 380.0, 2280.0, 430.0)]
    )
    assert len(regions) == 1
    pole = SymbolDetection(
        symbol="DATA POLE",
        box=BoundingBox(1927, 433, 2011, 527),
        score=0.80,
        source="bowtie",
    )
    assert suppress_in_boxes([pole], regions) == []
