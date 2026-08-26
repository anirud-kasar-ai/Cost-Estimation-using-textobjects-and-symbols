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
