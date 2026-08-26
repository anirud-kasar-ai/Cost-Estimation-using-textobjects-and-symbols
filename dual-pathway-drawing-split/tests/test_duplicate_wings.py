"""Tests for duplicate wing name handling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline.job import _merge_wing_maps  # noqa: E402
from pipeline.sheet_scan import WingHint, expand_wing_hints  # noqa: E402
from pipeline.wing_crop import assign_wing_instance_ids, anchors_from_specs  # noqa: E402


def test_assign_wing_instance_ids_suffixes_duplicates():
    specs = [
        {"name": "A-WING-EAST", "box": [0.1, 0.1, 0.4, 0.9]},
        {"name": "A-WING-EAST", "box": [0.6, 0.1, 0.9, 0.9]},
        {"name": "B-WING-WEST", "box": [0.2, 0.2, 0.5, 0.8]},
    ]
    out = assign_wing_instance_ids(specs)
    assert [item["instance_id"] for item in out] == [
        "A-WING-EAST",
        "A-WING-EAST-2",
        "B-WING-WEST",
    ]


def test_assign_wing_instance_ids_works_for_admin_block():
    specs = [
        {"name": "ADMIN BLOCK", "box": [0.05, 0.1, 0.45, 0.9]},
        {"name": "ADMIN BLOCK", "box": [0.55, 0.1, 0.95, 0.9]},
        {"name": "MULTI-USE BLDG", "box": [0.2, 0.2, 0.8, 0.8]},
    ]
    out = assign_wing_instance_ids(specs)
    assert [item["name"] for item in out] == [
        "ADMIN-BLOCK",
        "ADMIN-BLOCK",
        "MULTI-USE-BLDG",
    ]
    assert [item["instance_id"] for item in out] == [
        "ADMIN-BLOCK",
        "ADMIN-BLOCK-2",
        "MULTI-USE-BLDG",
    ]


def test_anchors_from_specs_keeps_duplicate_names():
    specs = assign_wing_instance_ids(
        [
            {"name": "A-WING-EAST", "box": [0.05, 0.1, 0.45, 0.9], "source": "vision_model"},
            {"name": "A-WING-EAST", "box": [0.55, 0.1, 0.95, 0.9], "source": "vision_model"},
        ]
    )
    anchors = anchors_from_specs(specs)
    assert len(anchors) == 2
    assert {a.name for a in anchors} == {"A-WING-EAST"}
    assert {a.instance_id for a in anchors} == {"A-WING-EAST", "A-WING-EAST-2"}


def test_expand_wing_hints_splits_duplicate_admin_blocks():
    hints = [
        WingHint(name="ADMIN-BLOCK", box=[0.05, 0.02, 0.18, 0.06]),
        WingHint(name="ADMIN-BLOCK", box=[0.62, 0.02, 0.78, 0.06]),
        WingHint(name="GYMNASIUM", box=[0.30, 0.02, 0.45, 0.06]),
    ]
    expanded = expand_wing_hints(hints)
    assert len(expanded) == 3
    admin_boxes = [item.box for item in expanded if item.name == "ADMIN-BLOCK"]
    assert len(admin_boxes) == 2
    assert admin_boxes[0][2] <= admin_boxes[1][0] + 0.05


def test_merge_wing_maps_keeps_all_building_instances():
    merged = _merge_wing_maps(
        llm_wings=[
            {"name": "ADMIN-BLOCK", "box": [0.05, 0.1, 0.4, 0.9], "source": "vision_model"},
            {"name": "ADMIN-BLOCK", "box": [0.55, 0.1, 0.95, 0.9], "source": "vision_model"},
            {"name": "ANNEX", "box": [0.2, 0.2, 0.8, 0.8], "source": "vision_model"},
        ],
        pdf_hints=[],
        label_points=[],
    )
    assert len(merged) == 3
    assert [item["instance_id"] for item in merged] == [
        "ADMIN-BLOCK",
        "ADMIN-BLOCK-2",
        "ANNEX",
    ]


def test_merge_wing_maps_keeps_duplicate_wing_names_from_pdf_and_vision():
    merged = _merge_wing_maps(
        llm_wings=[
            {"name": "A-WING-EAST", "box": [0.55, 0.1, 0.95, 0.9], "source": "vision_model"},
        ],
        pdf_hints=[
            {"name": "A-WING-EAST", "box": [0.05, 0.1, 0.45, 0.9], "source": "pdf_text"},
            {"name": "A-WING-EAST", "box": [0.55, 0.1, 0.95, 0.9], "source": "pdf_text"},
        ],
        label_points=[],
    )
    assert len(merged) == 3
    assert [item["instance_id"] for item in merged] == [
        "A-WING-EAST",
        "A-WING-EAST-2",
        "A-WING-EAST-3",
    ]
