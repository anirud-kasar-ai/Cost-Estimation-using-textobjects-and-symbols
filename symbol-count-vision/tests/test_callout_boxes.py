from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.callout_boxes import (
    normalize_callout_text,
    normalize_stacked_callout,
    reinforce_hash_from_callouts,
)
from pipeline.cv.nms import BoundingBox, SymbolDetection
from PIL import Image


def test_normalize_callout_text_accepts_pipe_forms():
    assert normalize_callout_text("1|13") == "1|13"
    assert normalize_callout_text("1 | 5") == "1|5"
    assert normalize_callout_text("1丨8") == "1|8"


def test_normalize_callout_text_accepts_space_dash_slash():
    assert normalize_callout_text("1 13") == "1|13"
    assert normalize_callout_text("1-5") == "1|5"
    assert normalize_callout_text("1/8") == "1|8"
    assert normalize_callout_text("113") == "1|13"


def test_normalize_stacked_callout():
    assert normalize_stacked_callout("1", "13") == "1|13"
    assert normalize_stacked_callout("1", "5") == "1|5"


def test_normalize_callout_text_rejects_elevation_room_sheet():
    assert normalize_callout_text("+48") is None
    assert normalize_callout_text("+49") is None
    assert normalize_callout_text("A-101") is None
    assert normalize_callout_text("1/D5") is None


def test_callout_near_triangle_does_not_double_count():
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    existing = [
        SymbolDetection(
            symbol="#",
            box=BoundingBox(90, 90, 110, 110),
            score=0.85,
            source="triangle",
            qty=1,
        )
    ]
    hints = [{"cx": 100.0, "cy": 100.0, "qty": 2, "text": "1|5"}]
    out = reinforce_hash_from_callouts(
        image=img,
        hash_key="#",
        existing=existing,
        callout_hints=hints,
        radius_px=80.0,
    )
    drops = [d for d in out if d.source in {"triangle", "callout_drop"}]
    assert len(drops) == 1
    assert drops[0].source == "triangle"
    assert drops[0].qty == 2


def test_callout_without_triangle_does_not_seed_hash():
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    hints = [{"cx": 50.0, "cy": 60.0, "qty": 1, "text": "1|13"}]
    out = reinforce_hash_from_callouts(
        image=img,
        hash_key="#",
        existing=[],
        callout_hints=hints,
        radius_px=80.0,
    )
    drops = [d for d in out if d.source == "callout_drop"]
    assert drops == []
