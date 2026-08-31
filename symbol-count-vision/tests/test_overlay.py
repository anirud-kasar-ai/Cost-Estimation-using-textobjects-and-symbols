from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.overlay import draw_detections_on_image, short_overlay_label


def test_short_overlay_label_skips_singleton_hash():
    assert short_overlay_label("#", 1) is None
    assert short_overlay_label("#", 4) == "#4"


def test_short_overlay_label_aliases():
    assert short_overlay_label("AP") == "AP"
    assert short_overlay_label("CAT6A DATA DROP LOCATION (QTY = 1) - NETWORK CAMERA") == "CAM"
    assert short_overlay_label("DATA POLE (30TC-4**V)") == "POLE"
    assert short_overlay_label("SURFACE RACEWAY (WM5400)") == "5400"


def test_draw_detections_does_not_emit_source_or_score():
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    out = draw_detections_on_image(
        img,
        [
            {
                "symbol": "#",
                "qty": 1,
                "score": 0.85,
                "source": "triangle",
                "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
            }
        ],
    )
    assert out.size == (80, 80)
