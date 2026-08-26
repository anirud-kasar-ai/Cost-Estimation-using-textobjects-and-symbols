from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.cv.template_match import detect_symbols_from_glyphs


def test_template_match_finds_glyph_on_canvas(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("pipeline.cv.template_match.config.CV_TEMPLATE_THRESHOLD", 0.5)
    glyph = Image.new("L", (24, 24), color=255)
    import cv2

    glyph_arr = np.array(glyph)
    cv2.rectangle(glyph_arr, (6, 6), (18, 18), 0, -1)
    glyph_path = tmp_path / "AP.png"
    Image.fromarray(glyph_arr).save(glyph_path)

    plan = Image.new("RGB", (200, 200), color=(255, 255, 255))
    plan_arr = np.array(plan)
    plan_arr[80:104, 80:104] = 0
    plan = Image.fromarray(plan_arr)

    legend = SymbolTableInfo(entries=[SymbolEntry(symbol="AP", description="ACCESS POINT")])
    hits = detect_symbols_from_glyphs(plan, legend, tmp_path)
    assert any(h.symbol == "AP" for h in hits)
