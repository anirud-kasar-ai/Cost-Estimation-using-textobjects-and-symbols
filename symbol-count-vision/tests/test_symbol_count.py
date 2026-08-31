from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.symbol_count import count_symbols_in_image_cv


def test_count_symbols_cv_merges_and_nms(tmp_path: Path):
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="R", description="DATA RACK"),
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
        ]
    )
    glyph_dir = tmp_path / "glyphs"
    glyph_dir.mkdir()
    (glyph_dir / "AP.png").write_bytes(b"png")

    from pipeline.detect.candidates import Candidate

    ocr_cands = [
        Candidate(
            tile_id="t",
            bbox=BoundingBox(x1=10, y1=10, x2=30, y2=30),
            shape_class_guess="letter_tag",
            ocr_text="R",
            score=0.9,
            source="ocr",
            extras={"legend_hint": "R"},
        ),
        Candidate(
            tile_id="t",
            bbox=BoundingBox(x1=12, y1=12, x2=28, y2=28),
            shape_class_guess="letter_tag",
            ocr_text="R",
            score=0.8,
            source="ocr",
            extras={"legend_hint": "R"},
        ),
        Candidate(
            tile_id="t",
            bbox=BoundingBox(x1=100, y1=100, x2=120, y2=120),
            shape_class_guess="letter_tag",
            ocr_text="AP",
            score=0.95,
            source="ocr",
            extras={"legend_hint": "AP"},
        ),
    ]

    with patch("pipeline.symbol_count.config.llm_enabled", return_value=False), \
         patch("pipeline.symbol_count.config.MERGE_EVAL_ENABLED", False), \
         patch("pipeline.detect.candidates.propose_candidates", return_value=ocr_cands):
        payload = count_symbols_in_image_cv(
            image=img,
            legend=legend,
            glyph_dir=glyph_dir,
            symbol_file_suffix=".pdf",
            nms_iou=0.5,
        )

    assert payload["status"] == "done"
    assert payload["method"] == "single_image_cv"
    assert payload["counts"]["R"] == 1
    assert payload["counts"]["AP"] == 1
    assert payload["post_nms_count"] == 2
