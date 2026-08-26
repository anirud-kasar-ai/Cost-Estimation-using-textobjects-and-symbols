from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.merge_evaluator import evaluate_merged_detections
from pipeline.tile_merge import DEFAULT_TILE_SIZE, resolve_effective_overlap


def _legend_with_glyph() -> SymbolTableInfo:
    glyph = Image.new("L", (24, 24), 255)
    draw = ImageDraw.Draw(glyph)
    draw.rectangle([6, 6, 18, 18], fill=0)
    buf = BytesIO()
    glyph.save(buf, format="PNG")
    return SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="AP",
                description="ACCESS POINT",
                symbol_image_png=buf.getvalue(),
            )
        ]
    )


def test_resolve_effective_overlap_uses_recovered_step():
    tile_meta = {
        "overlap_pct": 0.10,
        "overlap_snap": {"step_x": 576, "step_y": 576},
    }
    overlap_pct, overlap_px, center_dist, warnings = resolve_effective_overlap(
        tile_meta, tile_size=768, configured_overlap=0.10
    )
    assert overlap_pct == 0.25
    assert overlap_px == 192
    assert center_dist == 64.0
    assert warnings


def test_evaluator_auto_accepts_geometry():
    roi = Image.new("RGB", (200, 200), (255, 255, 255))
    det = SymbolDetection(
        symbol="#",
        box=BoundingBox(10, 10, 40, 40),
        score=0.85,
        source="triangle",
        qty=2,
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert len(evaluated) == 1
    assert evaluated[0].verdict == "accepted"
    assert summary["accepted"] == 1


def test_evaluator_rejects_low_ocr_without_glyph():
    roi = Image.new("RGB", (120, 120), (255, 255, 255))
    det = SymbolDetection(
        symbol="AP",
        box=BoundingBox(10, 10, 30, 30),
        score=0.42,
        source="ocr",
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=False
    ):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].verdict == "rejected"
    assert summary["rejected"] == 1


def test_evaluator_rejects_hourglass_without_glyph():
    roi = Image.new("RGB", (120, 120), (255, 255, 255))
    det = SymbolDetection(
        symbol="NETWORK CAMERA",
        box=BoundingBox(10, 10, 40, 40),
        score=0.8,
        source="hourglass",
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=False
    ), patch("pipeline.merge_evaluator.score_glyph_on_crop", return_value=0.10):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].verdict == "rejected"
    assert summary["rejected"] == 1


def test_evaluator_vision_judge_mocked():
    roi = Image.new("RGB", (120, 120), (255, 255, 255))
    det = SymbolDetection(
        symbol="AP",
        box=BoundingBox(10, 10, 30, 30),
        score=0.55,
        source="template",
    )
    judge_result = {
        "ok": True,
        "present": True,
        "symbol_matches": True,
        "confidence": 0.9,
        "reason": "matches legend glyph",
    }
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=True
    ), patch(
        "pipeline.merge_evaluator.score_glyph_on_crop", return_value=0.50
    ), patch("pipeline.llm_verify.judge_detection_crop", return_value=judge_result):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].verdict == "accepted"
    assert evaluated[0].judged is True
    assert summary["judged"] == 1
