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


def _legend_hash_and_ap() -> SymbolTableInfo:
    def _png(kind: str) -> bytes:
        glyph = Image.new("L", (24, 24), 255)
        draw = ImageDraw.Draw(glyph)
        if kind == "hash":
            draw.polygon([(12, 4), (4, 20), (20, 20)], fill=0)
        else:
            draw.ellipse([2, 2, 22, 22], outline=0, width=2)
            draw.polygon([(12, 6), (6, 18), (18, 18)], fill=0)
        buf = BytesIO()
        glyph.save(buf, format="PNG")
        return buf.getvalue()

    return SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="PERMANENT LINK",
                symbol_image_png=_png("hash"),
            ),
            SymbolEntry(
                symbol="AP",
                description="ACCESS POINT",
                symbol_image_png=_png("ap"),
            ),
        ]
    )


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
    assert summary.get("remapped", 0) == 0


def test_evaluator_remaps_triangle_to_better_glyph():
    roi = Image.new("RGB", (200, 200), (255, 255, 255))
    det = SymbolDetection(
        symbol="#",
        box=BoundingBox(10, 10, 40, 40),
        score=0.85,
        source="triangle",
        qty=2,
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=False
    ), patch(
        "pipeline.merge_evaluator._rank_keys_on_crop",
        return_value=[("AP", 0.82), ("#", 0.40)],
    ):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_hash_and_ap(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].verdict == "accepted"
    assert evaluated[0].detection.symbol == "AP"
    assert evaluated[0].detection.qty == 1
    assert evaluated[0].mapped_from == "#"
    assert evaluated[0].detection.classify_source == "glyph_validate"
    assert "mapped # → AP" in evaluated[0].reason
    assert summary["remapped"] == 1
    assert det.symbol == "#"


def test_evaluator_ocr_does_not_remap_to_lookalike():
    roi = Image.new("RGB", (120, 120), (255, 255, 255))
    det = SymbolDetection(
        symbol="5400",
        box=BoundingBox(10, 10, 50, 24),
        score=0.90,
        source="ocr",
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=False
    ), patch("pipeline.merge_evaluator.score_glyph_on_crop", return_value=0.70):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].detection.symbol == "5400"
    assert evaluated[0].mapped_from is None
    assert summary["remapped"] == 0


def test_evaluator_does_not_remap_hourglass_to_pole():
    roi = Image.new("RGB", (200, 200), (255, 255, 255))
    cam_png = Image.new("L", (24, 24), 255)
    pole_png = Image.new("L", (24, 24), 255)
    ImageDraw.Draw(cam_png).rectangle([4, 10, 20, 18], outline=0)
    ImageDraw.Draw(pole_png).polygon([(2, 2), (2, 22), (12, 12)], fill=0)
    cam_buf = BytesIO()
    pole_buf = BytesIO()
    cam_png.save(cam_buf, format="PNG")
    pole_png.save(pole_buf, format="PNG")
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="",
                description="NETWORK CAMERA",
                symbol_image_png=cam_buf.getvalue(),
            ),
            SymbolEntry(
                symbol="",
                description="DATA POLE",
                part_number="30TC-4**V",
                symbol_image_png=pole_buf.getvalue(),
            ),
        ]
    )
    det = SymbolDetection(
        symbol="NETWORK CAMERA",
        box=BoundingBox(10, 10, 40, 40),
        score=0.82,
        source="hourglass",
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=False
    ), patch(
        "pipeline.merge_evaluator._rank_keys_on_crop",
        return_value=[
            ("DATA POLE (30TC-4**V)", 0.90),
            ("NETWORK CAMERA", 0.70),
        ],
    ):
        evaluated, summary = evaluate_merged_detections(
            roi_image=roi,
            legend=legend,
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].detection.symbol == "NETWORK CAMERA"
    assert evaluated[0].mapped_from is None
    assert summary["remapped"] == 0


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
        score=0.62,
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
        score=0.70,
        source="hourglass",
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
        "pipeline.config.MERGE_EVAL_MAX_JUDGE_CALLS", 12
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


def test_evaluator_keeps_camera_when_judge_quota_fails():
    roi = Image.new("RGB", (120, 120), (255, 255, 255))
    det = SymbolDetection(
        symbol="NETWORK CAMERA",
        box=BoundingBox(10, 10, 40, 40),
        score=0.70,
        source="hourglass",
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=True
    ), patch(
        "pipeline.config.MERGE_EVAL_MAX_JUDGE_CALLS", 12
    ), patch(
        "pipeline.merge_evaluator.score_glyph_on_crop", return_value=0.0
    ), patch(
        "pipeline.llm_verify.judge_detection_crop",
        return_value={"ok": False, "error": "HTTP Error 429: Too Many Requests"},
    ):
        evaluated, _summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].verdict == "accepted"


def test_suppress_hash_on_camera():
    from pipeline.cv.nms import suppress_hash_on_devices

    camera = SymbolDetection(
        symbol="NETWORK CAMERA",
        box=BoundingBox(10, 10, 40, 30),
        score=0.82,
        source="hourglass",
    )
    drop = SymbolDetection(
        symbol="#",
        box=BoundingBox(12, 12, 28, 28),
        score=0.85,
        source="triangle",
    )
    kept = suppress_hash_on_devices([camera, drop])
    assert [d.symbol for d in kept] == ["NETWORK CAMERA"]


def test_suppress_hash_keeps_jacks_beside_pole():
    from pipeline.cv.nms import suppress_hash_on_devices

    pole = SymbolDetection(
        symbol="DATA POLE (30TC-4**V)",
        box=BoundingBox(100, 100, 140, 140),
        score=0.80,
        source="bowtie",
    )
    drop = SymbolDetection(
        symbol="#",
        box=BoundingBox(40, 110, 70, 130),
        score=0.85,
        source="triangle",
        qty=4,
    )
    kept = suppress_hash_on_devices([pole, drop])
    assert sorted(d.source for d in kept) == ["bowtie", "triangle"]


def test_suppress_overlapping_camera_and_pole():
    from pipeline.cv.nms import suppress_overlapping_devices

    camera = SymbolDetection(
        symbol="NETWORK CAMERA",
        box=BoundingBox(10, 10, 40, 30),
        score=0.82,
        source="hourglass",
    )
    pole = SymbolDetection(
        symbol="DATA POLE",
        box=BoundingBox(12, 12, 38, 32),
        score=0.80,
        source="bowtie",
    )
    kept = suppress_overlapping_devices([camera, pole])
    assert [d.source for d in kept] == ["bowtie"]


def test_evaluator_keeps_bowtie_when_judge_unavailable():
    roi = Image.new("RGB", (120, 120), (255, 255, 255))
    det = SymbolDetection(
        symbol="DATA POLE",
        box=BoundingBox(10, 10, 40, 40),
        score=0.80,
        source="bowtie",
    )
    with patch("pipeline.config.MERGE_EVAL_ENABLED", True), patch(
        "pipeline.config.llm_enabled", return_value=True
    ), patch(
        "pipeline.merge_evaluator.score_glyph_on_crop", return_value=0.0
    ), patch(
        "pipeline.llm_verify.judge_detection_crop",
        return_value={"ok": False, "error": "HTTP Error 429: Too Many Requests"},
    ):
        evaluated, _summary = evaluate_merged_detections(
            roi_image=roi,
            legend=_legend_with_glyph(),
            glyph_dir=None,
            detections=[det],
        )
    assert evaluated[0].verdict == "accepted"


def test_suppress_stub_on_triangle():
    from pipeline.cv.nms import suppress_stub_on_devices

    stub = SymbolDetection(
        symbol="CONDUIT STUB",
        box=BoundingBox(1888, 1087, 1915, 1118),
        score=0.78,
        source="stub",
    )
    drop = SymbolDetection(
        symbol="#",
        box=BoundingBox(1888, 1087, 1915, 1118),
        score=0.85,
        source="triangle",
    )
    kept = suppress_stub_on_devices([stub, drop])
    assert [d.source for d in kept] == ["triangle"]
