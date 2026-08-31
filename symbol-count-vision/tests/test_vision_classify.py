from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.classify.vision_classify import _parse_key_choice, classify_ambiguous, classify_one
from pipeline.cv.nms import BoundingBox
from pipeline.detect.candidates import Candidate
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.taxonomy.builder import taxonomy_from_legend


def test_parse_key_choice_ignores_bbox_payload():
    raw = json.dumps({"key": "AP", "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}})
    assert _parse_key_choice(raw, {"AP", "#"}) == "AP"
    raw_none = json.dumps({"key": "none"})
    assert _parse_key_choice(raw_none, {"AP", "#"}) == "none"


def _png(color=(10, 20, 30)):
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", (24, 24), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_classify_one_never_writes_a_new_box():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK",
                symbol_image_png=_png((1, 1, 1)),
            ),
            SymbolEntry(
                symbol="AP",
                description="ACCESS POINT",
                symbol_image_png=_png((2, 2, 2)),
            ),
        ]
    )
    tax = taxonomy_from_legend(legend)
    box = BoundingBox(10, 10, 40, 40)
    cand = Candidate(
        tile_id="t",
        bbox=box,
        shape_class_guess="triangle_like",
        status="ambiguous",
        candidate_keys=["#", "AP"],
        has_enclosing_circle=True,
    )
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    with patch("pipeline.config.llm_enabled", return_value=True), patch(
        "pipeline.classify.vision_classify.config.llm_enabled", return_value=True
    ), patch(
        "pipeline.llm_verify._call_vision_json",
        return_value=(json.dumps({"key": "AP", "x1": 9, "y1": 9}), None),
    ):
        out = classify_one(img, cand, tax, legend=legend)
    assert out.resolved_key == "AP"
    assert out.bbox == box


def test_classify_fails_toward_cv_on_error():
    legend = SymbolTableInfo(
        entries=[SymbolEntry(symbol="#", description="DATA PERMANENT LINK")]
    )
    tax = taxonomy_from_legend(legend)
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(1, 1, 10, 10),
        shape_class_guess="triangle_like",
        status="ambiguous",
        candidate_keys=["#"],
    )
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    with patch("pipeline.classify.vision_classify.config.llm_enabled", return_value=True), patch(
        "pipeline.llm_verify._call_vision_json",
        return_value=(None, "timeout"),
    ):
        out = classify_one(img, cand, tax)
    assert out.classify_source == "cv_fallback"
    assert out.bbox.x1 == 1


def test_classify_one_attaches_legend_glyphs():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK",
                symbol_image_png=_png((30, 30, 30)),
            ),
            SymbolEntry(
                symbol="AP",
                description="ACCESS POINT",
                symbol_image_png=_png((40, 40, 40)),
            ),
        ]
    )
    tax = taxonomy_from_legend(legend)
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(10, 10, 40, 40),
        shape_class_guess="triangle_like",
        status="ambiguous",
        candidate_keys=["#", "AP"],
    )
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    captured: dict = {}

    def fake_vision(*, prompt, image_b64, system_prompt, max_tokens=256, reference_images=None):
        captured["refs"] = [k for k, _ in (reference_images or [])]
        captured["prompt"] = prompt
        return json.dumps({"key": "#"}), None

    with patch("pipeline.classify.vision_classify.config.llm_enabled", return_value=True), patch(
        "pipeline.llm_verify._call_vision_json",
        side_effect=fake_vision,
    ):
        out = classify_one(img, cand, tax, legend=legend)

    assert out.resolved_key == "#"
    assert captured["refs"][:2] == ["#", "AP"]
    assert "legend glyph artwork" in captured["prompt"].lower()


def test_classify_ambiguous_skips_single_key_and_caps_calls():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
        ]
    )
    tax = taxonomy_from_legend(legend)
    single = Candidate(
        tile_id="t",
        bbox=BoundingBox(1, 1, 8, 8),
        shape_class_guess="triangle_like",
        status="ambiguous",
        candidate_keys=["#"],
        score=0.9,
    )
    lookalikes = [
        Candidate(
            tile_id="t",
            bbox=BoundingBox(10 + i, 10, 20 + i, 20),
            shape_class_guess="triangle_like",
            status="ambiguous",
            candidate_keys=["#", "AP"],
            score=float(i),
        )
        for i in range(4)
    ]
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    calls = {"n": 0}

    def fake_vision(**kwargs):
        calls["n"] += 1
        return json.dumps({"key": "#"}), None

    with patch("pipeline.classify.vision_classify.config.llm_enabled", return_value=True), patch(
        "pipeline.classify.vision_classify.config.VISION_CLASSIFY_MAX_PER_TILE", 2
    ), patch(
        "pipeline.llm_verify._call_vision_json",
        side_effect=fake_vision,
    ):
        out = classify_ambiguous(img, [single, *lookalikes], tax, legend=legend)

    assert calls["n"] == 2
    assert out[0].classify_source == "cv_fallback"
    assert sum(1 for c in out if c.classify_source == "vision_classify") == 2
    assert sum(1 for c in out if c.classify_source == "cv_fallback") == 3


def test_classify_ambiguous_stops_after_quota():
    from pipeline.llm_verify import reset_gemini_quota, _trip_gemini_quota

    reset_gemini_quota()
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
        ]
    )
    tax = taxonomy_from_legend(legend)
    lookalikes = [
        Candidate(
            tile_id="t",
            bbox=BoundingBox(10 + i, 10, 20 + i, 20),
            shape_class_guess="triangle_like",
            status="ambiguous",
            candidate_keys=["#", "AP"],
            score=float(10 - i),
        )
        for i in range(4)
    ]
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    calls = {"n": 0}

    def fake_vision(**kwargs):
        calls["n"] += 1
        _trip_gemini_quota()
        return None, "HTTP 429 quota"

    with patch("pipeline.classify.vision_classify.config.llm_enabled", return_value=True), patch(
        "pipeline.classify.vision_classify.config.VISION_CLASSIFY_MAX_PER_TILE", 6
    ), patch(
        "pipeline.llm_verify._call_vision_json",
        side_effect=fake_vision,
    ):
        out = classify_ambiguous(img, lookalikes, tax, legend=legend)

    assert calls["n"] == 1
    assert sum(1 for c in out if c.classify_source == "cv_fallback") == 4
    reset_gemini_quota()
