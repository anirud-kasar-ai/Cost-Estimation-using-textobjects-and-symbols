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
from pipeline.llm_verify import _extract_json, verify_counts_with_llm
from pipeline.symbol_count import count_symbols_in_image_cv


def _legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="R", description="DATA RACK"),
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
        ]
    )


def test_extract_json_handles_markdown_fences():
    assert _extract_json('```json\n{"counts": {"R": 2}}\n```') == {"counts": {"R": 2}}
    assert _extract_json('noise {"counts": {"R": 1}} trailing') == {"counts": {"R": 1}}
    assert _extract_json("not json") is None


def test_verify_counts_filters_to_legend_keys():
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"counts": {"R": 3, "AP": "2", "BOGUS": 9}, "notes": "ok"}'
                }
            }
        ]
    }
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    with patch("pipeline.llm_verify._call_groq", return_value=response), patch(
        "pipeline.llm_verify.config.llm_enabled", return_value=True
    ), patch("pipeline.llm_verify.config.llm_provider", return_value="groq"):
        result = verify_counts_with_llm(img, _legend(), {"R": 1})

    assert result["ok"] is True
    assert result["counts"] == {"R": 3, "AP": 2}
    assert result["notes"] == "ok"
    assert result["provider"] == "groq"


def test_verify_counts_gemini_provider():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"counts": {"R": 5, "AP": 1}, "notes": "gemini ok"}'}
                    ]
                }
            }
        ]
    }
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    with patch("pipeline.llm_verify._call_gemini", return_value=response), patch(
        "pipeline.llm_verify.config.llm_enabled", return_value=True
    ), patch("pipeline.llm_verify.config.llm_provider", return_value="gemini"), patch(
        "pipeline.llm_verify.config.llm_model", return_value="gemini-3.6-flash"
    ):
        result = verify_counts_with_llm(img, _legend(), {"R": 1})

    assert result["ok"] is True
    assert result["counts"] == {"R": 5, "AP": 1}
    assert result["provider"] == "gemini"
    assert result["notes"] == "gemini ok"


def test_count_symbols_merges_llm_counts_when_cv_has_marks():
    from pipeline.cv.nms import BoundingBox, SymbolDetection

    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    llm_result = {
        "ok": True,
        "counts": {"R": 4},
        "model": "test-model",
        "notes": None,
        "error": None,
    }
    cv_det = SymbolDetection(
        symbol="R",
        box=BoundingBox(1, 1, 10, 10),
        score=0.9,
        source="ocr",
    )
    meta = {
        "linear_keys": set(),
        "hash_key": None,
        "camera_key": None,
        "glyphs_available": False,
    }
    with patch("pipeline.symbol_count.config.llm_enabled", return_value=True), patch(
        "pipeline.llm_verify.verify_counts_with_llm", return_value=llm_result
    ), patch("pipeline.symbol_count.tesseract_available", return_value=False), patch(
        "pipeline.symbol_count.detect_symbols_raw",
        return_value=([cv_det], [], meta),
    ):
        payload = count_symbols_in_image_cv(
            image=img, legend=_legend(), glyph_dir=None, symbol_file_suffix=".json"
        )

    assert payload["method"] == "cv_plus_llm_verify"
    assert payload["counts"]["R"] == 4
    assert payload["llm"]["used"] is True


def test_count_symbols_allows_llm_to_fill_cv_misses():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    llm_result = {
        "ok": True,
        "counts": {"R": 4},
        "model": "test-model",
        "notes": None,
        "error": None,
    }
    with patch("pipeline.symbol_count.config.llm_enabled", return_value=True), patch(
        "pipeline.symbol_count.config.MERGE_EVAL_ENABLED", False
    ), patch(
        "pipeline.llm_verify.verify_counts_with_llm", return_value=llm_result
    ), patch("pipeline.symbol_count.tesseract_available", return_value=False), patch(
        "pipeline.symbol_count.config.vision_ocr_enabled", return_value=False
    ):
        payload = count_symbols_in_image_cv(
            image=img, legend=_legend(), glyph_dir=None, symbol_file_suffix=".json"
        )

    assert payload["method"] == "cv_plus_llm_verify"
    assert payload["counts"].get("R", 0) == 4
    assert any("Vision filled CV misses" in n for n in payload.get("notes") or [])


def test_cross_source_dedupes_ocr_and_vision():
    from pipeline.cv.nms import suppress_cross_source_duplicates

    ocr = SymbolDetection(
        symbol="AP",
        box=BoundingBox(10, 10, 30, 30),
        score=0.9,
        source="ocr",
    )
    vision = SymbolDetection(
        symbol="AP",
        box=BoundingBox(12, 11, 32, 31),
        score=0.8,
        source="vision_ocr",
    )
    kept = suppress_cross_source_duplicates([ocr, vision])
    assert len(kept) == 1
    assert kept[0].source == "ocr"


def test_overlap_prompt_injected_into_verify():
    from pipeline.symbol_count import SYMBOL_COUNT_OVERLAP_PROMPT
    from pipeline.llm_verify import _build_prompt

    prompt = _build_prompt(
        _legend(),
        {"AP": 1},
        extra_rules=[SYMBOL_COUNT_OVERLAP_PROMPT],
        detection_hints=[{"symbol": "AP", "source": "ocr", "qty": 1, "cx": 10, "cy": 10}],
    )
    assert "OVERLAP / DENSE ZOOM" in prompt
    assert "CV detection centers" in prompt
    assert "UNIQUE" in prompt.upper() or "unique" in prompt.lower()


def test_cad_construction_knowledge_injected():
    from pipeline.symbol_count import CAD_CONSTRUCTION_SYMBOL_KNOWLEDGE
    from pipeline.llm_verify import _SYSTEM_PROMPT, _build_prompt

    prompt = _build_prompt(
        _legend(),
        {"#": 0},
        extra_rules=[CAD_CONSTRUCTION_SYMBOL_KNOWLEDGE],
    )
    assert "CONSTRUCTION CAD" in prompt
    assert "DATA PERMANENT LINK" in prompt
    assert "5400" in prompt
    assert "direction" in _SYSTEM_PROMPT.lower()


def test_count_symbols_keeps_cv_on_llm_failure():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    llm_result = {
        "ok": False,
        "counts": None,
        "model": "test-model",
        "notes": None,
        "error": "boom",
    }
    with patch("pipeline.symbol_count.config.llm_enabled", return_value=True), patch(
        "pipeline.llm_verify.verify_counts_with_llm", return_value=llm_result
    ), patch("pipeline.symbol_count.tesseract_available", return_value=False):
        payload = count_symbols_in_image_cv(
            image=img, legend=_legend(), glyph_dir=None, symbol_file_suffix=".json"
        )

    assert payload["method"] == "single_image_cv"
    assert any("boom" in n for n in payload["notes"])
