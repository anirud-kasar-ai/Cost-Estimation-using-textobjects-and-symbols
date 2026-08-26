from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.ocr_tags import detect_tags_from_vision_ocr
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.llm_verify import architect_ocr_prompt_spec, parse_crop_text_with_llm


def test_architect_ocr_prompt_spec_matches_user_schema():
    spec = architect_ocr_prompt_spec(expected_context="Equipment Tags")
    assert "licensed senior architect" in spec["system_instruction"]
    assert "CAD/BIM" in spec["system_instruction"]
    assert spec["inputs"]["expected_context"] == "Equipment Tags"
    assert len(spec["extraction_rules"]) == 10
    assert "ALPHABETIC TAGS" in spec["extraction_rules"][4]
    assert "NUMERIC LABELS" in spec["extraction_rules"][5]
    assert "DENSE OVERLAP" in spec["extraction_rules"][9]
    assert "parsed_entities" in spec["response_format"]
    assert "equipment_tags" in spec["response_format"]["parsed_entities"]
    assert "qty_digits" in spec["response_format"]["parsed_entities"]
    assert "text_entities" in spec["response_format"]


def test_count_verify_prompt_covers_alpha_and_numeric():
    from pipeline.llm_verify import _build_prompt

    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
            SymbolEntry(symbol="J", description="JUNCTION BOX"),
            SymbolEntry(symbol="", description="SURFACE RACEWAY", part_number="WM2300"),
        ]
    )
    prompt = _build_prompt(legend, {"AP": 0, "J": 0}, linear_keys=set())
    assert "ALPHABETIC TAGS" in prompt
    assert "NUMERIC LABELS" in prompt
    assert "2300" in prompt
    assert "expert CAD" in prompt.lower() or "CAD bid set" in prompt


def test_parse_crop_text_with_llm_mocked():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    mock_response = {
        "raw_text": "AP J 2300",
        "parsed_entities": {
            "room_labels": ["A-6E"],
            "data_drop_tags": [],
            "raceway_conduit_notes": ["2300"],
            "general_keynotes": ["AP"],
        },
        "text_entities": [
            {
                "text": "AP",
                "category": "equipment_tag",
                "bbox_rel": {"x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6},
                "confidence": 0.92,
            }
        ],
        "confidence": 0.9,
        "reasoning": "Clear upright text.",
    }
    with patch("pipeline.llm_verify.config.vision_ocr_enabled", return_value=True), patch(
        "pipeline.llm_verify._call_vision_json",
        return_value=(__import__("json").dumps(mock_response), None),
    ):
        result = parse_crop_text_with_llm(img, expected_context="Equipment Tags")
    assert result["ok"] is True
    assert result["parsed_entities"]["general_keynotes"] == ["AP"]


def test_detect_tags_from_vision_ocr_maps_to_legend():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
            SymbolEntry(
                symbol="",
                description="SURFACE RACEWAY",
                part_number="WM2300",
            ),
        ]
    )
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    mock_parse = {
        "ok": True,
        "parsed_entities": {},
        "text_entities": [
            {
                "text": "AP",
                "category": "equipment_tag",
                "bbox_rel": {"x1": 0.4, "y1": 0.4, "x2": 0.5, "y2": 0.5},
                "confidence": 0.9,
            }
        ],
    }
    with patch("pipeline.config.vision_ocr_enabled", return_value=True), patch(
        "pipeline.llm_verify.parse_crop_text_with_llm", return_value=mock_parse
    ):
        hits = detect_tags_from_vision_ocr(img, legend)
    assert len(hits) == 1
    assert hits[0].symbol == "AP"
    assert hits[0].source == "vision_ocr"
