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
    assert "LOOK-ALIKE" in prompt
    assert "J-HOOK" in prompt
    assert "CONDUIT STUB" in prompt


def test_judge_prompt_rejects_shared_fragments():
    from pipeline.llm_verify import _build_judge_prompt

    prompt = _build_judge_prompt(
        "#",
        SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
    )
    assert "FULL glyph" in prompt
    assert "LOOK-ALIKE" in prompt
    assert "DATA POLE" in prompt


def test_detect_legend_symbols_enumerates_legend_and_clamps_boxes():
    from io import BytesIO

    import pipeline.llm_verify as llm_verify
    from pipeline.llm_verify import detect_legend_symbols_with_llm

    llm_verify._GLYPH_DESC_CACHE.clear()
    glyph_png = BytesIO()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(glyph_png, format="PNG")
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK",
                symbol_image_png=glyph_png.getvalue(),
            ),
            SymbolEntry(symbol="AP", description="ACCESS POINT"),
        ]
    )
    img = Image.new("RGB", (653, 653), color=(255, 255, 255))
    reply = {
        "detections": [
            {
                "symbol": "#",
                "kind": "drop",
                "qty": 2,
                "confidence": 0.9,
                # Oversized box covering most of the tile — must be clamped.
                "bbox_rel": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95},
            },
            {
                "symbol": "NORTH ARROW",
                "qty": 1,
                "confidence": 0.9,
                "bbox_rel": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            },
        ],
        "notes": "ok",
    }
    captured: dict = {}

    def fake_call(**kwargs):
        import json as _json

        # First call: glyph shape description pass for the uploaded legend.
        if '"glyphs"' in kwargs.get("prompt", ""):
            return _json.dumps({"glyphs": {"#": "small solid filled triangle"}}), None
        captured.update(kwargs)
        return _json.dumps(reply), None

    with patch("pipeline.llm_verify.config.llm_enabled", return_value=True), patch(
        "pipeline.llm_verify._call_vision_json", side_effect=fake_call
    ), patch(
        "pipeline.legend_reference.lookup_descriptions_for_legend", return_value={}
    ):
        result = detect_legend_symbols_with_llm(img, legend)

    assert result["ok"] is True
    prompt = captured["prompt"]
    assert "VALID_SYMBOLS" in prompt
    assert "NEVER report" in prompt
    assert '"name": "#"' in prompt
    # Shape description from the uploaded technical-symbol sheet is injected.
    assert '"glyph": "small solid filled triangle"' in prompt
    assert result.get("glyph_descriptions") == {"#": "small solid filled triangle"}
    assert "KEY COLUMN" in prompt
    assert "ONLY a STANDALONE solid FILLED triangle" in prompt or "ONLY a solid FILLED black triangle" in prompt
    assert "LOOK-ALIKE" in prompt
    assert "J-HOOK" in prompt
    # Legend glyph artwork attached as a visual reference.
    refs = captured.get("reference_images") or []
    assert len(refs) == 1 and refs[0][0] == "#"
    assert result["reference_glyphs_attached"] == 1
    # Bogus non-legend key dropped; oversized box clamped around its center.
    assert len(result["detections"]) == 1
    det = result["detections"][0]
    assert det["symbol"] == "#"
    max_side = 0.15 * 653
    assert (det["x2"] - det["x1"]) <= max_side + 1
    assert (det["y2"] - det["y1"]) <= max_side + 1
    assert result.get("clamped_boxes") == 1


def test_legend_reference_images_skip_linear_and_prioritize_devices():
    from io import BytesIO

    from pipeline.llm_verify import _legend_reference_images

    def png(color):
        buf = BytesIO()
        Image.new("RGB", (24, 24), color=color).save(buf, format="PNG")
        return buf.getvalue()

    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="",
                description="SURFACE RACEWAY",
                part_number="WM5400",
                symbol_image_png=png((0, 0, 0)),
            ),
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK",
                symbol_image_png=png((10, 10, 10)),
            ),
            SymbolEntry(
                symbol="AP",
                description="ACCESS POINT",
                symbol_image_png=png((20, 20, 20)),
            ),
        ]
    )
    refs = _legend_reference_images(legend, set(), max_refs=8)
    labels = [k for k, _ in refs]
    assert labels[0] == "#"
    assert "AP" in labels
    assert all("RACEWAY" not in k for k in labels)


def test_lookalike_glyph_refs_forced_even_if_marked_linear():
    from io import BytesIO

    from pipeline.llm_verify import _legend_reference_images

    def png(color):
        buf = BytesIO()
        Image.new("RGB", (24, 24), color=color).save(buf, format="PNG")
        return buf.getvalue()

    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(
                symbol="",
                description="CONDUIT STUB",
                symbol_image_png=png((1, 1, 1)),
            ),
            SymbolEntry(
                symbol="",
                description="J-HOOK (SINGLE/STACKED)",
                symbol_image_png=png((2, 2, 2)),
            ),
            SymbolEntry(
                symbol="#",
                description="DATA PERMANENT LINK",
                symbol_image_png=png((3, 3, 3)),
            ),
            SymbolEntry(
                symbol="AP",
                description="ACCESS POINT",
                symbol_image_png=png((4, 4, 4)),
            ),
        ]
    )
    linear = {"CONDUIT STUB", "J-HOOK (SINGLE/STACKED)"}
    refs = _legend_reference_images(legend, linear, max_refs=4)
    labels = [k for k, _ in refs]
    assert labels[:4] == ["#", "AP", "CONDUIT STUB", "J-HOOK (SINGLE/STACKED)"]


def test_reference_descriptions_never_cross_between_different_sheets(tmp_path):
    """A stored reference must only apply to legends whose entries provably
    match it (same glyph artwork or identical row) — different projects can
    reuse a key like '#' for a completely different symbol."""
    import json
    from io import BytesIO

    import pipeline.legend_reference as legend_reference
    from pipeline.legend_reference import lookup_descriptions_for_legend

    def png_bytes(color):
        buf = BytesIO()
        Image.new("RGB", (24, 24), color=color).save(buf, format="PNG")
        return buf.getvalue()

    ref_png = png_bytes((0, 0, 0))
    import hashlib

    ref_dir = tmp_path / "refA"
    ref_dir.mkdir()
    (ref_dir / "descriptions.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "key": "#",
                        "tag": "#",
                        "part": None,
                        "description": "DATA PERMANENT LINK",
                        "glyph": "small solid filled triangle",
                        "glyph_sha1": hashlib.sha1(ref_png).hexdigest(),
                    }
                ],
                "glyph_descriptions": {"#": "small solid filled triangle"},
            }
        ),
        encoding="utf-8",
    )

    with patch.object(legend_reference, "REFERENCE_DIR", tmp_path):
        # Same artwork -> matches even without identical description text.
        same = SymbolTableInfo(
            entries=[
                SymbolEntry(
                    symbol="#",
                    description="DATA PERMANENT LINK, CAT6A",
                    symbol_image_png=ref_png,
                )
            ]
        )
        assert lookup_descriptions_for_legend(same) == {
            "#": "small solid filled triangle"
        }
        # Same row metadata, no artwork -> matches via row fingerprint.
        same_row = SymbolTableInfo(
            entries=[SymbolEntry(symbol="#", description="DATA PERMANENT LINK")]
        )
        assert lookup_descriptions_for_legend(same_row) == {
            "#": "small solid filled triangle"
        }
        # Same key but different sheet (other artwork + other meaning) -> no match.
        other = SymbolTableInfo(
            entries=[
                SymbolEntry(
                    symbol="#",
                    description="SPRINKLER HEAD",
                    symbol_image_png=png_bytes((128, 0, 0)),
                )
            ]
        )
        assert lookup_descriptions_for_legend(other) == {}


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
