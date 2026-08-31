from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.classify.context_resolver import resolve_candidates
from pipeline.cv.nms import BoundingBox
from pipeline.detect.candidates import Candidate, propose_candidates
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.taxonomy.builder import taxonomy_from_legend


def _legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
            SymbolEntry(symbol="AP", description="INTERIOR WIRELESS ACCESS POINT"),
            SymbolEntry(symbol="", description="NETWORK CAMERA"),
            SymbolEntry(symbol="", description="DATA POLE", part_number="30TC-4**V"),
            SymbolEntry(symbol="J", description="JUNCTION BOX"),
            SymbolEntry(symbol="", description="J-HOOK (SINGLE/STACKED)"),
        ]
    )


def test_standalone_triangle_resolves_to_hash_not_ap():
    tax = taxonomy_from_legend(_legend())
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(10, 10, 30, 30),
        shape_class_guess="triangle_like",
    )
    out = resolve_candidates([cand], tax)
    assert len(out) == 1
    assert out[0].resolved_key == "#"
    assert out[0].classify_source == "context_rule"


def test_triangle_in_circle_resolves_to_ap_not_hash():
    tax = taxonomy_from_legend(_legend())
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(10, 10, 30, 30),
        shape_class_guess="triangle_like",
        has_enclosing_circle=True,
    )
    out = resolve_candidates([cand], tax)
    assert out[0].resolved_key == "AP"
    assert out[0].resolved_key != "#"


def test_hourglass_pair_is_camera_not_two_drops():
    tax = taxonomy_from_legend(_legend())
    cands = [
        Candidate(
            tile_id="t",
            bbox=BoundingBox(10, 10, 80, 40),
            shape_class_guess="hourglass_like",
            has_opposing_triangle=True,
        ),
        Candidate(
            tile_id="t",
            bbox=BoundingBox(10, 12, 30, 38),
            shape_class_guess="triangle_like",
            has_opposing_triangle=True,
        ),
        Candidate(
            tile_id="t",
            bbox=BoundingBox(60, 12, 80, 38),
            shape_class_guess="triangle_like",
            has_opposing_triangle=True,
        ),
    ]
    out = resolve_candidates(cands, tax)
    keys = [c.resolved_key for c in out]
    assert keys.count("NETWORK CAMERA") == 1
    assert "#" not in keys


def test_bowtie_not_hash():
    tax = taxonomy_from_legend(_legend())
    out = resolve_candidates(
        [
            Candidate(
                tile_id="t",
                bbox=BoundingBox(0, 0, 40, 40),
                shape_class_guess="bowtie_like",
                has_enclosing_square=True,
            ),
            Candidate(
                tile_id="t",
                bbox=BoundingBox(5, 5, 18, 35),
                shape_class_guess="triangle_like",
                has_enclosing_square=True,
            ),
        ],
        tax,
    )
    keys = [c.resolved_key for c in out]
    assert "DATA POLE (30TC-4**V)" in keys
    assert "#" not in keys


def test_collinear_js_become_one_jhook():
    tax = taxonomy_from_legend(_legend())
    cands = [
        Candidate(
            tile_id="t",
            bbox=BoundingBox(10, 20, 22, 36),
            shape_class_guess="letter_tag",
            ocr_text="J",
            extras={"legend_hint": "J"},
        ),
        Candidate(
            tile_id="t",
            bbox=BoundingBox(40, 21, 52, 37),
            shape_class_guess="letter_tag",
            ocr_text="J",
            extras={"legend_hint": "J"},
        ),
        Candidate(
            tile_id="t",
            bbox=BoundingBox(70, 19, 82, 35),
            shape_class_guess="letter_tag",
            ocr_text="J",
            extras={"legend_hint": "J"},
        ),
    ]
    out = resolve_candidates(cands, tax)
    hooks = [c for c in out if c.resolved_key and "HOOK" in c.resolved_key]
    jboxes = [c for c in out if c.resolved_key == "J"]
    assert len(hooks) == 1
    assert jboxes == []


def test_isolated_j_stays_junction_box():
    tax = taxonomy_from_legend(_legend())
    out = resolve_candidates(
        [
            Candidate(
                tile_id="t",
                bbox=BoundingBox(10, 20, 22, 36),
                shape_class_guess="letter_tag",
                ocr_text="J",
                extras={"legend_hint": "J"},
            )
        ],
        tax,
    )
    assert len(out) == 1
    assert out[0].resolved_key == "J"


def test_propose_ap_glyph_is_not_hash_drop():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((170, 170, 230, 230), outline=(0, 0, 0), width=2)
    draw.polygon([(200, 188), (188, 212), (212, 212)], fill=(0, 0, 0))
    legend = _legend()
    tax = taxonomy_from_legend(legend)
    cands = propose_candidates(img, legend, tax, exclude_top_pct=0.0)
    resolved = resolve_candidates(cands, tax)
    hash_hits = [c for c in resolved if c.resolved_key == "#"]
    ap_hits = [c for c in resolved if c.resolved_key == "AP"]
    assert hash_hits == []
    assert len(ap_hits) >= 1
