from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.taxonomy.builder import taxonomy_from_legend
from pipeline.taxonomy.schema import taxonomy_rules_prompt_block


def _legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
            SymbolEntry(symbol="AP", description="INTERIOR WIRELESS ACCESS POINT"),
            SymbolEntry(symbol="", description="NETWORK CAMERA"),
            SymbolEntry(symbol="", description="DATA POLE", part_number="30TC-4**V"),
            SymbolEntry(symbol="J", description="JUNCTION BOX"),
            SymbolEntry(symbol="", description="J-HOOK (SINGLE/STACKED)"),
            SymbolEntry(symbol="", description="CONDUIT STUB"),
            SymbolEntry(symbol="", description="SURFACE RACEWAY", part_number="WM5400"),
            SymbolEntry(symbol="", description="CONDUIT (EMT & GRC)"),
        ]
    )


def test_infer_lookalike_rules_live_only_on_taxonomy():
    tax = taxonomy_from_legend(_legend())
    by = {e.key: e for e in tax.entries}
    assert by["#"].shape_class == "filled_triangle"
    assert by["#"].count_semantics == "qty_expand"
    assert "hourglass" in by["#"].context_rule.excludes_if_matches
    assert by["AP"].context_rule.requires_enclosing_shape == "circle"
    assert by["NETWORK CAMERA"].shape_class == "hourglass"
    assert by["NETWORK CAMERA"].context_rule.requires_opposing_shape == "triangle"
    assert by["DATA POLE (30TC-4**V)"].shape_class == "bowtie"
    assert by["DATA POLE (30TC-4**V)"].context_rule.requires_enclosing_square is True
    hook = by["J-HOOK (SINGLE/STACKED)"]
    assert hook.shape_class == "jhook_run"
    assert hook.context_rule.requires_collinear_min == 2
    assert hook.context_rule.collinear_of_tag == "J"
    assert by["CONDUIT STUB"].context_rule.middle_bar_longer is True
    assert by["SURFACE RACEWAY (WM5400)"].count_semantics == "discrete"
    assert by["CONDUIT (EMT & GRC)"].count_semantics == "linear_suppressed"


def test_prompt_block_is_json_not_prose_table():
    block = taxonomy_rules_prompt_block(taxonomy_from_legend(_legend()))
    assert "TAXONOMY_CONTEXT_RULES" in block
    assert "requires_enclosing_shape" in block
    assert "LOOK-ALIKE GLYPHS" not in block
