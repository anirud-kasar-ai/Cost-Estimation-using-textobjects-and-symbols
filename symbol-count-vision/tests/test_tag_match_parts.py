from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.cv.tag_match import (
    count_key,
    match_span_to_text_key,
    text_legend_keys,
)
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo


def _raceway_legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(
                description="SURFACE RACEWAY",
                mfg_model="WIREMOLD",
                part_number="WM2300",
            ),
            SymbolEntry(
                description="SURFACE RACEWAY",
                mfg_model="WIREMOLD",
                part_number="WM5400",
            ),
            SymbolEntry(symbol="G", description="GROUND BOX"),
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK, CAT6A JACK / CABLE"),
        ]
    )


def test_count_key_includes_part_for_raceway():
    legend = _raceway_legend()
    keys = [count_key(e) for e in legend.entries]
    assert "SURFACE RACEWAY (WM2300)" in keys
    assert "SURFACE RACEWAY (WM5400)" in keys
    assert "G" in keys
    assert "#" in keys


def test_text_legend_keys_map_2300_to_wm2300():
    keys = text_legend_keys(_raceway_legend())
    alias_map = {alias: display for alias, display in keys}
    assert alias_map["2300"] == "SURFACE RACEWAY (WM2300)"
    assert alias_map["WM2300"] == "SURFACE RACEWAY (WM2300)"
    assert alias_map["5400"] == "SURFACE RACEWAY (WM5400)"
    assert alias_map["G"] == "G"
    assert "#" not in alias_map  # drop marks own '#'


def test_match_span_2300_dash():
    keys = text_legend_keys(_raceway_legend())
    assert match_span_to_text_key("2300-", keys, size=14) == "SURFACE RACEWAY (WM2300)"
    assert match_span_to_text_key("2300", keys, size=14) == "SURFACE RACEWAY (WM2300)"
    assert match_span_to_text_key("WM5400", keys, size=14) == "SURFACE RACEWAY (WM5400)"
