from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.cv.tag_match import match_span_to_tag, tagged_legend_keys


def _legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(description="DATA RACK", symbol="R"),
            SymbolEntry(description="ACCESS POINT", symbol="AP"),
            SymbolEntry(description="DATA LINK", symbol="#"),
        ]
    )


def test_match_span_rejects_partial_tags():
    tags = tagged_legend_keys(_legend())
    assert match_span_to_tag("AP", tags) == "AP"
    assert match_span_to_tag("AP-1", tags) == "AP"
    assert match_span_to_tag("APPROVED", tags) is None
    assert match_span_to_tag("R", tags) == "R"
