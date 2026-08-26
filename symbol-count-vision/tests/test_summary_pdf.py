from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.symbol_legend import parse_symbol_file

SUMMARY_PDF = (
    project_root.parent
    / "dual-pathway-drawing-split"
    / "storage"
    / "technical_symbols"
    / "1948BIDDrawings technical symbol.pdf"
)


TEXT_ONLY_PDF = Path(r"d:\Downloads\1948BIDDrawings technical symbol (1).pdf")


def test_technical_symbol_summary_pdf_extracts_tags_and_glyphs(tmp_path: Path):
    if not SUMMARY_PDF.is_file():
        return

    legend, glyph_dir = parse_symbol_file(SUMMARY_PDF, tmp_path)
    tagged = sum(1 for entry in legend.entries if entry.symbol)
    glyph_pngs = list(glyph_dir.glob("*.png")) if glyph_dir else []

    assert legend.entry_count >= 30
    assert tagged >= 10
    assert len(glyph_pngs) >= 30


def test_text_only_summary_pdf_resolves_glyphs_from_library(tmp_path: Path):
    """A summary PDF without any artwork still gets glyphs via the library."""
    if not (TEXT_ONLY_PDF.is_file() and SUMMARY_PDF.is_file()):
        return

    legend, glyph_dir = parse_symbol_file(TEXT_ONLY_PDF, tmp_path)

    # Tags must land on the right rows (vector row bands).
    by_desc = {e.description: e.symbol for e in legend.entries}
    assert by_desc.get("DATA PERMANENT LINK, CAT6A JACK / CABLE") == "#"
    assert by_desc.get("GROUND BOX") == "G"

    from_library = sum(1 for e in legend.entries if e.glyph_source == "library")
    assert from_library >= 15
    assert glyph_dir is not None
    assert len(list(glyph_dir.glob("*.png"))) >= 15
