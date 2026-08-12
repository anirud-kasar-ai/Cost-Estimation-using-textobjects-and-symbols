"""Tests for technical symbol legend extraction and PDF generation."""

from __future__ import annotations

import io
import re
from pathlib import Path

import fitz
from PIL import Image as PILImage

from ml.symbol_table_extractor import (
    SymbolEntry,
    SymbolTableInfo,
    _parse_legend_page,
    _tokenize_tech_notes,
    extract_symbol_table_from_pdf,
)
from ml.symbol_table_pdf import (
    generate_technical_symbol_pdf,
    technical_symbol_pdf_filename,
)

TECH_LEGEND = """
TECHNOLOGY SYMBOL LEGEND:
ALL EQUIPMENT AND MATERIALS ARE CONTRACTOR FURNISHED, INSTALLED AND CONFIGURED (UNO)
MFG/MODEL
UNDERGROUND CONDUIT
EXISTING
CONDUIT (EMT & GRC)
COMMERCIAL
GENERIC
DATA RACK
DATA PERMANENT LINK, CAT6A JACK / CABLE
# = QTY., NO # = 1
SEE 27 10 00
R
PANDUIT
N/A
GROUND BOX
G
JUNCTION BOX
J
CONDUIT STUB
SURFACE RACEWAY
WM2300
WIREMOLD
NETWORK VIDEO RECORDER
NVR
SIGNAL TERMINAL CABINET
STC
CAT6A DATA DROP LOCATION (QTY = 1)
- IP CLOCK/SPEAKER/IP MODULE COMBO BOX BOGEN
CONSULTANTS
CARROLL ENGINEERING
"""

GARLAND_LEGEND = """
SYMBOL LEGEND
1.
CURB DETAIL, DTL C/4
2.
EMBEDDED EDGE METAL, DTL D/4
3. WALL FLASHING, DTL E/4
4.
PIPE PENETRATION, DTL F/4
SHEET INDEX
SHT 1 - COVER
"""


class TestTechnicalSymbolFilename:
    def test_filename_convention(self) -> None:
        assert technical_symbol_pdf_filename("abc.pdf") == "abc technical symbol.pdf"
        assert technical_symbol_pdf_filename("1961BIDDrawings.PDF") == (
            "1961BIDDrawings technical symbol.pdf"
        )


class TestParseLegendFixtures:
    def test_technology_symbol_legend_entries(self) -> None:
        lines = [line.strip() for line in TECH_LEGEND.splitlines() if line.strip()]
        entries = _parse_legend_page(lines, page_number=2)
        descriptions = " | ".join(e.description.upper() for e in entries)
        assert any("JUNCTION BOX" in e.description.upper() for e in entries)
        assert any("DATA RACK" in e.description.upper() for e in entries)
        assert any(e.symbol == "G" for e in entries)
        assert any(e.symbol == "J" for e in entries)
        assert "JUNCTION BOX" in descriptions
        assert len(entries) >= 5

    def test_garland_numbered_symbol_legend(self) -> None:
        lines = [line.strip() for line in GARLAND_LEGEND.splitlines() if line.strip()]
        entries = _parse_legend_page(lines, page_number=3)
        assert len(entries) >= 3
        assert entries[0].symbol == "1"
        assert "CURB DETAIL" in entries[0].description.upper()
        assert any("WALL FLASHING" in e.description.upper() for e in entries)


class TestTechnicalSymbolPdf:
    def test_generates_pdf(self, tmp_path: Path) -> None:
        info = SymbolTableInfo(
            entries=[
                SymbolEntry(
                    symbol="J",
                    description="JUNCTION BOX",
                    mfg_model="COMMERCIAL GENERIC",
                    part_number="N/A",
                    notes=["GREY = EXISTING"],
                    source_page=2,
                    legend_title="TECHNOLOGY SYMBOL LEGEND",
                    symbol_image_png=_tiny_png(),
                ),
                SymbolEntry(
                    symbol="1",
                    description="CURB DETAIL, DTL C/4",
                    source_page=3,
                    legend_title="SYMBOL LEGEND",
                    symbol_image_png=_tiny_png(),
                ),
            ],
            legend_pages=[2, 3],
            total_pages=10,
        )
        out = tmp_path / "demo technical symbol.pdf"
        generate_technical_symbol_pdf(info, "demo.pdf", out)
        assert out.exists() and out.stat().st_size > 400
        with fitz.open(str(out)) as doc:
            text = "".join(page.get_text() for page in doc)
            # Numbered legend embeds the glyph image (not only the index digit).
            assert any(page.get_images() for page in doc)
        assert "TECHNICAL SYMBOL SUMMARY" in text.upper()
        assert "JUNCTION BOX" in text.upper()
        assert "CURB DETAIL" in text.upper()
        assert "MFG/MODEL" in text.upper()
        compact = " ".join(text.upper().split())
        assert "COMMERCIAL" in compact and "GENERIC" in compact
        assert "JUNCTION BOX" in compact


def _tiny_png() -> bytes:
    """1x1 PNG for embedding tests."""
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


class TestTechnologyLegend1948:
    def test_1948_spatial_order_and_pairing(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data\1948BIDDrawings.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        tech = [
            e
            for e in info.entries
            if "TECHNOLOGY" in e.legend_title.upper()
        ]
        assert len(tech) >= 25
        descs = [e.description.upper() for e in tech]
        assert descs[0].startswith("DATA RACK")
        assert "GROUND BOX" in descs[1]
        assert "JUNCTION BOX" in descs[2]
        assert "UNDERGROUND CONDUIT" in descs[3]
        by_sym = {e.symbol: e for e in tech if e.symbol}
        assert by_sym["G"].description.upper().startswith("GROUND BOX")
        assert by_sym["J"].description.upper().startswith("JUNCTION BOX")
        assert by_sym["NVR"].description.upper().startswith("NETWORK VIDEO")
        assert by_sym["STC"].description.upper().startswith("SIGNAL TERMINAL")
        assert by_sym["TGB"].description.upper().startswith("GROUND BUS")
        raceways = [
            e for e in tech if e.description.upper().startswith("SURFACE RACEWAY")
        ]
        parts = {(e.part_number or "").upper() for e in raceways}
        assert {"WM2300", "WM5400", "WM5500"} <= parts
        assert any("LADDER RACK" in d for d in descs)
        with_img = sum(1 for e in tech if e.symbol_image_png)
        assert with_img >= 20
        # Boxed tags must appear even when CAD cells do not rasterize.
        assert by_sym["TGB"].symbol_image_png
        assert by_sym["STC"].symbol_image_png
        assert by_sym["R"].description.upper().startswith("DATA RACK")
        # Notes are a list of distinct callouts — not one space-joined blob.
        noted = [e for e in tech if e.notes]
        assert noted
        assert all(isinstance(e.notes, list) for e in noted)
        bubble_rows = [
            e for e in noted if any(re.fullmatch(r"T-\d+", n) for n in (e.notes or []))
        ]
        if bubble_rows:
            for e in bubble_rows:
                assert not any(re.fullmatch(r"\d{1,2}", n) for n in (e.notes or []))
                # Distinct T-* items stay separate list entries.
                t_refs = [n for n in (e.notes or []) if re.fullmatch(r"T-\d+", n)]
                assert len(t_refs) == len(set(t_refs))
        # PDF columns preserve order / MFG.
        out = Path("tmp_1948_tech_test.pdf")
        try:
            generate_technical_symbol_pdf(info, pdf.name, out)
            with fitz.open(str(out)) as doc:
                text = "".join(page.get_text() for page in doc).upper()
            assert "DATA RACK" in text and "PANDUIT" in text
            assert text.index("DATA RACK") < text.index("GROUND BOX")
            assert text.index("GROUND BOX") < text.index("JUNCTION BOX")
            if bubble_rows:
                assert "T-800" in text or "T-801" in text
        finally:
            if out.exists():
                out.unlink()


class TestTokenizeTechNotes:
    def test_splits_bubbles_and_merges_phrases(self) -> None:
        result = _tokenize_tech_notes(
            ["2", "12", "SEE", "SINGLE", "LINE", "T-800", "T-801", "T-802"]
        )
        assert result == ["SEE SINGLE LINE", "T-800", "T-801", "T-802"]

    def test_dedupes_repeated_bubbles_and_grey_phrase(self) -> None:
        result = _tokenize_tech_notes(
            ["13", "15", "11", "12", "14", "T-800", "T-800", "T-800", "GREY", "=", "EXISTING"]
        )
        assert result == ["T-800", "GREY = EXISTING"]

    def test_grey_phrase_with_interleaved_bubble(self) -> None:
        result = _tokenize_tech_notes(["GREY", "=", "T-800", "EXISTING"])
        assert result == ["GREY = EXISTING", "T-800"]
        result2 = _tokenize_tech_notes(["GREY", "=", "(E)", "T-801"])
        assert result2 == ["GREY = EXISTING", "T-801"]

    def test_empty_and_na(self) -> None:
        assert _tokenize_tech_notes([]) is None
        assert _tokenize_tech_notes(["N/A", "n/a."]) is None


class TestSymbolsAndAbbreviations1945:
    def test_1945_drafting_symbols_and_abbreviations(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1945BIDDrawingsRVES.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        assert 1 in info.legend_pages
        titles = " | ".join(e.legend_title.upper() for e in info.entries)
        assert "DRAFTING" in titles
        assert "ABBREVIATIONS" in titles
        descs = " ".join(e.description.upper() for e in info.entries)
        assert "DATUM" in descs or "CONTROL POINT" in descs
        assert "REVISION CLOUD" in descs
        assert "DETAIL VIEW" in descs or "SECTION VIEW" in descs
        abbrs = {
            (e.symbol or "").upper(): e.description.upper()
            for e in info.entries
            if "ABBREVIATION" in e.legend_title.upper()
        }
        assert abbrs.get("AFF") == "ABOVE FINISH FLOOR" or any(
            "ABOVE FINISH FLOOR" in d for d in abbrs.values()
        )
        assert abbrs.get("CMU") == "CONCRETE MASONRY UNIT" or any(
            "CONCRETE MASONRY UNIT" in d for d in abbrs.values()
        )
        assert info.entry_count >= 40
        drafting = [
            e for e in info.entries if "DRAFTING" in e.legend_title.upper()
        ]
        assert sum(1 for e in drafting if e.symbol_image_png) >= 5
        by_desc = {e.description.upper(): e for e in drafting}
        if "REVISION TAG" in by_desc and "PROJECT NORTH ARROW" in by_desc:
            tag = PILImage.open(io.BytesIO(by_desc["REVISION TAG"].symbol_image_png))
            north = PILImage.open(
                io.BytesIO(by_desc["PROJECT NORTH ARROW"].symbol_image_png)
            )
            # Tag must not swallow the tall north-arrow row beneath it.
            assert tag.size[1] < north.size[1]
            assert tag.size[1] < 180
        if "COLUMN GRID MARKER" in by_desc and "REVISION CLOUD" in by_desc:
            grid = PILImage.open(
                io.BytesIO(by_desc["COLUMN GRID MARKER"].symbol_image_png)
            )
            cloud = PILImage.open(
                io.BytesIO(by_desc["REVISION CLOUD"].symbol_image_png)
            )
            # Grid crop must stay a single marker — not grid+cloud stacked.
            assert grid.size[1] < 220
            assert cloud.size[1] < 280
            assert grid.size[1] < cloud.size[1] + 40


class TestArchitectPlanLegends:
    def test_1962_floor_and_ceiling_legends_no_duplicates(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1962BIDMDESDrawings.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        floor = [
            e for e in info.entries if e.legend_title.upper() == "LEGEND - FLOOR PLAN"
        ]
        ceiling = [
            e for e in info.entries if e.legend_title.upper() == "LEGEND - CEILING"
        ]
        assert len(floor) == 5
        floor_descs = [e.description.upper() for e in floor]
        assert any("ROOM NAME" == d for d in floor_descs)
        assert any("ROOM NUMBER" in d for d in floor_descs)
        assert any("ROLLER SHADES" in d and "TAG" in d for d in floor_descs)
        assert any("SINGLE OR DUAL ROLLER" in d for d in floor_descs)
        assert all(e.symbol_image_png for e in floor)

        assert len(ceiling) >= 12
        ceil_blob = " ".join(e.description.upper() for e in ceiling)
        assert "CEILING HEIGHT" in ceil_blob
        assert "SMOKE DETECTOR" in ceil_blob
        assert "SUPPLY REGISTER" in ceil_blob
        assert "ILLUMINATED EXIT" in ceil_blob
        # Same glyph, two meanings — both rows kept.
        smoke = [e for e in ceiling if "SMOKE DETECTOR" in e.description.upper()]
        assert len(smoke) == 2
        assert all(e.symbol_image_png for e in ceiling)
        # Cross-page legend repeats must not double the list.
        assert len({e.description.upper() for e in floor}) == len(floor)
        assert len({e.description.upper() for e in ceiling}) == len(ceiling)
        # No keynotes / general-notes paragraphs and no floating detail bubbles.
        assert "CUTSHEETS" not in ceil_blob
        assert "2A-922" not in ceil_blob
        assert not any("KEYNOTE" in e.description.upper() for e in floor + ceiling)


class TestRealDataIntegration:
    def test_1961_technology_legend_when_present(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data\1961BIDDrawings.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        assert info.total_pages >= 1
        assert info.entry_count >= 5
        assert 2 in info.legend_pages or any(
            "TECHNOLOGY" in e.legend_title.upper() for e in info.entries
        )
        descs = " ".join(e.description.upper() for e in info.entries)
        assert "JUNCTION BOX" in descs or "DATA RACK" in descs

    def test_ayers_symbol_legend_when_present(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1954BIDDrawings-AyersES011325.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        assert info.entry_count >= 3
        descs = " ".join(e.description.upper() for e in info.entries)
        assert "CURB" in descs or "FLASHING" in descs or "DRAIN" in descs


class TestNumberedLegendGlyphCrops:
    def test_shadeland_uses_legend_column_not_scope(self) -> None:
        """SYMBOL LEGEND sits beside a scope list — crops must use the legend column."""
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1952BIDDrawings-Shadeland-Sunrise010625.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        legend = [
            e
            for e in info.entries
            if "SYMBOL" in e.legend_title.upper()
            and "TECHNOLOGY" not in e.legend_title.upper()
        ]
        by_num = {e.symbol: e for e in legend if e.symbol}
        assert "EQUIPMENT SUPPORT" in by_num["1"].description.upper()
        assert "ROOF HATCH" in by_num["5"].description.upper()
        assert "EXHAUST FAN" in by_num["14"].description.upper()
        # Scope-of-work verbs must not appear as legend rows.
        descs = " ".join(e.description.upper() for e in legend)
        assert "REMOVE AND DISPOSE" not in descs
        assert "INSTALL 2 PLY" not in descs
        assert "GARLAND COMPANY" not in descs
        for num in ("5", "7", "10", "11", "13", "14", "15"):
            assert by_num[num].symbol_image_png, f"missing glyph for {num}"
            assert len(by_num[num].symbol_image_png) >= 250
        # One row per legend index (cross-page duplicates collapsed).
        assert len(by_num) >= 12
        assert len(legend) <= 16

    def test_glenbrook_crops_glyphs_left_of_numbers(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1974BIDDrawings-GlenbrookMS.pdf"
        )
        if not pdf.exists():
            return
        info = extract_symbol_table_from_pdf(pdf)
        legend = [
            e
            for e in info.entries
            if e.legend_title.upper().startswith("SYMBOL LEGEND")
            or e.legend_title.upper() == "SYMBOL LEGEND"
        ]
        assert len(legend) >= 15
        by_num = {e.symbol: e for e in legend if e.symbol}
        # Rows with CAD glyphs on the sheet.
        for num in ("3", "4", "5", "7", "11", "12", "13", "15", "19"):
            assert by_num[num].symbol_image_png, f"missing glyph for item {num}"
        # Rows with no glyph stay image-empty.
        for num in ("1", "2", "6", "8", "14"):
            assert by_num[num].symbol_image_png is None
        assert "EXHAUST FAN" in by_num["3"].description.upper()
        assert "SKYLIGHT" in by_num["19"].description.upper()

        # Closed box glyphs must keep all four sides (grid-strip used to erase them).
        for num in ("3", "7", "15"):
            img = PILImage.open(io.BytesIO(by_num[num].symbol_image_png)).convert("L")
            mask = img.point(lambda p: 255 if p < 245 else 0)
            bbox = mask.getbbox()
            assert bbox is not None
            left, top, right, bottom = bbox
            width, height = right - left, bottom - top
            assert width >= 20 and height >= 20
            assert 0.75 <= (width / height) <= 1.35, (
                f"item {num} looks clipped: {width}x{height}"
            )

        out = Path("tmp_glenbrook_symbols_test.pdf")
        try:
            generate_technical_symbol_pdf(info, pdf.name, out)
            with fitz.open(str(out)) as doc:
                text = "".join(page.get_text() for page in doc).upper()
                assert any(page.get_images() for page in doc)
                # Numbered glyph rows keep the list index in the Symbol cell.
                assert "3" in text and "19" in text
            assert "EXHAUST FAN" in text and "SKYLIGHT" in text
        finally:
            if out.exists():
                out.unlink()
