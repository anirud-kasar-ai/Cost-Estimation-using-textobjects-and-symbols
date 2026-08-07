"""Tests for requirement metadata + scope extraction and PDF generation."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from ml.logo_firm import (
    fill_provider_company_from_logo,
    lookup_firm_from_stamp,
    parse_firm_from_ocr_lines,
)
from ml.requirement_extractor import (
    PageText,
    RequirementInfo,
    ScopeSection,
    extract_pdf_text,
    extract_requirement,
    extract_requirement_from_pdf,
)
from ml.requirement_pdf import generate_requirement_pdf, requirement_pdf_filename, sanitize_requirement


GARLAND_COVER = """
Mt. Diablo Unified School District
AYERS ELEMENTARY ROOF RESTORATION
5120 MYRTLE DR, CONCORD, CA 94521
SHT 1  - COVER SHEET
SHT 2  - SCOPE OF WORK & TYPICAL DETAIL
SHT 3 -  ROOF PLAN
SHT 4 -  DETAILS
SHT 5 -  DETAILS
DATE:
CHK BY:
DWG BY:
PHONE (800) 321-9336 / FAX (216) 641-0633
3800 EAST 91st STREET - CLEVELAND, OHIO 44105-2197
THE GARLAND COMPANY INC
AGENT:
GARLAND
DOUG CLARK
1-13-25
"""

GARLAND_SCOPE = """
LIQUID APPLIED ROOF RESTORATION SCOPE:
1.
REMOVE AND DISPOSE OF ALL VERTICAL FLASHINGS AND DETAIL FLASHINGS.
2.
INSTALL NEW FLASHINGS USING STRESSBASE 80 PLUS.
3.
POWER WASH ROOF SYSTEM USING TSP AND WATER SOLUTION.
SCOPE OF WORK:
1.
TEST ALL DRAINS AND DOWNSPOUT PRIOR TO CONSTRUCTION.
2.
CLEAN WORK SITE AT END OF EVERY WORK DAY.
"""

GARLAND_AGENT_SPLIT = """
THE GARLAND COMPANY INC
AGENT:
�
GARLAND
DOUG CLARK
Mt. Diablo Unified School District
AYERS ELEMENTARY
5120 MYRTLE DR, CONCORD, CA 94521
"""

ARCHITECT_COVER = """
PROJECT OWNER & TITLE
MOUNT DIABLO UNIFIED SCHOOL DISTRICT
1936 Carlotta Drive
Concord, California 94519
MEADOW HOMES ES MODERNIZATION
1371 Detroit Ave
Concord, CA 94520
DATE:
AUGUST 01, 2025
JOB NUMBER:
23119.01
C I V I L   E N G I N E E R
CARROLL ENGINEERING
1101 South Winchester Blvd., Suite H184
San Jose, CA 95128
TEL (408) 261-9800
M E C H A N I C A L   E N G I N E E R
CYPRESS ENGINEERING GROUP
9 Harris Court, Suite A8
Monterey, CA 93940
TEL (831) 218-1802
"""

# Mirrors Real data/1945BIDDrawingsRVES.pdf title block (split lines).
ARCHITECT_1945_COVER = """
CONSULTANTS
PROJECT OWNER & TITLE
SHEET NO.
C-36107
REN. 01-31-25
991 W HEDDING ST., SUITE 101
SAN JOSE, CA 95126
TEL (408) 715-4470
SHEET INDEX & OVERALL SITE
PLAN
RIO VISTA ES
MODERNIZATION &
SITE IMPROVEMENTS
SEPTEMBER 27, 2024
23118.01
611 Pacifica Ave
Bay Point, CA 94565
MOUNT DIABLO UNIFIED
SCHOOL DISTRICT
1936 Carlotta Dr. Concord
Concord, CA 94519
CONSTRUCTION DOCUMENT
BASE Design Inc.
www.BASEdesigninc.com
"""

# Same stamp as 1945 but firm is logo-only (no searchable company line).
ARCHITECT_LOGO_ONLY_COVER = """
CONSULTANTS
PROJECT OWNER & TITLE
SHEET NO.
C-36107
REN. 01-31-25
991 W. HEDDING STREET, SUITE 101
SAN JOSE, CA 95126
TEL (408) 715-4470
LOMA VISTA ADULT CENTER
TECHNOLOGY INFRASTRUCTURE UPGRADE
NOVEMBER 18, 2024
24103.01
1266 San Carlos Ave.
Concord, CA 94518
MOUNT DIABLO UNIFIED
SCHOOL DISTRICT
1936 Carlotta Dr.
Concord, CA 94519
CONSTRUCTION DOCUMENT
"""

ARCHITECT_SCOPE = """
1. THE CONTRACTOR SHALL FULLY RE-CABLE THE DATA NETWORK, COPPER AND FIBER.
2. THE CONTRACTOR WILL ESTABLISH (N) MDF AND IDF LOCATIONS AS INDICATED ON THE PLANS.
3. THE CONTRACTOR SHALL PROVIDE A (N) BOGEN NYQUIST INTERCOM SYSTEM IN ACCORDANCE WITH THE PLANS.
TECHNOLOGY SCOPE OF WORK:
1. PRIOR TO BEGINNING ANY SITE WORK, AN ON-SITE PRE-CONSTRUCTION MEETING SHALL BE HELD WITH THE LOW VOLTAGE DESIGNER.
2. CONDUIT BODIES AND LB FITTINGS ARE PROHIBITED IN ANY PATHWAY CONTAINING DATA CABLING.
PAINT SCOPE OF WORK:
1. CONTRACTOR SHALL PATCH AND PAINT ALL WALL AND CEILING SURFACES DISTURBED BY DEMO.
2. CONTRACTOR SHALL CLEAN, PRIME, AND PAINT ALL (N) CONDUIT AND J-BOXES ON FINISHED WALLS.
3. CONTRACTOR SHALL COORDINATE WITH DISTRICT REPRESENTATIVE FOR PAINT COLORS.
DATA LOCATION, CAT6A JACK/CABLE,
SEE 27 10 00
PANDUIT
"""

ARCHITECT_PROSE_SCOPE = """
EXTERIOR REPAINTING SCOPE OF WORK:
1.
2.
3.
4.
REPAIR ALL DAMAGED SURFACES TO ACHIEVE A FLUSH, CLEAN FINISH.
SAND, CLEAN, AND RE-PRIME ALL EXISTING SURFACES.
REPAINT ALL EXISTING PAINTED SURFACES, INCLUDING BUT NOT LIMITED TO WALLS,
WALKWAY SOFFITS, CONDUITS, PIPES, GUTTERS, DOWNSPOUTS, TRIM, AND DOOR AND
WINDOW FRAMES OF EXISTING BUILDINGS.
USE COLORS AS SPECIFIED IN THE DISTRICT-APPROVED COLOR PALETTE.
TYP. 09.91.B
TYP. 09.91.C
"""

ARCHITECT_CODES_AND_SUMMARY = """
PARTIAL LIST OF APPLICABLE CODES AS OF JANUARY 1, 2023
CALIFORNIA BUILDING CODE (CBC) - CCR, TITLE 24, PART 2
CALIFORNIA ELECTRICAL CODE (CEC) - CCR, TITLE 24, PART 3
CALIFORNIA MECHANICAL CODE (CMC) - CCR, TITLE 24, PART 4
CALIFORNIA PLUMBING CODE (CPC) - CCR, TITLE 24, PART 5
PROJECT DESCRIPTION
- REPLACEMENT IN-KIND OF HVAC EQUIPMENT IN SOME BUILDINGS, EQUIPMENT TUNE-UP IN OTHERS.
- PROVIDE NEW HVAC CONTROLS WHERE INDICATED ON THE MECHANICAL DRAWINGS.
LIMITS OF WORK
MODERNIZATION INC. 2 - THIS SET
BID ALTERNATE A — ADDITIONAL ROOFTOP UNIT REPLACEMENT AT BUILDING 4
"""

# 1963-style cover with explicit DSA application number.
DSA_COVER_1963 = """
EL DORADO MIDDLE SCHOOL MULTI-PURPOSE
MT. DIABLO UNIFIED SCHOOL DISTRICT
1936 CARLOTTA DR, CONCORD, CA 94519
600 Q Street, Suite 100
Sacramento, CA 95811
DSA #
01-121540
JOB NO.
23100.01
DSA SUBMITTAL
GENERAL SCOPE OF WORK INCLUDES BUT IS NOT LIMITED TO:
PROVIDE NEW HVAC SYSTEMS THROUGHOUT THE BUILDING, INCLUDING THE KITCHEN.
MECHANICAL SCOPE OF WORK:
1. DEMOLISH EXISTING HVAC EQUIPMENT AS SHOWN.
2. INSTALL NEW PACKAGED ROOFTOP UNITS PER SCHEDULE.
"""

NO_SCOPE = """
SOME RANDOM DRAWING
ACME CORP INC
PHONE (555) 123-4567
123 MAIN STREET
SPRINGFIELD, IL 62701
"""


class TestRequirementExtractor:
    def test_garland_style_provider_and_scope(self) -> None:
        pages = [
            PageText(1, GARLAND_COVER),
            PageText(2, GARLAND_SCOPE),
        ]
        info = extract_requirement(pages)

        assert info.provider_company and "GARLAND" in info.provider_company.upper()
        assert info.provider_agent and "DOUG" in info.provider_agent.upper()
        assert info.provider_phone is not None
        assert info.provider_address and "CLEVELAND" in info.provider_address.upper()
        assert "91" in info.provider_address and "44105" in info.provider_address
        assert info.client and "SCHOOL DISTRICT" in info.client.upper()
        assert info.project_title and "AYERS" in info.project_title.upper()
        assert info.site_address and "CONCORD" in info.site_address.upper()
        assert "MYRTLE" in info.site_address.upper()
        assert info.date and ("2025" in info.date or "1-13" in info.date or "01-13" in info.date)
        assert info.work_type and "ROOF" in info.work_type.upper()
        assert any(entry.upper().startswith("SHT 2") for entry in info.sheet_index)
        assert len(info.scope_sections) >= 1
        assert any(len(section.items) >= 2 for section in info.scope_sections)

    def test_garland_split_agent_and_numbered_lines(self) -> None:
        """Real PDFs put AGENT:/1. on their own lines with values below."""
        pages = [
            PageText(1, GARLAND_AGENT_SPLIT),
            PageText(2, GARLAND_SCOPE),
        ]
        info = extract_requirement(pages)
        assert info.provider_agent and "DOUG" in info.provider_agent.upper()
        assert sum(len(s.items) for s in info.scope_sections) >= 5
        assert any("FLASHINGS" in item.upper() for s in info.scope_sections for item in s.items)
        # Distinct bodies under two headings stay as two sections.
        titles = [s.title.upper() for s in info.scope_sections]
        assert any("LIQUID" in t for t in titles)
        assert any("SCOPE OF WORK" in t for t in titles)

    def test_empty_scope_of_work_does_not_copy_prior_section(self) -> None:
        """Bare SCOPE OF WORK with no body must not repeat the prior scope list."""
        pages = [
            PageText(
                2,
                """
LIQUID APPLIED ROOF RESTORATION SCOPE:
1.
REMOVE AND DISPOSE OF ALL VERTICAL FLASHINGS AND DETAIL FLASHINGS.
2.
INSTALL NEW FLASHINGS USING STRESSBASE 80 PLUS.
3.
POWER WASH ROOF SYSTEM USING TSP AND WATER SOLUTION.
4.
REINFORCE ALL SEAMS BY APPLYING LIQUITEC AND POLYESTER.
5.
APPLY A BASE AND TOP COAT OF LIQUITEC.
SCOPE OF WORK:
5120 MYRTLE DR, CONCORD, CA 94521
THE GARLAND COMPANY INC
PHONE (800) 321-9336
""",
            ),
        ]
        info = extract_requirement(pages)
        assert len(info.scope_sections) == 1
        assert "LIQUID" in info.scope_sections[0].title.upper()
        assert not any(
            s.title.upper().strip() in {"SCOPE OF WORK", "SCOPE OF WORK:"}
            or re.fullmatch(r"SCOPE(?:\s+OF\s+WORK)?", s.title.strip(), re.I)
            for s in info.scope_sections
        )

    def test_architect_style_consultants_and_trade_scopes(self) -> None:
        pages = [
            PageText(1, ARCHITECT_COVER),
            PageText(2, ARCHITECT_SCOPE),
        ]
        info = extract_requirement(pages)

        assert info.client and "MOUNT DIABLO" in info.client.upper()
        assert info.job_number == "23119.01"
        assert info.date is not None
        assert len(info.consultants) >= 1
        by_title = {s.title.upper(): s for s in info.scope_sections}
        assert any("TECHNOLOGY" in t for t in by_title)
        assert any("PAINT" in t for t in by_title)
        tech = next(s for s in info.scope_sections if "TECHNOLOGY" in s.title.upper())
        paint = next(s for s in info.scope_sections if "PAINT" in s.title.upper())
        # Primary tech list above the heading — exact drawing bullets, no merged extras.
        assert len(tech.items) == 3
        assert any("RE-CABLE" in item.upper() for item in tech.items)
        assert any("BOGEN" in item.upper() for item in tech.items)
        assert not any("PRE-CONSTRUCTION" in item.upper() for item in tech.items)
        assert len(paint.items) == 3
        assert all("PAINT" in item.upper() or "COORDINATE" in item.upper() for item in paint.items)
        assert not any("RE-CABLE" in item.upper() or "BOGEN" in item.upper() for item in paint.items)
        assert not any("PANDUIT" in item.upper() or "CAT6A" in item.upper() for item in paint.items)
        assert sum(len(section.items) for section in info.scope_sections) >= 3

    def test_architect_1945_split_title_block(self) -> None:
        pages = [
            PageText(1, ARCHITECT_1945_COVER),
            PageText(2, ARCHITECT_PROSE_SCOPE),
            PageText(3, ARCHITECT_CODES_AND_SUMMARY),
        ]
        info = extract_requirement(pages)

        assert info.client and "MOUNT DIABLO UNIFIED SCHOOL DISTRICT" in info.client.upper()
        assert info.project_title and "RIO VISTA" in info.project_title.upper()
        assert "MODERNIZATION" in info.project_title.upper()
        assert info.job_number == "23118.01"
        assert info.date and "2024" in info.date
        assert info.site_address and "PACIFICA" in info.site_address.upper()
        assert "BAY POINT" in info.site_address.upper()
        assert info.provider_phone == "(408) 715-4470"
        assert info.provider_address and "HEDDING" in info.provider_address.upper()
        assert info.provider_company and "BASE" in info.provider_company.upper()
        assert info.aor_license == "C-36107"
        assert info.document_status and "CONSTRUCTION" in info.document_status.upper()
        assert info.work_type and "MODERNIZATION" in info.work_type.upper()
        assert info.district_address and "CARLOTTA" in info.district_address.upper()
        assert any("MECHANICAL CODE" in code.upper() for code in info.applicable_codes)
        assert any("HVAC" in bullet.upper() for bullet in info.project_summary)
        assert any("INC" in item.upper() for item in info.limits_of_work)
        assert any("ALTERNATE" in alt.upper() for alt in info.alternates)
        assert any(
            "REPAIR ALL DAMAGED" in item.upper()
            for section in info.scope_sections
            for item in section.items
        )

    def test_dsa_cover_1963_style(self) -> None:
        info = extract_requirement([PageText(1, DSA_COVER_1963)])
        assert info.dsa_number == "01-121540"
        assert info.document_status and "DSA" in info.document_status.upper()
        assert any("HVAC" in bullet.upper() for bullet in info.project_summary)
        assert any("MECHANICAL" in s.title.upper() for s in info.scope_sections)

    def test_garland_scope_excludes_titleblock_chrome(self) -> None:
        """Numbered scope sits above the heading; stamp/phone must not become bullets."""
        pages = [
            PageText(
                1,
                """
SCOPE OF WORK: REPLACEMENT
PHONE (800) 321-9336 / FAX (216) 641-0633
THE GARLAND COMPANY INC
GLENBROOK MIDDLE SCHOOL
Mt. Diablo Unified School District
""",
            ),
            PageText(
                2,
                """
1.
REMOVE AND DISPOSE OF ALL ROOFING, EDGE METAL, AND COUNTER-FLASHING DOWN TO STRUCTURAL DECK.
2.
PERFORM ANY REPAIRS AS NEEDED.
3.
NAIL RED ROSIN PAPER.
4.
INSTALL NEW KYNAR COATED EDGE METAL AND COUNTERFLASHING.
SCOPE OF WORK: REPLACEMENT
PHONE (800) 321-9336 / FAX (216) 641-0633
THE GARLAND COMPANY INC.
GLENBROOK MIDDLE SCHOOL
Mt. Diablo Unified School District
2351 OLIVERA RD, CONCORD, CA 94520
""",
            ),
        ]
        info = extract_requirement(pages)
        assert info.scope_sections
        items = [item.upper() for section in info.scope_sections for item in section.items]
        assert any("REMOVE AND DISPOSE" in item for item in items)
        assert any("NAIL RED ROSIN" in item for item in items)
        assert not any("PHONE" in item for item in items)
        assert not any("GARLAND COMPANY" in item for item in items)
        assert not any(item.strip() == "GLENBROOK MIDDLE SCHOOL" for item in items)

        info = extract_requirement([PageText(1, NO_SCOPE)])
        assert info.provider_company and "ACME" in info.provider_company.upper()
        assert info.provider_phone is not None
        assert any("no explicit scope" in note.lower() for note in info.notes)

    def test_empty_pdf_notes(self) -> None:
        info = extract_requirement([PageText(1, "")])
        assert info.pages_with_text == 0
        assert info.scope_sections == []


class TestRequirementPdf:
    def test_filename_convention(self) -> None:
        assert requirement_pdf_filename("abc.pdf") == "abc requirement.pdf"
        assert requirement_pdf_filename("1954BIDDrawings-AyersES011325.PDF") == (
            "1954BIDDrawings-AyersES011325 requirement.pdf"
        )

    def test_generates_readable_pdf(self, tmp_path: Path) -> None:
        pages = [PageText(1, GARLAND_COVER), PageText(2, GARLAND_SCOPE)]
        info = extract_requirement(pages)
        out = tmp_path / "ayers requirement.pdf"
        generate_requirement_pdf(info, "ayers.pdf", out)

        assert out.exists()
        assert out.stat().st_size > 500
        with fitz.open(str(out)) as doc:
            text = "".join(page.get_text() for page in doc)
        assert "REQUIREMENT SUMMARY" in text.upper()
        assert "WHO PROVIDED" in text.upper()
        assert "GARLAND" in text.upper()
        assert "SCOPE" in text.upper()
        assert "PROJECT DETAILS" in text.upper()
        assert "DRAWING / SHEET INDEX" in text.upper()
        assert "WORK TYPE" in text.upper()
        # Sparse fields omitted from the printed requirement PDF.
        assert "AGENT / CONTACT PERSON" not in text.upper()
        assert "AOR LICENSE" not in text.upper()
        assert "JOB / BID NUMBER" not in text.upper()
        assert "DSA NUMBER" not in text.upper()

    def test_restoration_pdf_omits_empty_scope_of_work_subsection(self, tmp_path: Path) -> None:
        """Empty/generic Scope of Work must not be printed when only restoration scope exists."""
        pages = [
            PageText(
                2,
                """
LIQUID APPLIED ROOF RESTORATION SCOPE:
1.
REMOVE AND DISPOSE OF ALL VERTICAL FLASHINGS AND DETAIL FLASHINGS. REMOVE ALL BLISTERS.
2.
INSTALL NEW FLASHINGS USING STRESSBASE 80 PLUS AND STRESSPLY PLUS FR MINERAL.
3.
POWER WASH ROOF SYSTEM USING TSP AND WATER SOLUTION.
4.
REINFORCE ALL SEAMS BY APPLYING LIQUITEC AND POLYESTER PER SPECIFIED REQUIREMENTS.
5.
APPLY A BASE AND TOP COAT OF LIQUITEC PER SPECIFIED REQUIREMENTS.
SCOPE OF WORK:
5120 MYRTLE DR, CONCORD, CA 94521
THE GARLAND COMPANY INC
PHONE (800) 321-9336
""",
            ),
        ]
        info = extract_requirement(pages)
        out = tmp_path / "restoration requirement.pdf"
        generate_requirement_pdf(info, "restoration.pdf", out)
        with fitz.open(str(out)) as doc:
            text = "".join(page.get_text() for page in doc)
        assert "Liquid Applied Roof Restoration Scope" in text
        assert text.upper().count("REMOVE AND DISPOSE OF ALL VERTICAL") == 1
        # Subsection heading like "1. Scope Of Work (source page …)" must not appear.
        assert not re.search(
            r"\d+\.\s*Scope Of Work\s*\(source page",
            text,
            re.I,
        )

    def test_sanitize_drops_overlapping_generic_scope_of_work(self) -> None:
        """PDF sanitize layer drops a duplicate Scope of Work even if extractor leaked it."""
        items = [
            "REMOVE AND DISPOSE OF ALL VERTICAL FLASHINGS AND DETAIL FLASHINGS PER MANUFACTURER.",
            "INSTALL NEW FLASHINGS USING STRESSBASE 80 PLUS AND STRESSPLY PLUS FR MINERAL.",
            "POWER WASH ROOF SYSTEM USING TSP AND WATER SOLUTION BEFORE COATING.",
            "REINFORCE ALL SEAMS BY APPLYING LIQUITEC AND POLYESTER PER REQUIREMENTS.",
            "APPLY A BASE AND TOP COAT OF LIQUITEC PER SPECIFIED REQUIREMENTS.",
        ]
        info = RequirementInfo(
            scope_sections=[
                ScopeSection(
                    title="LIQUID APPLIED ROOF RESTORATION SCOPE",
                    items=items,
                    page_number=2,
                ),
                ScopeSection(title="SCOPE OF WORK", items=list(items), page_number=2),
            ],
            pages_with_text=1,
            total_pages=1,
        )
        cleaned = sanitize_requirement(info)
        titles = [s.title.upper() for s in cleaned.scope_sections]
        assert any("LIQUID" in t for t in titles)
        assert not any(re.fullmatch(r"SCOPE(?: OF WORK)?", t.strip()) for t in titles)

    def test_architect_pdf_includes_new_sections(self, tmp_path: Path) -> None:
        pages = [
            PageText(1, ARCHITECT_1945_COVER),
            PageText(2, ARCHITECT_PROSE_SCOPE),
            PageText(3, ARCHITECT_CODES_AND_SUMMARY),
        ]
        info = extract_requirement(pages)
        out = tmp_path / "1945 requirement.pdf"
        generate_requirement_pdf(info, "1945.pdf", out)
        with fitz.open(str(out)) as doc:
            text = "".join(page.get_text() for page in doc)
        assert "DOCUMENT STATUS" in text.upper()
        assert "APPLICABLE CODES" in text.upper()
        assert "PROJECT / HVAC SUMMARY" in text.upper()
        assert "BID ALTERNATES" in text.upper()
        assert "AGENT / CONTACT PERSON" not in text.upper()
        assert "AOR LICENSE" not in text.upper()
        assert "JOB / BID NUMBER" not in text.upper()
        assert "DSA NUMBER" not in text.upper()
        # Upload-only naming convention; path is caller-chosen (storage/requirements).
        assert out.name == "1945 requirement.pdf"

    def test_consultants_table_includes_address(self, tmp_path: Path) -> None:
        pages = [PageText(1, ARCHITECT_COVER), PageText(2, ARCHITECT_SCOPE)]
        info = extract_requirement(pages)
        out = tmp_path / "meadow requirement.pdf"
        generate_requirement_pdf(info, "meadow.pdf", out)
        with fitz.open(str(out)) as doc:
            text = "".join(page.get_text() for page in doc)
        assert "CONSULTANTS" in text.upper()
        assert "ADDRESS" in text.upper()
        assert "CARROLL" in text.upper() or "CYPRESS" in text.upper()


class TestVerticalTitleBlock:
    def test_delta_view_vertical_sidebar_fields(self) -> None:
        """Garland cover uses a rotated right-hand title strip."""
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1954BIDDrawings-DeltaViewES011325.pdf"
        )
        if not pdf.exists():
            return
        info = extract_requirement_from_pdf(pdf)
        assert info.provider_company and "GARLAND" in info.provider_company.upper()
        assert info.provider_address and "CLEVELAND" in info.provider_address.upper()
        assert info.provider_agent and "DOUG" in info.provider_agent.upper()
        assert info.provider_phone == "(800) 321-9336"
        assert info.client and "DIABLO" in info.client.upper()
        assert info.project_title and "DELTA VIEW" in info.project_title.upper()
        assert info.site_address and "RIO VERDE" in info.site_address.upper()
        assert info.date and "2025" in info.date

    def test_vertical_reconstruction_keeps_date_beside_label(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data"
            r"\1954BIDDrawings-DeltaViewES011325.pdf"
        )
        if not pdf.exists():
            return
        pages = extract_pdf_text(pdf)
        text = pages[0].text
        upper = text.upper()
        assert "DATE:" in upper
        assert "1-6-25" in text
        # Use the last DATE: occurrence (from reconstructed vertical strip).
        idx = upper.rfind("DATE:")
        window = text[idx : idx + 30].upper()
        assert "1-6-25" in window


class TestLogoFirmExtraction:
    def test_stamp_map_hedding_phone(self) -> None:
        firm = lookup_firm_from_stamp(
            phone="(408) 715-4470",
            address="991 W. HEDDING STREET, SUITE 101, SAN JOSE, CA 95126",
        )
        assert firm == "196 Architects"

    def test_stamp_map_ignores_unrelated(self) -> None:
        assert lookup_firm_from_stamp(phone="(800) 321-9336", address="CLEVELAND, OH") is None

    def test_parse_ocr_196_architects(self) -> None:
        firm = parse_firm_from_ocr_lines(
            ["196 ARCHITECTS", "991 W. HEDDING STREET", "SAN JOSE, CA 95126"]
        )
        assert firm == "196 Architects"

    def test_parse_ocr_split_number_and_word(self) -> None:
        firm = parse_firm_from_ocr_lines(["196", "ARCHITECTS", "SAN JOSE"])
        assert firm == "196 Architects"

    def test_text_company_beats_stamp_map(self) -> None:
        """BASE Design Inc. in text must not be overwritten by stamp map."""
        info = extract_requirement([PageText(1, ARCHITECT_1945_COVER)])
        assert info.provider_company and "BASE" in info.provider_company.upper()
        fill_provider_company_from_logo(Path("unused.pdf"), info)
        assert "BASE" in info.provider_company.upper()

    def test_logo_only_cover_filled_via_stamp_map(self, tmp_path: Path) -> None:
        """Phone/address present, no firm text → stamp map fills Company / Firm."""
        # Build a minimal PDF so extract_requirement_from_pdf runs the logo hook.
        pdf = tmp_path / "logo_only.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), ARCHITECT_LOGO_ONLY_COVER.strip(), fontsize=9)
        doc.save(pdf)
        doc.close()

        info = extract_requirement_from_pdf(pdf)
        assert info.provider_phone == "(408) 715-4470"
        assert info.provider_address and "HEDDING" in info.provider_address.upper()
        assert info.provider_company == "196 Architects"

    def test_fill_skips_when_company_set(self) -> None:
        info = RequirementInfo(provider_company="Already Set Inc.")
        info.provider_phone = "(408) 715-4470"
        info.provider_address = "991 W HEDDING ST., SUITE 101, SAN JOSE, CA 95126"
        fill_provider_company_from_logo(Path("unused.pdf"), info)
        assert info.provider_company == "Already Set Inc."

    def test_real_1948_logo_only_if_present(self) -> None:
        pdf = Path(
            r"D:\Cost Estimation Using Text and Object\Real data\1948BIDDrawings.pdf"
        )
        if not pdf.exists():
            return
        info = extract_requirement_from_pdf(pdf)
        assert info.provider_company == "196 Architects"
        assert info.provider_phone == "(408) 715-4470"


class TestExtractFromSamplePdf:
    def test_sample_layout_has_scope(self, sample_pdf: Path) -> None:
        info = extract_requirement_from_pdf(sample_pdf)
        assert info.pages_with_text >= 1
        assert len(info.scope_sections) >= 1
        assert any("DIFFUSER" in item.upper() or "SENSOR" in item.upper()
                   for section in info.scope_sections for item in section.items)
