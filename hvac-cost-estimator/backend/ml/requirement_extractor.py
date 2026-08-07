"""Extract requirement metadata and scope-of-work from bid/drawing PDFs.

Uses PyMuPDF text extraction (no OCR) — the Real data set has embedded text.
Handles two common format families:

- Garland-style: AGENT, company/phone/address on the cover; numbered SCOPE OF WORK.
- Architect CD sets: owner/project/consultants on the cover; trade SCOPE sections
  or GENERAL NOTES / SUMMARY OF WORK fallbacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ml.metadata_mapper import normalize_date

PHONE_RE = re.compile(
    r"(?:PHONE|TEL|FAX)?\s*[\(:]?\s*"
    r"(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})",
    re.IGNORECASE,
)
STANDALONE_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")
FAX_RE = re.compile(r"FAX\s*[\(:]?\s*(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})", re.I)
ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
# Garland title blocks use full state names: "CLEVELAND, OHIO 44105-2197"
STATE_ZIP_RE = re.compile(
    r"\b(?:[A-Z]{2}|"
    r"ALABAMA|ALASKA|ARIZONA|ARKANSAS|CALIFORNIA|COLORADO|CONNECTICUT|DELAWARE|"
    r"FLORIDA|GEORGIA|HAWAII|IDAHO|ILLINOIS|INDIANA|IOWA|KANSAS|KENTUCKY|"
    r"LOUISIANA|MAINE|MARYLAND|MASSACHUSETTS|MICHIGAN|MINNESOTA|MISSISSIPPI|"
    r"MISSOURI|MONTANA|NEBRASKA|NEVADA|NEW\s+HAMPSHIRE|NEW\s+JERSEY|"
    r"NEW\s+MEXICO|NEW\s+YORK|NORTH\s+CAROLINA|NORTH\s+DAKOTA|OHIO|OKLAHOMA|"
    r"OREGON|PENNSYLVANIA|RHODE\s+ISLAND|SOUTH\s+CAROLINA|SOUTH\s+DAKOTA|"
    r"TENNESSEE|TEXAS|UTAH|VERMONT|VIRGINIA|WASHINGTON|WEST\s+VIRGINIA|"
    r"WISCONSIN|WYOMING)\s+\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)
ADDRESS_RE = re.compile(
    r"^\d{1,6}\s+[A-Z0-9][\w\s.,#'-]{3,80}$",
    re.IGNORECASE,
)
# Single-line office/site: "3800 EAST 91st STREET - CLEVELAND, OHIO 44105-2197"
SINGLE_LINE_ADDRESS_RE = re.compile(
    r"^\d{1,6}\s+.+\b(?:[A-Z]{2}|"
    r"ALABAMA|ALASKA|ARIZONA|ARKANSAS|CALIFORNIA|COLORADO|CONNECTICUT|DELAWARE|"
    r"FLORIDA|GEORGIA|HAWAII|IDAHO|ILLINOIS|INDIANA|IOWA|KANSAS|KENTUCKY|"
    r"LOUISIANA|MAINE|MARYLAND|MASSACHUSETTS|MICHIGAN|MINNESOTA|MISSISSIPPI|"
    r"MISSOURI|MONTANA|NEBRASKA|NEVADA|NEW\s+HAMPSHIRE|NEW\s+JERSEY|"
    r"NEW\s+MEXICO|NEW\s+YORK|NORTH\s+CAROLINA|NORTH\s+DAKOTA|OHIO|OKLAHOMA|"
    r"OREGON|PENNSYLVANIA|RHODE\s+ISLAND|SOUTH\s+CAROLINA|SOUTH\s+DAKOTA|"
    r"TENNESSEE|TEXAS|UTAH|VERMONT|VIRGINIA|WASHINGTON|WEST\s+VIRGINIA|"
    r"WISCONSIN|WYOMING)\s+\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)
AGENT_RE = re.compile(r"^\s*AGENT\s*:\s*(.+)?$", re.IGNORECASE)
JOB_NUMBER_RE = re.compile(r"^\s*JOB\s+NUMBER\s*:\s*(.*)$", re.IGNORECASE)
# Common A/E job numbers: 23118.01
JOB_NUMBER_VALUE_RE = re.compile(r"^\d{4,6}\.\d{2}$")
DATE_LABEL_RE = re.compile(r"^\s*DATE\s*:\s*(.*)$", re.IGNORECASE)
DATE_VALUE_RE = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{4})\b",
    re.IGNORECASE,
)
LICENSE_RE = re.compile(r"^C-\d{4,6}$", re.IGNORECASE)
AOR_LICENSE_RE = re.compile(r"\b(C-\d{4,6})\b", re.IGNORECASE)
DSA_NUMBER_RE = re.compile(
    r"\bDSA\s*#?\s*[:.]?\s*(0\d-\d{5,7})\b",
    re.IGNORECASE,
)
DSA_NUMBER_LOOSE_RE = re.compile(r"\b(0\d-\d{5,7})\b")
DOCUMENT_STATUS_RE = re.compile(
    r"\b(CONSTRUCTION\s+DOCUMENTS?|BID\s+SET|FOR\s+BID|DSA\s+SUBMITTAL|"
    r"ISSUED\s+FOR\s+[A-Z][A-Z\s]{2,40})\b",
    re.IGNORECASE,
)
WORK_TYPE_RE = re.compile(
    r"\b(ROOF\s+RESTORATION|ROOF\s+REPLACEMENT|MODERNIZATION(?:\s*&\s*SITE\s+IMPROVEMENTS?)?|"
    r"SITE\s+IMPROVEMENTS?|INFRASTRUCTURE\s+UPGRADE|HVAC\s+REPLACEMENT)\b",
    re.IGNORECASE,
)
SHT_INDEX_RE = re.compile(
    r"^SHT\s+(\d+)\s*[-–]\s*(.+)$",
    re.IGNORECASE,
)
SHEET_INDEX_LINE_RE = re.compile(
    r"^(\d?[A-Z]{1,3}-?\d{2,3}[A-Z]?)\s+([A-Z][A-Z0-9\s/&.,'-]{3,60})$",
    re.IGNORECASE,
)
CODE_LINE_RE = re.compile(
    r"\b(CALIFORNIA\s+(?:BUILDING|ELECTRICAL|MECHANICAL|PLUMBING|FIRE|GREEN\s+BUILDING|"
    r"ENERGY|ADMINISTRATIVE)\s+CODE|"
    r"TITLE\s+24|CCR(?:\s*,?\s*TITLE\s+24)?|"
    r"\bCBC\b|\bCEC\b|\bCMC\b|\bCPC\b|\bCFC\b|\bCALGreen\b)",
    re.IGNORECASE,
)
HVAC_SUMMARY_RE = re.compile(
    r"\b(HVAC|MECHANICAL|AIR\s+CONDITION|ROOF\s+TOP\s+UNIT|PROVIDE\s+NEW\s+HVAC|"
    r"REPLACEMENT\s+IN-KIND\s+OF\s+HVAC|GENERAL\s+SCOPE\s+OF\s+WORK\s+INCLUDES)\b",
    re.IGNORECASE,
)
LIMITS_OF_WORK_RE = re.compile(
    r"\b(LIMITS?\s+OF\s+WORK|MODERNIZATION\s+INC\.?\s*\d+|INCREMENT\s+\d+)\b",
    re.IGNORECASE,
)
BID_ALTERNATE_RE = re.compile(
    r"^(?:BID\s+)?ALTERNATE\s*(?:NO\.?\s*)?#?\s*([A-Z0-9]+)"
    r"(?:\s*[-–—:]\s*(.+))?$",
    re.IGNORECASE,
)
SCHOOL_CODE_RE = re.compile(
    r"\b([A-Z][A-Z\s.'-]{2,40}\s+(?:ES|MS|HS|ELEMENTARY|MIDDLE|HIGH(\s+SCHOOL)?))\b",
    re.IGNORECASE,
)
TITLE_CONTINUATION_RE = re.compile(
    r"\b(MODERNIZATION|SITE\s+IMPROVEMENTS?|ROOF\s+(RESTORATION|REPLACEMENT)|"
    r"INFRASTRUCTURE|CONSTRUCTION\s+DOCUMENTS?)\b",
    re.IGNORECASE,
)

COMPANY_HINTS = re.compile(
    r"\b(COMPANY|INC\.?|LLC|LLP|LTD\.?|CORP\.?|ARCHITECTS?|ENGINEERING|ENGINEERS?|"
    r"CONSULTANTS?|GROUP|ASSOCIATES|PARTNERS)\b",
    re.IGNORECASE,
)
DISTRICT_RE = re.compile(r"\b(SCHOOL\s+DISTRICT|UNIFIED|DISTRICT)\b", re.IGNORECASE)
SKIP_LINES = re.compile(
    r"^(OF|SHEET|REVISION|CHK\s*BY|DWG\s*BY|SCALE|N\.?I\.?C\.?|"
    r"DRAWING\s+INDEX|LOCATION\s+MAP|KEY\s+PLAN|SHEET\s+NO|"
    r"FILE\s+LOCATION|DATE\s+PLOTTED|#|NO\.?)$",
    re.IGNORECASE,
)

SCOPE_HEADING_RE = re.compile(
    r"^(?:SHT\s+\d+\s*[-–]?\s*)?(.+?\s+)?SCOPE(?:\s+OF\s+WORK)?(?:\s*:.*)?$",
    re.IGNORECASE,
)
SCOPE_FALLBACK_RE = re.compile(
    r"^(SUMMARY\s+OF\s+WORK|WORK\s+INCLUDES|GENERAL\s+NOTES)(?:\s*:.*)?$",
    re.IGNORECASE,
)
_SCOPE_HEADING_SENTENCE_RE = re.compile(
    r"\b(SHALL|WILL|REQUIRED\s+TO|CONTRACTOR|COMPLETE\s+THE|MATERIALS|"
    r"RESPONSIBLE|INCLUDING|PROVIDE|INSTALL|REMOVE)\b",
    re.IGNORECASE,
)
NUMBERED_ITEM_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")
DISCIPLINE_RE = re.compile(
    r"^(CIVIL|STRUCTURAL|MECHANICAL|ELECTRICAL|TECHNOLOGY|"
    r"ACOUSTICAL|LANDSCAPE|FIRE\s+PROTECTION)\s+"
    r"(ENGINEER|ENGINEERING|CONSULTANT)S?\s*$",
    re.IGNORECASE,
)


def _is_scope_heading_line(line: str) -> bool:
    """True for real section titles, not sentences that merely mention scope of work."""
    text = re.sub(r"\s+", " ", line).strip()
    if not text or len(text) >= 80:
        return False
    if re.match(r"^SHT\b", text, re.I):
        return False
    if SCOPE_FALLBACK_RE.match(text):
        return True
    if not SCOPE_HEADING_RE.match(text):
        return False
    # Reject body sentences: "…TO COMPLETE THE SCOPE OF WORK."
    if _SCOPE_HEADING_SENTENCE_RE.search(text):
        return False
    if text.count(" ") > 8 and not text.endswith(":"):
        return False
    return True


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str

    @property
    def lines(self) -> list[str]:
        return [line.strip() for line in self.text.splitlines() if line.strip()]


@dataclass
class Consultant:
    discipline: str
    firm: str
    phone: str | None = None
    address: str | None = None


@dataclass
class ScopeSection:
    title: str
    items: list[str] = field(default_factory=list)
    page_number: int = 1
    notes: str | None = None


@dataclass
class RequirementInfo:
    """Structured requirement metadata + scope extracted from a bid PDF."""

    provider_company: str | None = None
    provider_agent: str | None = None
    provider_phone: str | None = None
    provider_fax: str | None = None
    provider_address: str | None = None
    aor_license: str | None = None
    client: str | None = None
    project_title: str | None = None
    work_type: str | None = None
    site_address: str | None = None
    site_addresses: list[str] = field(default_factory=list)
    district_address: str | None = None
    job_number: str | None = None
    dsa_number: str | None = None
    date: str | None = None
    document_status: str | None = None
    consultants: list[Consultant] = field(default_factory=list)
    sheet_index: list[str] = field(default_factory=list)
    project_summary: list[str] = field(default_factory=list)
    applicable_codes: list[str] = field(default_factory=list)
    limits_of_work: list[str] = field(default_factory=list)
    alternates: list[str] = field(default_factory=list)
    scope_sections: list[ScopeSection] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pages_with_text: int = 0
    total_pages: int = 0

    @property
    def provider_summary(self) -> str | None:
        parts: list[str] = []
        if self.provider_company:
            parts.append(self.provider_company)
        if self.provider_agent:
            parts.append(f"Agent: {self.provider_agent}")
        if self.provider_phone:
            parts.append(self.provider_phone)
        return " · ".join(parts) if parts else None


def extract_pdf_text(pdf_path: Path) -> list[PageText]:
    """Extract embedded text from every page, including vertical title blocks.

    Plain ``get_text()`` often returns rotated title-block lines in a scrambled
    order (e.g. ``DATE:`` far from ``1-6-25``). We append a reconstructed
    vertical strip so Garland-style sidebars stay readable for the parsers.
    """
    import fitz

    document = fitz.open(str(pdf_path))
    pages: list[PageText] = []
    try:
        for index in range(document.page_count):
            page = document.load_page(index)
            horizontal = page.get_text() or ""
            vertical = _reconstruct_vertical_text(page)
            if vertical:
                # Keep both: horizontal body + ordered vertical title strip.
                text = horizontal.rstrip() + "\n" + vertical
            else:
                text = horizontal
            pages.append(PageText(page_number=index + 1, text=text))
    finally:
        document.close()
    return pages


def _reconstruct_vertical_text(page: object) -> str:
    """Rebuild right-side vertical title-block lines in logical reading order.

    PyMuPDF reports rotated lines with ``dir`` ≈ (0, -1). Sorting each vertical
    column by descending Y puts ``AGENT:`` / ``DATE:`` next to their values.
    """
    try:
        data = page.get_text("dict")  # type: ignore[attr-defined]
    except Exception:
        return ""

    # (x_bucket, y0, y1, dy, text)
    vertical_lines: list[tuple[float, float, float, float, str]] = []
    page_width = float(getattr(page, "rect").width)  # type: ignore[attr-defined]

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            direction = line.get("dir") or (1.0, 0.0)
            dx, dy = float(direction[0]), float(direction[1])
            # Mostly vertical writing direction.
            if abs(dy) < 0.7:
                continue
            text = " ".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if not text:
                continue
            bbox = line.get("bbox") or (0, 0, 0, 0)
            x0, y0, _x1, y1 = (float(v) for v in bbox)
            # Prefer the right-hand title strip (ignore stray vertical notes).
            if x0 < page_width * 0.55:
                continue
            vertical_lines.append((x0, y0, y1, dy, text))

    if not vertical_lines:
        return ""

    # Cluster into columns by X, then order each column along the reading axis.
    vertical_lines.sort(key=lambda item: item[0])
    columns: list[list[tuple[float, float, float, float, str]]] = []
    for item in vertical_lines:
        if not columns or abs(item[0] - columns[-1][0][0]) > 8:
            columns.append([item])
        else:
            columns[-1].append(item)

    ordered: list[str] = []
    seen: set[str] = set()
    for column in columns:
        dy = column[0][3]
        # dir (0, -1) reads upward → descending Y; (0, +1) reads downward.
        column.sort(key=lambda item: item[1], reverse=(dy < 0))
        for _x0, _y0, _y1, _dy, text in column:
            key = text.upper()
            # Keep duplicates that are meaningful labels, but skip pure repeats
            # of the same long company/address line.
            if key in seen and len(text) > 24:
                continue
            seen.add(key)
            ordered.append(text)

    return "\n".join(ordered)


# Cheap gate before running the heavier scope parser on a page.
_SCOPE_HINT_RE = re.compile(
    r"\b(SCOPE(\s+OF\s+WORK)?|SUMMARY\s+OF\s+WORK|GENERAL\s+NOTES)\b",
    re.IGNORECASE,
)


def extract_requirement(pages: list[PageText]) -> RequirementInfo:
    """Parse requirement provider, project info, and scope sections from page text."""
    info = RequirementInfo(
        total_pages=len(pages),
        pages_with_text=sum(1 for page in pages if page.text.strip()),
    )
    if info.pages_with_text == 0:
        info.notes.append("No extractable text found in the PDF.")
        return info

    # Pick the real title-block page(s), not symbol/abbreviation sheets.
    cover_pages = _select_cover_pages(pages)
    cover_lines = _collect_lines(cover_pages)
    # Full-document lines — used to fill anything the cover pass misses.
    all_lines = _collect_lines(pages)

    # Architect CD sets (PROJECT OWNER & TITLE) vs Garland (AGENT:) sheets.
    if _is_architect_cd_set(cover_lines) or _is_architect_cd_set(all_lines[:80]):
        _extract_architect_title_block(cover_lines, info)
    else:
        _extract_provider(cover_lines, info)
        _extract_project(cover_lines, info)

    _extract_consultants(cover_lines, info)

    # If Garland-style left provider empty but architect block fields exist, try that too.
    if not info.provider_company and not info.client:
        _extract_architect_title_block(cover_lines, info)

    # Second pass: harvest missing identity fields from EVERY page (title blocks
    # repeat on each sheet; CAD text order varies page to page).
    _fill_missing_from_all_pages(pages, all_lines, info)

    # P0 / P1 enrichment across the full set.
    _extract_compliance_and_status(cover_lines, all_lines, info)
    _extract_work_type(all_lines if not info.work_type else cover_lines, info)
    if not info.district_address:
        _extract_district_address(all_lines, info)
    else:
        _extract_district_address(cover_lines, info)
    _extract_sheet_index(pages, info)
    _extract_applicable_codes(pages, info)
    _extract_project_summary(pages, info)
    _extract_limits_of_work(all_lines, info)
    _extract_bid_alternates(all_lines, info)
    _extract_multi_site_addresses(all_lines, info)

    # Only parse pages that look like they contain scope/notes (skips symbol sheets).
    scope_pages = [
        page
        for page in pages
        if page.text and _SCOPE_HINT_RE.search(page.text)
    ]
    info.scope_sections = _extract_scope_sections(scope_pages)

    if not info.scope_sections:
        info.notes.append(
            "No explicit scope section found (SCOPE OF WORK / SUMMARY OF WORK / GENERAL NOTES)."
        )
        fallback = _loose_general_notes(_collect_lines(scope_pages or pages[:5]))
        if fallback:
            info.scope_sections.append(fallback)

    return info


def _fill_missing_from_all_pages(
    pages: list[PageText],
    all_lines: list[tuple[int, str]],
    info: RequirementInfo,
) -> None:
    """Fill empty identity fields by rescanning title-block pages across the set.

    Title-block text is repeated on most sheets; scanning only the 'best' cover
    page often misses DATE / address / job when CAD extraction order differs.
    """
    title_hint = re.compile(
        r"\b(AGENT\s*:|PROJECT\s+OWNER|THE\s+GARLAND|JOB\s+NUMBER|DATE\s*:|"
        r"DSA\s*#|CONSTRUCTION\s+DOCUMENT|SCHOOL\s+DISTRICT)\b",
        re.I,
    )
    # Try title-block-like pages for any still-missing fields.
    for page in pages:
        if not page.text.strip() or not title_hint.search(page.text):
            continue
        # Skip pure abbreviation/symbol sheets — they pollute agent/title heuristics.
        upper_page = page.text.upper()
        if "ABBREVIATIONS" in upper_page and "DRAFTING ITEMS" in upper_page:
            continue
        page_lines = _collect_lines([page])
        if not page_lines:
            continue

        probe = RequirementInfo(
            provider_company=info.provider_company,
            client=info.client,
            project_title=info.project_title,
            site_address=info.site_address,
        )
        if _is_architect_cd_set(page_lines) or "PROJECT OWNER" in upper_page:
            _extract_architect_title_block(page_lines, probe)
        else:
            _extract_provider(page_lines, probe)
            _extract_project(page_lines, probe)
        _extract_consultants(page_lines, probe)
        _merge_identity(info, probe)

        # Also try architect parsers on the same page for license/job/status gaps.
        if not info.aor_license or not info.job_number:
            probe2 = RequirementInfo()
            _extract_architect_title_block(page_lines, probe2)
            _merge_identity(info, probe2)

    # Final targeted sweeps on the full line list for stubborn gaps.
    texts = [text for _, text in all_lines]
    if not info.provider_address:
        for text in texts:
            if SINGLE_LINE_ADDRESS_RE.match(text) and not re.search(
                r"\b(CA|CALIFORNIA)\b", text, re.I
            ):
                # Out-of-state single line near Garland / company is office.
                if info.provider_company and "GARLAND" in (info.provider_company or "").upper():
                    info.provider_address = re.sub(r"\s+", " ", text).strip()
                    break
                if re.search(r"\b(OHIO|OH|NY|TX|FL|IL|PA)\b", text, re.I):
                    info.provider_address = re.sub(r"\s+", " ", text).strip()
                    break
        if not info.provider_address:
            for index, text in enumerate(texts):
                if ADDRESS_RE.match(text) and index + 1 < len(texts) and (
                    ZIP_RE.search(texts[index + 1]) or STATE_ZIP_RE.search(texts[index + 1])
                ):
                    candidate = f"{text}, {texts[index + 1]}"
                    if info.site_address and _address_fingerprint(candidate) == _address_fingerprint(
                        info.site_address
                    ):
                        continue
                    info.provider_address = candidate
                    break

    if not info.date:
        info.date = _pick_labeled_or_nearby_date(texts)

    if not info.job_number:
        for text in texts:
            if JOB_NUMBER_VALUE_RE.match(text.strip()):
                info.job_number = text.strip()
                break
        if not info.job_number:
            for index, text in enumerate(texts):
                job = JOB_NUMBER_RE.match(text)
                if job:
                    value = job.group(1).strip()
                    info.job_number = value or (
                        texts[index + 1] if index + 1 < len(texts) else None
                    )
                    break

    if not info.provider_phone or not info.provider_fax:
        for text in texts:
            if not info.provider_fax:
                fax = FAX_RE.search(text)
                if fax:
                    info.provider_fax = _normalize_phone(fax.group(1))
            if not info.provider_phone and (
                "PHONE" in text.upper() or text.upper().startswith("TEL")
            ):
                phone = STANDALONE_PHONE_RE.search(text)
                if phone:
                    info.provider_phone = _normalize_phone(phone.group(0))

    if not info.provider_agent:
        for index, text in enumerate(texts):
            agent_match = AGENT_RE.match(text)
            if not agent_match:
                continue
            value = (agent_match.group(1) or "").strip()
            if value and _looks_like_person_name(value):
                info.provider_agent = value.title() if value.isupper() else value
                break
            for offset in range(1, 12):
                if index + offset >= len(texts):
                    break
                candidate = texts[index + offset]
                if candidate.isupper() and _looks_like_person_name(candidate):
                    info.provider_agent = candidate.title()
                    break
            break

    if not info.aor_license:
        for text in texts:
            if LICENSE_RE.match(text.strip()):
                info.aor_license = text.strip().upper()
                break


def _identity_snapshot(info: RequirementInfo) -> tuple:
    return (
        info.provider_company,
        info.provider_address,
        info.provider_phone,
        info.provider_agent,
        info.client,
        info.project_title,
        info.site_address,
        info.job_number,
        info.date,
        info.aor_license,
        info.document_status,
        info.dsa_number,
        len(info.consultants),
    )


def _identity_complete(info: RequirementInfo) -> bool:
    return bool(
        info.provider_company
        and info.provider_phone
        and (info.provider_address or info.provider_agent)
        and info.client
        and info.project_title
        and info.site_address
        and info.date
    )


def _merge_identity(target: RequirementInfo, source: RequirementInfo) -> None:
    """Copy non-empty source fields into empty target slots (never overwrite)."""
    # Reject common CAD form labels that look like titles / agents.
    bad_title = re.compile(
        r"^(PROJECT\s+NAME|SCHOOL\s+NAME|PROJECT\s+TITLE|SHEET\s+TITLE|"
        r"PROJECT\s+NAME/SCHOOL|NAME/SCHOOL)\b",
        re.I,
    )
    bad_agent = re.compile(
        r"\b(CODE\s+DATA|DATA|SHEET|TITLE|INDEX|LEGEND|NORTH|GARLAND)\b",
        re.I,
    )

    def _ok(field_name: str, value: str | None) -> bool:
        if not value:
            return False
        if field_name == "project_title" and bad_title.search(value):
            return False
        if field_name == "provider_agent" and (
            bad_agent.search(value) or not _looks_like_person_name(value)
        ):
            return False
        return True

    for field_name in (
        "provider_company",
        "provider_agent",
        "provider_phone",
        "provider_fax",
        "provider_address",
        "aor_license",
        "client",
        "project_title",
        "work_type",
        "site_address",
        "district_address",
        "job_number",
        "dsa_number",
        "date",
        "document_status",
    ):
        current = getattr(target, field_name)
        incoming = getattr(source, field_name)
        if current is None and _ok(field_name, incoming):
            setattr(target, field_name, incoming)
        # Replace known-bad titles/agents if we later find a better value.
        if field_name == "project_title" and current and bad_title.search(current):
            if incoming and _ok(field_name, incoming):
                setattr(target, field_name, incoming)
        if field_name == "provider_agent" and current and not _looks_like_person_name(current):
            if incoming and _ok(field_name, incoming):
                setattr(target, field_name, incoming)

    if not target.site_addresses and source.site_addresses:
        target.site_addresses = list(source.site_addresses)
    # Merge consultants by discipline+firm.
    seen = {(c.discipline.upper(), c.firm.upper()) for c in target.consultants}
    for consultant in source.consultants:
        key = (consultant.discipline.upper(), consultant.firm.upper())
        if key not in seen:
            target.consultants.append(consultant)
            seen.add(key)


def _select_cover_pages(pages: list[PageText]) -> list[PageText]:
    """Return pages that look like title blocks / cover sheets.

    Prefer a single strong title-block page. Neighbors (symbol sheets, code
    dumps) pollute date/title heuristics, so only keep them when they also
    score as title-like.
    """
    scored: list[tuple[int, PageText]] = []
    for page in pages[:12]:  # title info is almost always early
        text = page.text.upper()
        score = 0
        if "PROJECT OWNER" in text:
            score += 8
        if "SCHOOL DISTRICT" in text or "UNIFIED" in text:
            score += 6
        if "JOB NUMBER" in text or re.search(r"\b\d{5}\.\d{2}\b", text):
            score += 4
        if "AGENT:" in text or "THE GARLAND" in text:
            score += 5
        if "CONSULTANTS" in text:
            score += 3
        # Prefer true cover/index sheets over later sheets that repeat the stamp.
        if "SHEET INDEX" in text or "DRAWING INDEX" in text:
            score += 4
        if re.search(r"\b\d{5}\.\d{2}\b", text) and "CONSTRUCTION DOCUMENT" in text:
            score += 3
        if "ABBREVIATIONS" in text or "DRAFTING ITEMS" in text:
            score -= 10
        if "PARTIAL LIST OF APPLICABLE" in text or "TITLE 24" in text:
            score -= 6
        if "(E) ACCESSIBLE" in text or "SAFETY DISPERSAL AREA" in text:
            score -= 5
        if score > 0:
            scored.append((score, page))
    if not scored:
        return pages[:3]
    scored.sort(key=lambda item: (-item[0], item[1].page_number))
    best_score, best = scored[0]
    # Strong title block alone is enough (avoids abbreviations / CALGreen noise).
    if best_score >= 10:
        return [best]
    by_num = {page.page_number: page for page in pages}
    selected = [best]
    score_by_num = {page.page_number: score for score, page in scored}
    for offset in (-1, 1):
        neighbor = by_num.get(best.page_number + offset)
        if neighbor and score_by_num.get(neighbor.page_number, 0) >= 4:
            selected.append(neighbor)
    selected.sort(key=lambda page: page.page_number)
    return selected


def _is_architect_cd_set(lines: list[tuple[int, str]]) -> bool:
    texts = " ".join(text.upper() for _, text in lines)
    return "PROJECT OWNER" in texts or (
        "SCHOOL DISTRICT" in texts and "CONSULTANTS" in texts and "AGENT:" not in texts
    )


def _extract_architect_title_block(
    lines: list[tuple[int, str]], info: RequirementInfo
) -> None:
    """Parse A/E construction-document title blocks (MDUSD / BASE-style sets)."""
    texts = [text for _, text in lines]

    # --- Client: often split across lines ("MOUNT DIABLO UNIFIED" + "SCHOOL DISTRICT")
    for index, text in enumerate(texts):
        joined = text
        if index + 1 < len(texts) and "DISTRICT" in texts[index + 1].upper():
            joined = f"{text} {texts[index + 1]}"
        if re.search(r"UNIFIED\s+SCHOOL\s+DISTRICT|SCHOOL\s+DISTRICT", joined, re.I):
            if len(joined) < 120 and not DISCLAIMER_LIKE(joined):
                info.client = re.sub(r"\s+", " ", joined).strip()
                break
        if text.upper() in {"MOUNT DIABLO UNIFIED", "MT. DIABLO UNIFIED"} and index + 1 < len(
            texts
        ):
            info.client = f"{text} {texts[index + 1]}".strip()
            break

    # --- Project title: prefer full one-line title, else school code + continuation
    info.project_title = _pick_project_title(texts)

    # --- Job number like 23118.01
    for text in texts:
        if JOB_NUMBER_VALUE_RE.match(text.strip()):
            info.job_number = text.strip()
            break
    if not info.job_number:
        for index, text in enumerate(texts):
            job = JOB_NUMBER_RE.match(text)
            if job:
                value = job.group(1).strip()
                info.job_number = value or (
                    texts[index + 1] if index + 1 < len(texts) else None
                )
                break

    # --- Date: prefer issue date near job number, not code-effective / license REN dates
    info.date = _pick_issue_date(texts, info.job_number)

    # --- Site address: prefer the address near the school/project title, not consultants
    info.site_address = _pick_site_address(texts, info.project_title)

    # --- Architect / provider office: license C-##### then street + TEL
    for text in texts:
        lic = AOR_LICENSE_RE.search(text.strip())
        if lic and LICENSE_RE.match(text.strip()):
            info.aor_license = lic.group(1).upper()
            break
    if not info.aor_license:
        for text in texts:
            lic = AOR_LICENSE_RE.search(text)
            if lic:
                info.aor_license = lic.group(1).upper()
                break

    provider = _pick_architect_office(texts)
    if provider:
        company, phone, address = provider
        if company and not info.provider_company:
            info.provider_company = company
        if phone and not info.provider_phone:
            info.provider_phone = phone
        if address and not info.provider_address:
            info.provider_address = address

    # Fallback phone/address near title block if still missing
    if not info.provider_phone:
        for text in texts:
            if text.upper().startswith("TEL") or "PHONE" in text.upper():
                phone = STANDALONE_PHONE_RE.search(text)
                if phone:
                    info.provider_phone = _normalize_phone(phone.group(0))
                    break


def _extract_compliance_and_status(
    cover_lines: list[tuple[int, str]],
    early_lines: list[tuple[int, str]],
    info: RequirementInfo,
) -> None:
    """DSA application number + document issue status."""
    # Prefer title-block / cover lines; avoid site-plan building DSA tags (#01-xxxxx).
    cover_texts = [text for _, text in cover_lines]
    early_texts = [text for _, text in early_lines]

    def _is_project_dsa_line(text: str) -> bool:
        upper = text.upper()
        # Reject site-plan feature tags / keynotes that cite prior DSA apps.
        if re.search(
            r"\b(\(E\)|RAMP|RESTROOM|SOLAR|PLAYGROUND|PARKING|BUILDING|BLDG)\b",
            upper,
        ):
            return False
        if len(DSA_NUMBER_LOOSE_RE.findall(text)) > 1:
            return False
        if len(text) > 60:
            return False
        return True

    for text in cover_texts + early_texts:
        match = DSA_NUMBER_RE.search(text)
        if match and _is_project_dsa_line(text):
            info.dsa_number = match.group(1)
            break
    if not info.dsa_number:
        # Only search cover/title-block lines for unlabeled / split DSA#.
        for index, text in enumerate(cover_texts):
            if not _is_project_dsa_line(text) and not re.search(r"DSA\s*#?\s*$", text, re.I):
                continue
            upper = text.upper()
            if "DSA" not in upper:
                continue
            if len(text) > 80 and "DSA #" not in upper and "DSA#" not in upper:
                continue
            loose = DSA_NUMBER_LOOSE_RE.search(text)
            if (
                loose
                and _is_project_dsa_line(text)
                and ("DSA #" in upper or "DSA#" in upper or upper.strip().startswith("DSA"))
            ):
                info.dsa_number = loose.group(1)
                break
            if index + 1 < len(cover_texts) and re.search(r"DSA\s*#?\s*$", text, re.I):
                nxt = DSA_NUMBER_LOOSE_RE.fullmatch(cover_texts[index + 1].strip())
                if nxt and _is_project_dsa_line(cover_texts[index + 1]):
                    info.dsa_number = nxt.group(1)
                    break

    for text in cover_texts + early_texts:
        match = DOCUMENT_STATUS_RE.search(text)
        if not match:
            continue
        status = re.sub(r"\s+", " ", match.group(1)).strip().title()
        # Normalize common forms
        upper = status.upper()
        if "CONSTRUCTION DOCUMENT" in upper:
            info.document_status = "Construction Document"
        elif "BID SET" in upper or upper == "FOR BID":
            info.document_status = "Bid Set"
        elif "DSA SUBMITTAL" in upper:
            info.document_status = "DSA Submittal"
        else:
            info.document_status = status
        break


def _extract_work_type(
    cover_lines: list[tuple[int, str]], info: RequirementInfo
) -> None:
    """Roof restoration / modernization / etc. from title block."""
    candidates: list[str] = []
    if info.project_title:
        candidates.append(info.project_title)
    candidates.extend(text for _, text in cover_lines)
    for text in candidates:
        match = WORK_TYPE_RE.search(text)
        if not match:
            continue
        work = re.sub(r"\s+", " ", match.group(1)).strip()
        # Prefer fuller modernization phrasing when present in title.
        if info.project_title and "MODERNIZATION" in info.project_title.upper():
            if "SITE IMPROVEMENT" in info.project_title.upper():
                info.work_type = "Modernization & Site Improvements"
            else:
                info.work_type = "Modernization"
            return
        info.work_type = work.title() if work.isupper() else work
        return


def _extract_district_address(
    cover_lines: list[tuple[int, str]], info: RequirementInfo
) -> None:
    """Owner/district office address near the client line (not the school site)."""
    texts = [text for _, text in cover_lines]

    def _candidate_at(index: int) -> str | None:
        text = texts[index]
        candidate = None
        if ADDRESS_RE.match(text) and index + 1 < len(texts) and (
            ZIP_RE.search(texts[index + 1]) or STATE_ZIP_RE.search(texts[index + 1])
        ):
            candidate = f"{text}, {texts[index + 1]}"
        elif SINGLE_LINE_ADDRESS_RE.match(text) or (
            ZIP_RE.search(text) and re.match(r"^\d{1,6}\s+", text)
        ):
            candidate = text
        elif ADDRESS_RE.match(text) and "CONCORD" in text.upper():
            city = texts[index + 1] if index + 1 < len(texts) else ""
            candidate = f"{text}, {city}".strip(", ") if ZIP_RE.search(city) else text
        if not candidate:
            return None
        if info.site_address and _address_fingerprint(candidate) == _address_fingerprint(
            info.site_address
        ):
            return None
        if info.provider_address and _address_fingerprint(candidate) == _address_fingerprint(
            info.provider_address
        ):
            return None
        if info.site_address and "PACIFICA" in candidate.upper():
            return None
        return re.sub(r"\s+", " ", candidate).strip(" ,")

    # Strong signal: district HQ street seen in MDUSD sets.
    for index, text in enumerate(texts):
        if "CARLOTTA" in text.upper():
            candidate = _candidate_at(index)
            if candidate:
                info.district_address = candidate
                return

    # Otherwise: address within a few lines of any school-district name line.
    for index, text in enumerate(texts):
        if "SCHOOL DISTRICT" not in text.upper() and not (
            info.client
            and info.client.split()[0].upper() in text.upper()
            and DISTRICT_RE.search(text)
        ):
            continue
        for offset in range(0, 8):
            if index + offset >= len(texts):
                break
            candidate = _candidate_at(index + offset)
            if candidate:
                info.district_address = candidate
                return
            # Sometimes street sits just above the district name.
            if index - offset >= 0:
                candidate = _candidate_at(index - offset)
                if candidate:
                    info.district_address = candidate
                    return


def _address_fingerprint(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _extract_sheet_index(pages: list[PageText], info: RequirementInfo) -> None:
    """Garland SHT n - title lines and Architect sheet-index entries."""
    entries: list[str] = []
    seen: set[str] = set()

    # Scan all pages — index lines often repeat on every sheet title block / cover.
    for page in pages:
        for raw in page.lines:
            line = _clean_line(raw)
            if not line or len(line) > 90:
                continue
            sht = SHT_INDEX_RE.match(line)
            if sht:
                entry = f"SHT {sht.group(1)} — {sht.group(2).strip()}"
            else:
                # Architect sheet codes with titles (skip bare codes / abbreviations).
                sheet = SHEET_INDEX_LINE_RE.match(line)
                if not sheet:
                    continue
                title = sheet.group(2).strip()
                if len(title.split()) < 2:
                    continue
                if re.search(
                    r"\b(ABBREVIATION|SYMBOL|LEGEND|NORTH|PENNY|POUND)\b", title, re.I
                ):
                    continue
                entry = f"{sheet.group(1).upper()} — {title}"
            key = entry.upper()
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
            if len(entries) >= 40:
                info.sheet_index = entries
                return

    info.sheet_index = entries[:40]


def _extract_applicable_codes(pages: list[PageText], info: RequirementInfo) -> None:
    """Title-24 / CA code lines from early code sheets."""
    codes: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for raw in page.lines:
            line = _clean_line(raw)
            if not line or not CODE_LINE_RE.search(line):
                continue
            if DISCLAIMER_LIKE(line):
                continue
            # Prefer list-style code names; skip long compliance prose.
            if len(line) > 120 or line.count(" ") > 18:
                match = re.search(
                    r"(CALIFORNIA\s+(?:BUILDING|ELECTRICAL|MECHANICAL|PLUMBING|FIRE|"
                    r"GREEN\s+BUILDING|ENERGY|ADMINISTRATIVE)\s+CODE"
                    r"(?:\s*\([^)]{0,40}\))?)",
                    line,
                    re.I,
                )
                if not match:
                    continue
                line = match.group(1)
            # List rows typically start with the code name (optional year prefix).
            if not re.match(
                r"^(?:20\d{2}\s+)?CALIFORNIA\s+(?:BUILDING|ELECTRICAL|MECHANICAL|PLUMBING|FIRE|"
                r"GREEN\s+BUILDING|ENERGY|ADMINISTRATIVE)\s+CODE\b",
                line,
                re.I,
            ):
                continue
            cleaned = re.sub(r"\s+", " ", line).strip(" -–")
            cleaned = re.sub(r"^(?:20\d{2}\s+)", "", cleaned)
            key = cleaned.upper()
            if key in seen:
                continue
            seen.add(key)
            codes.append(cleaned)
            if len(codes) >= 8:
                info.applicable_codes = codes
                return
    info.applicable_codes = codes


def _extract_project_summary(pages: list[PageText], info: RequirementInfo) -> None:
    """Short HVAC / general project summary bullets near cover narrative."""
    bullets: list[str] = []
    seen: set[str] = set()
    capture = False
    strong_hvac = re.compile(
        r"\b(PROVIDE\s+NEW\s+HVAC|REPLACEMENT\s+IN-KIND\s+OF\s+HVAC|"
        r"NEW\s+HVAC\s+SYSTEMS?|HVAC\s+EQUIPMENT|HVAC\s+CONTROLS)\b",
        re.I,
    )
    for page in pages:  # full set — summary bullets can sit on later narrative sheets
        lines = [_clean_line(line) for line in page.lines]
        lines = [line for line in lines if line]
        for line in lines:
            upper = line.upper()
            if "GENERAL SCOPE OF WORK INCLUDES" in upper or "PROJECT DESCRIPTION" in upper:
                capture = True
                continue
            candidate = line.lstrip("-•* ").strip()
            is_bullet = line.startswith(("-", "•", "*"))
            if capture and is_bullet:
                keep = True
            elif strong_hvac.search(candidate):
                keep = True
            elif capture and len(candidate) > 40 and re.match(
                r"^(PROVIDE|INSTALL|REPLACE|REMOVE|DEMOLISH|INCLUDE)\b", candidate, re.I
            ):
                keep = True
            else:
                keep = False
            if not keep:
                if capture and (
                    SCOPE_HEADING_RE.match(line)
                    or line.upper() in {"CONSULTANTS", "REVISIONS", "SHEET INDEX"}
                ):
                    capture = False
                continue
            if len(candidate) < 28 or len(candidate) > 280:
                continue
            if DISCLAIMER_LIKE(candidate) or _looks_like_label(candidate):
                continue
            if re.search(
                r"\b(ABBREVIATION|SEE\s+MECHANICAL/PLUMBING|SEISMIC\s+ANCHORAGE|"
                r"SHALL\s+CONFORM|SHALL\s+BE\s+PROVIDED\s+BY\s+THE\s+CONTRACTOR)\b",
                candidate,
                re.I,
            ):
                continue
            key = candidate.upper()
            if key in seen:
                continue
            seen.add(key)
            bullets.append(candidate)
            if len(bullets) >= 12:
                info.project_summary = bullets
                return
    info.project_summary = bullets


def _extract_limits_of_work(
    early_lines: list[tuple[int, str]], info: RequirementInfo
) -> None:
    items: list[str] = []
    seen: set[str] = set()
    for _, text in early_lines:
        if not LIMITS_OF_WORK_RE.search(text):
            continue
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) < 8 or len(cleaned) > 160:
            continue
        if DISCLAIMER_LIKE(cleaned):
            continue
        key = cleaned.upper()
        if key in seen:
            continue
        seen.add(key)
        items.append(cleaned)
        if len(items) >= 8:
            break
    info.limits_of_work = items


def _extract_bid_alternates(
    lines: list[tuple[int, str]], info: RequirementInfo
) -> None:
    """Named bid alternates — ignore abbreviation legend / code-prose 'alternate'."""
    items: list[str] = []
    seen: set[str] = set()
    noise = re.compile(
        r"\b(MATERIALS?|DESIGNS?|METHODS?\s+OF\s+CONSTRUCTION|WASTE\s+REDUCTION|"
        r"ALUMINUM|ABBREVIATION|WORKING\s+WITH\s+LOCAL)\b",
        re.I,
    )
    for _, text in lines:
        cleaned = re.sub(r"\s+", " ", text).strip()
        match = BID_ALTERNATE_RE.match(cleaned)
        if not match:
            continue
        token = match.group(1)
        desc = (match.group(2) or "").strip()
        if token.upper() in {"ALUMINUM", "ALT", "MATERIALS", "DESIGNS"}:
            continue
        if noise.search(cleaned):
            continue
        # Require a short alternate id (A, B, 1, No. 1) — not a prose sentence.
        if len(token) > 6:
            continue
        entry = f"Alternate {token}" + (f" — {desc}" if desc else "")
        if len(entry) > 160:
            continue
        key = entry.upper()
        if key in seen:
            continue
        seen.add(key)
        items.append(entry)
        if len(items) >= 10:
            break
    info.alternates = items


def _extract_multi_site_addresses(
    cover_lines: list[tuple[int, str]], info: RequirementInfo
) -> None:
    """Collect additional campus addresses when a set covers multiple sites."""
    texts = [text for _, text in cover_lines]
    found: list[str] = []
    seen: set[str] = set()
    if info.site_address:
        seen.add(_address_fingerprint(info.site_address))
        found.append(info.site_address)

    for index, text in enumerate(texts):
        candidate = None
        if ADDRESS_RE.match(text) and index + 1 < len(texts) and ZIP_RE.search(texts[index + 1]):
            candidate = f"{text}, {texts[index + 1]}"
        elif re.match(r"^\d{1,6}\s+.+\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", text, re.I):
            candidate = text
        if not candidate:
            continue
        if re.search(r"\bSUITE\b", candidate, re.I):
            continue
        if info.district_address and _address_fingerprint(candidate) == _address_fingerprint(
            info.district_address
        ):
            continue
        if info.provider_address and _address_fingerprint(candidate) == _address_fingerprint(
            info.provider_address
        ):
            continue
        fp = _address_fingerprint(candidate)
        if fp in seen:
            continue
        seen.add(fp)
        found.append(re.sub(r"\s+", " ", candidate).strip(" ,"))
        if len(found) >= 4:
            break

    info.site_addresses = found
    if not info.site_address and found:
        info.site_address = found[0]


def DISCLAIMER_LIKE(text: str) -> bool:
    return bool(
        re.search(
            r"DSA\s+(CERTIFIED|ACCEPTED)|DOES\s+NOT\s+REPRESENT|SHALL\s+BE\s+EMPLOYED|"
            r"TESTING\s+LABORATORY|DIRECTLY\s+EMPLOYED",
            text,
            re.I,
        )
    )


def _is_path_or_url_line(text: str) -> bool:
    upper = text.upper()
    return bool(
        "://" in text
        or "/" in text
        or "\\" in text
        or upper.startswith("AUTODESK")
        or upper.startswith("WWW.")
        or re.search(r"\.(RVT|DWG|PDF|DOCX?)\b", upper)
    )


def _pick_project_title(texts: list[str]) -> str | None:
    """Build project title from title-block school name + modernization lines."""
    # Prefer a single line that already includes the work description.
    for text in texts:
        if _is_path_or_url_line(text):
            continue
        if TITLE_CONTINUATION_RE.search(text) and SCHOOL_CODE_RE.search(text):
            cleaned = re.sub(r"\s+", " ", text).strip(" &")
            if 8 < len(cleaned) < 120 and not DISTRICT_RE.search(cleaned):
                return cleaned

    title_parts: list[str] = []
    for index, text in enumerate(texts):
        if _is_path_or_url_line(text):
            continue
        # Prefer short school codes (RIO VISTA ES) over bare ELEMENTARY SCHOOL lines
        # from code sheets that lack modernization context.
        school = SCHOOL_CODE_RE.search(text)
        if not school and not re.search(r"\b(ELEMENTARY|MIDDLE|HIGH\s+SCHOOL)\b", text, re.I):
            continue
        if DISTRICT_RE.search(text):
            continue
        # Skip code-sheet school names unless followed by modernization lines.
        nxt_blob = " ".join(texts[index + 1 : index + 4])
        if not TITLE_CONTINUATION_RE.search(nxt_blob) and not TITLE_CONTINUATION_RE.search(
            text
        ):
            if re.search(r"\bELEMENTARY\s+SCHOOL\b", text, re.I) and not re.search(
                r"\bES\b", text
            ):
                continue
        title_parts = [text.strip()]
        for offset in range(1, 4):
            if index + offset >= len(texts):
                break
            nxt = texts[index + offset]
            if _is_path_or_url_line(nxt):
                break
            if TITLE_CONTINUATION_RE.search(nxt) or (
                len(nxt.split()) <= 4
                and nxt.isupper()
                and not ZIP_RE.search(nxt)
                and not JOB_NUMBER_VALUE_RE.match(nxt)
            ):
                if not DISTRICT_RE.search(nxt) and not ADDRESS_RE.match(nxt):
                    if DATE_VALUE_RE.search(nxt):
                        break
                    title_parts.append(nxt.strip())
                    continue
            break
        if len(title_parts) >= 2 or TITLE_CONTINUATION_RE.search(title_parts[0]):
            break
        title_parts = []

    if title_parts:
        return re.sub(r"\s+", " ", " ".join(title_parts)).strip(" &")
    return None


def _pick_issue_date(texts: list[str], job_number: str | None) -> str | None:
    """Pick the drawing issue date, not license renewal or code effective dates."""
    job_index = -1
    if job_number:
        for index, text in enumerate(texts):
            if text.strip() == job_number or job_number in text:
                job_index = index
                break

    candidates: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        upper = text.upper()
        if "PLOTTED" in upper or upper.startswith("REN"):
            continue
        if re.search(r"\b(EFFECTIVE|APPLICABLE\s+CODES?|APPLICABLE\s+STANDARDS)\b", upper):
            continue
        if re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", text):  # plot timestamps
            continue
        match = DATE_VALUE_RE.search(text)
        if not match:
            continue
        raw = match.group(1)
        score = 0
        # Month-name issue dates are strongest (SEPTEMBER 27, 2024).
        if re.search(r"[A-Z]{3,}", raw, re.I) and re.search(r"\d{4}", raw):
            score += 8
        if job_index >= 0:
            score += max(0, 15 - abs(index - job_index))
        # Deprioritize license-style short dates near REN.
        if index > 0 and texts[index - 1].upper().startswith("REN"):
            score -= 20
        candidates.append((score, raw))

    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    return normalize_date(candidates[0][1])


def _pick_site_address(texts: list[str], project_title: str | None) -> str | None:
    """Choose the project site address, not a consultant / architect office."""
    candidates: list[tuple[int, str]] = []
    title_index = -1
    if project_title and not _is_path_or_url_line(project_title):
        first = project_title.split()[0].upper()
        for index, text in enumerate(texts):
            if first and first in text.upper() and not _is_path_or_url_line(text):
                title_index = index
                break
    # Also anchor on school-code lines like "RIO VISTA ES".
    school_index = next(
        (
            i
            for i, text in enumerate(texts)
            if SCHOOL_CODE_RE.search(text)
            and not DISTRICT_RE.search(text)
            and not _is_path_or_url_line(text)
        ),
        -1,
    )
    if title_index < 0:
        title_index = school_index

    for index, text in enumerate(texts):
        candidate = None
        if ADDRESS_RE.match(text) and index + 1 < len(texts) and ZIP_RE.search(texts[index + 1]):
            candidate = f"{text}, {texts[index + 1]}"
        elif re.match(
            r"^\d{1,6}\s+.+\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", text, re.I
        ):
            candidate = text
        if not candidate:
            continue
        score = 0
        # Prefer addresses close to the project title / school name.
        if title_index >= 0:
            score += max(0, 20 - abs(index - title_index))
        if school_index >= 0:
            score += max(0, 12 - abs(index - school_index))
        # Prefer CA school-site style (Ave/Rd) over suite offices / architect TEL blocks.
        if re.search(r"\b(AVE|AVENUE|RD|ROAD|DR|DRIVE|BLVD)\b", candidate, re.I):
            score += 8
        if re.search(r"\bSUITE\b", candidate, re.I):
            score -= 12
        # Architect stamp address sits next to license / TEL — push it down.
        nearby = " ".join(texts[max(0, index - 3) : index + 3]).upper()
        if re.search(r"\bC-\d{4,6}\b", nearby) or re.search(r"\bTEL\b", nearby):
            score -= 10
        if "HEDDING" in candidate.upper() or "SAN JOSE" in candidate.upper():
            # Common AOR office city for this district set — not the school site.
            if title_index >= 0 and abs(index - title_index) > 2:
                score -= 6
        candidates.append((score, candidate))

    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    return candidates[0][1]


def _pick_architect_office(
    texts: list[str],
) -> tuple[str | None, str | None, str | None] | None:
    """Find architect-of-record office near license number / title-block TEL."""
    company = None
    phone = None
    address = None

    for index, text in enumerate(texts):
        if not LICENSE_RE.match(text.strip()):
            continue
        # Scan nearby for address + TEL + optional firm name.
        window = texts[max(0, index - 2) : index + 8]
        for w_index, line in enumerate(window):
            if STANDALONE_PHONE_RE.search(line) and (
                line.upper().startswith("TEL") or "PHONE" in line.upper()
            ):
                phone = _normalize_phone(STANDALONE_PHONE_RE.search(line).group(0))
            if ADDRESS_RE.match(line) and w_index + 1 < len(window) and ZIP_RE.search(
                window[w_index + 1]
            ):
                address = f"{line}, {window[w_index + 1]}"
            if COMPANY_HINTS.search(line) and not DISCLAIMER_LIKE(line):
                if 2 <= len(line.split()) <= 8 and not DISTRICT_RE.search(line):
                    company = line
        if phone or address:
            break

    # If no license block, use the TEL nearest "PROJECT OWNER" / sheet title area
    # that is NOT a consultant TEL (consultants appear after CONSULTANTS heading).
    if not phone:
        consultant_start = next(
            (i for i, t in enumerate(texts) if t.upper() == "CONSULTANTS"), len(texts)
        )
        for index, text in enumerate(texts[:consultant_start]):
            if text.upper().startswith("TEL") or text.upper().startswith("PHONE"):
                match = STANDALONE_PHONE_RE.search(text)
                if match:
                    phone = _normalize_phone(match.group(0))
                    # Address just above TEL
                    for back in range(1, 4):
                        if index - back < 0:
                            break
                        prev = texts[index - back]
                        if ADDRESS_RE.match(prev) or ZIP_RE.search(prev):
                            parts = []
                            if index - back - 1 >= 0 and ADDRESS_RE.match(
                                texts[index - back - 1]
                            ):
                                parts.append(texts[index - back - 1])
                            parts.append(prev)
                            if ZIP_RE.search(prev) is None and index - back + 1 < index:
                                pass
                            address = ", ".join(parts) if parts else address
                            # Try street + city on consecutive lines
                            if index - back - 1 >= 0 and ADDRESS_RE.match(
                                texts[index - back - 1]
                            ) and ZIP_RE.search(prev):
                                address = f"{texts[index - back - 1]}, {prev}"
                            break
                    break

    # Firm name: first design firm BEFORE consultants list that isn't an engineer discipline header
    if not company:
        consultant_start = next(
            (i for i, t in enumerate(texts) if t.upper() == "CONSULTANTS"), len(texts)
        )
        # After consultants, firms are engineers — architect is often only in stamp/address.
        # Prefer "Design" firms that appear with SF/SLO addresses matching provider phone block.
        for text in texts:
            if re.search(r"\b(ARCHITECTURE|ARCHITECTS|DESIGN)\b", text, re.I) and re.search(
                r"\b(INC|LLC|LLP|STUDIO|GROUP)\b", text, re.I
            ):
                if not DISCLAIMER_LIKE(text) and not DISTRICT_RE.search(text):
                    company = text
                    break

    if not (company or phone or address):
        return None
    return company, phone, address


def extract_requirement_from_pdf(pdf_path: Path) -> RequirementInfo:
    """Convenience: open PDF, extract text, parse requirements.

    When the AOR firm is logo-only (no searchable company text), fills
    ``provider_company`` from a stamp map and/or logo ROI OCR.
    """
    from ml.logo_firm import fill_provider_company_from_logo

    info = extract_requirement(extract_pdf_text(pdf_path))
    if not info.provider_company:
        fill_provider_company_from_logo(pdf_path, info)
    return info


def _collect_lines(pages: list[PageText]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for page in pages:
        for line in page.lines:
            cleaned = _clean_line(line)
            if cleaned and not SKIP_LINES.match(cleaned):
                result.append((page.page_number, cleaned))
    return result


def _clean_line(line: str) -> str:
    # Collapse whitespace and strip common OCR/drawing noise characters.
    text = re.sub(r"\s+", " ", line).strip()
    text = text.replace("\u0000", "")
    return text


def _extract_provider(lines: list[tuple[int, str]], info: RequirementInfo) -> None:
    texts = [text for _, text in lines]

    for index, text in enumerate(texts):
        agent_match = AGENT_RE.match(text)
        if agent_match:
            value = (agent_match.group(1) or "").strip()
            if value and _looks_like_person_name(value):
                info.provider_agent = value
            else:
                # AGENT: on its own line (or junk next); scan nearby for a person name.
                # Prefer ALL-CAPS names typical of CAD title blocks (DOUG CLARK).
                for offset in range(1, 12):
                    if index + offset >= len(texts):
                        break
                    candidate = texts[index + offset]
                    if candidate.isupper() and _looks_like_person_name(candidate):
                        info.provider_agent = candidate.title()
                        break
            break

    # Company: prefer lines with "THE ... COMPANY" / INC / LLC near phones.
    company_candidates: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        if COMPANY_HINTS.search(text) and 2 <= len(text.split()) <= 10:
            if DISTRICT_RE.search(text):
                continue
            if DISCIPLINE_RE.match(text):
                continue
            upper = text.upper()
            # Skip disclaimer / boilerplate that mentions "architect" / "company".
            if any(
                bad in upper
                for bad in (
                    "DOES NOT REPRESENT",
                    "DISCLAIMS",
                    "PRECEDENCE",
                    "WASTE MANAGEMENT",
                    "SPECIFICATIONS ARE",
                    "DESIGNER/ENGINEER",
                    "INSPECTOR OF RECORD",
                    "OWNER AND CONTRACTOR",
                )
            ):
                continue
            score = 0
            if re.search(r"\bCOMPANY\b", text, re.I):
                score += 3
            if re.search(r"\bINC\.?\b|\bLLC\b", text, re.I):
                score += 4
            if "THE " in upper and re.search(r"\b(INC|LLC|COMPANY)\b", upper):
                score += 2
            if "GARLAND" in upper:
                score += 5
            if re.search(r"\bDESIGN\b", upper) and re.search(r"\bINC\b", upper):
                score += 3
            # Bare "architect/engineer" words without a firm suffix are weak.
            if score < 2:
                continue
            company_candidates.append((score, text))
    if company_candidates:
        company_candidates.sort(key=lambda item: (-item[0], texts.index(item[1])))
        info.provider_company = company_candidates[0][1]

    # Phone / fax near the company block.
    for text in texts:
        fax = FAX_RE.search(text)
        if fax and not info.provider_fax:
            info.provider_fax = _normalize_phone(fax.group(1))
        phone = PHONE_RE.search(text)
        if phone and not info.provider_phone:
            # Prefer PHONE/TEL labeled numbers over bare matches on other pages.
            if "PHONE" in text.upper() or "TEL" in text.upper() or not info.provider_phone:
                info.provider_phone = _normalize_phone(phone.group(1))

    # Address: street + city/state/zip (two-line or Garland single-line with OHIO).
    company_index = (
        texts.index(info.provider_company)
        if info.provider_company and info.provider_company in texts
        else -1
    )
    for index, text in enumerate(texts):
        candidate = None
        if SINGLE_LINE_ADDRESS_RE.match(text):
            candidate = re.sub(r"\s+", " ", text).strip()
        elif ADDRESS_RE.match(text) and index + 1 < len(texts):
            next_line = texts[index + 1]
            if ZIP_RE.search(next_line) or STATE_ZIP_RE.search(next_line):
                candidate = f"{text}, {next_line}"
        if not candidate:
            continue
        # Prefer address near the company / Garland block.
        if company_index >= 0 and abs(index - company_index) <= 8:
            info.provider_address = candidate
            break
        if info.provider_address is None:
            info.provider_address = candidate

    if not info.provider_agent:
        # Only guess a bare person name on Garland-style sheets that have AGENT:.
        has_agent_label = any(AGENT_RE.match(text) for text in texts)
        if has_agent_label:
            for text in texts:
                if text.isupper() and _looks_like_person_name(text):
                    info.provider_agent = text.title()
                    break


def _extract_project(lines: list[tuple[int, str]], info: RequirementInfo) -> None:
    texts = [text for _, text in lines]

    for index, text in enumerate(texts):
        if DISTRICT_RE.search(text) and len(text) > 10 and len(text) < 120:
            # Prefer short district names over long note sentences mentioning "district".
            if text.upper().count(" ") <= 10 and not NUMBERED_ITEM_RE.match(text):
                info.client = text
                break
        if re.match(r"^PROJECT\s+OWNER", text, re.I) and index + 1 < len(texts):
            # Prefer a district/owner name; never treat street addresses as client.
            for offset in range(1, 8):
                if index + offset >= len(texts):
                    break
                candidate = texts[index + offset]
                if _looks_like_label(candidate) or SKIP_LINES.match(candidate):
                    continue
                if ADDRESS_RE.match(candidate) or ZIP_RE.search(candidate) or STATE_ZIP_RE.search(
                    candidate
                ):
                    continue
                if DISTRICT_RE.search(candidate):
                    info.client = candidate
                    break
            break

    # Project title: school/project name lines that aren't the district/address.
    # Prefer short title-like lines; skip numbered scope items and long sentences.
    title_candidates: list[str] = []
    for text in texts:
        upper = text.upper()
        if DISTRICT_RE.search(text):
            continue
        if (
            ADDRESS_RE.match(text)
            or ZIP_RE.search(text)
            or STATE_ZIP_RE.search(text)
            or SINGLE_LINE_ADDRESS_RE.match(text)
        ):
            continue
        if COMPANY_HINTS.search(text) and "SCHOOL" not in upper:
            continue
        if NUMBERED_ITEM_RE.match(text):
            continue
        if len(text) > 80 or len(text.split()) > 12:
            continue
        if any(
            key in upper
            for key in (
                "COVER SHEET",
                "CONSTRUCTION DOCUMENTS",
                "DRAWING INDEX",
                "SCOPE",
                "REVISION",
                "SHEET",
                "PHONE",
                "TEL ",
                "FAX",
            )
        ):
            continue
        if re.search(
            r"\b(ELEMENTARY|MIDDLE|HIGH\s+SCHOOL|SCHOOL|ROOF|MODERNIZATION|"
            r"RESTORATION|REPLACEMENT|BID)\b",
            text,
            re.I,
        ):
            title_candidates.append(text)
    if title_candidates:
        # Prefer school-name style titles over long descriptive ones.
        def _title_score(candidate: str) -> tuple[int, int]:
            upper = candidate.upper()
            schoolish = 1 if re.search(r"\b(ELEMENTARY|MIDDLE|HIGH|SCHOOL)\b", upper) else 0
            return (schoolish, -len(candidate))

        title_candidates.sort(key=_title_score, reverse=True)
        info.project_title = title_candidates[0]

    # Site address — two-line (street + city/state/zip) or single-line forms.
    for index, text in enumerate(texts):
        if ADDRESS_RE.match(text) and index + 1 < len(texts) and (
            ZIP_RE.search(texts[index + 1]) or STATE_ZIP_RE.search(texts[index + 1])
        ):
            candidate = f"{text}, {texts[index + 1]}"
            if candidate != info.provider_address:
                info.site_address = candidate
                break
        if SINGLE_LINE_ADDRESS_RE.match(text) and text != info.provider_address:
            # Prefer CA school sites over out-of-state office lines.
            if re.search(r"\b(CA|CALIFORNIA)\b", text, re.I) or info.site_address is None:
                if info.provider_address and _address_fingerprint(text) == _address_fingerprint(
                    info.provider_address
                ):
                    continue
                # Don't overwrite a CA site with the Cleveland office.
                if info.site_address and "CA" in info.site_address.upper() and "CA" not in text.upper():
                    continue
                if "CA" in text.upper() or "CALIFORNIA" in text.upper():
                    info.site_address = text
                    break
                if info.site_address is None and text != info.provider_address:
                    # Defer non-CA single-line (likely office) unless nothing else found.
                    pass
        if ZIP_RE.search(text) and ADDRESS_RE.match(text) is None:
            if re.search(r"\d", text) and info.site_address is None:
                if text != info.provider_address:
                    info.site_address = text

    for index, text in enumerate(texts):
        job = JOB_NUMBER_RE.match(text)
        if job:
            value = job.group(1).strip()
            if value:
                info.job_number = value
            elif index + 1 < len(texts):
                info.job_number = texts[index + 1]
            break

    info.date = _pick_labeled_or_nearby_date(texts)


def _pick_labeled_or_nearby_date(texts: list[str]) -> str | None:
    """Resolve DATE: even when CAD order puts CHK BY / DWG BY before the value."""
    for index, text in enumerate(texts):
        date_label = DATE_LABEL_RE.match(text)
        if not date_label:
            continue
        value = (date_label.group(1) or "").strip()
        if value and DATE_VALUE_RE.search(value) and not _looks_like_label(value):
            return normalize_date(DATE_VALUE_RE.search(value).group(1))
        for offset in range(1, 20):
            if index + offset >= len(texts):
                break
            candidate = texts[index + offset]
            if _looks_like_label(candidate) or SKIP_LINES.match(candidate):
                continue
            if re.search(r"\b(CHK|DWG|DRAWN|BY)\b", candidate, re.I) and ":" in candidate:
                continue
            if AGENT_RE.match(candidate) or COMPANY_HINTS.search(candidate):
                continue
            match = DATE_VALUE_RE.search(candidate)
            if match and len(candidate) <= 40:
                return normalize_date(match.group(1))
        break
    for text in texts:
        if _looks_like_label(text) or re.search(r"\b(CHK|DWG|PLOTTED|REN)\b", text, re.I):
            continue
        match = DATE_VALUE_RE.search(text)
        if match and len(text) <= 40:
            return normalize_date(match.group(1))
    return None


def _extract_consultants(lines: list[tuple[int, str]], info: RequirementInfo) -> None:
    texts = [text for _, text in lines]
    consultants: list[Consultant] = []
    index = 0
    while index < len(texts):
        discipline_match = DISCIPLINE_RE.match(texts[index])
        if not discipline_match:
            # Spaced-out discipline headers like "C I V I L   E N G I N E E R"
            spaced = re.sub(r"\s+", "", texts[index])
            spaced_match = re.match(
                r"(CIVIL|STRUCTURAL|MECHANICAL|ELECTRICAL|TECHNOLOGY)ENGINEERS?$",
                spaced,
                re.I,
            )
            if spaced_match:
                discipline = spaced_match.group(1).title() + " Engineer"
            else:
                index += 1
                continue
        else:
            discipline = texts[index].title()

        firm = None
        phone = None
        address_parts: list[str] = []
        for offset in range(1, 6):
            if index + offset >= len(texts):
                break
            candidate = texts[index + offset]
            if DISCIPLINE_RE.match(candidate) or re.sub(r"\s+", "", candidate).upper().endswith(
                "ENGINEER"
            ):
                break
            if COMPANY_HINTS.search(candidate) or (
                firm is None and not STANDALONE_PHONE_RE.search(candidate)
            ):
                if firm is None and not _looks_like_label(candidate):
                    firm = candidate
                    continue
            phone_match = STANDALONE_PHONE_RE.search(candidate)
            if phone_match and phone is None:
                phone = _normalize_phone(phone_match.group(0))
                continue
            if ADDRESS_RE.match(candidate) or ZIP_RE.search(candidate):
                address_parts.append(candidate)
        if firm:
            consultants.append(
                Consultant(
                    discipline=discipline,
                    firm=firm,
                    phone=phone,
                    address=", ".join(address_parts) if address_parts else None,
                )
            )
        index += 1

    info.consultants = consultants


def _collect_numbered_items(lines: list[str], start: int = 0, stop_at_heading: bool = True) -> list[str]:
    """Collect numbered scope items, supporting ``1.`` / body-on-next-line layout."""
    items: list[str] = []
    current_parts: list[str] = []
    awaiting_body = False
    index = start

    def flush() -> None:
        nonlocal current_parts, awaiting_body
        text = " ".join(part for part in current_parts if part.strip()).strip()
        if text:
            items.append(text)
        current_parts = []
        awaiting_body = False

    while index < len(lines):
        line = lines[index]
        if stop_at_heading and _is_scope_heading_line(line):
            break
        if re.match(
            r"^(OF\s+SHEET|REVISION|DATE:|CHK\s*BY|DWG\s*BY|PHONE|TEL\b|THE\s+GARLAND)",
            line,
            re.I,
        ):
            # Title-block chrome ends the item stream on many sheets.
            if current_parts or awaiting_body:
                flush()
            if stop_at_heading:
                break
            index += 1
            continue

        numbered = NUMBERED_ITEM_RE.match(line)
        if numbered:
            flush()
            body = numbered.group(2).strip()
            if body:
                current_parts = [body]
                awaiting_body = False
            else:
                current_parts = []
                awaiting_body = True
            index += 1
            continue

        if awaiting_body or current_parts:
            if SKIP_LINES.match(line) or _looks_like_label(line):
                index += 1
                continue
            # Don't swallow short company / district stamp lines into scope items.
            # Mid-sentence uses of "DISTRICT" (e.g. "with the district representative")
            # must remain part of the current bullet.
            if COMPANY_HINTS.search(line) and len(line.split()) <= 6:
                flush()
                index += 1
                continue
            if DISTRICT_RE.search(line) and (
                len(line.split()) <= 6
                or re.search(r"\b(UNIFIED\s+)?SCHOOL\s+DISTRICT\b", line, re.I)
                and not _SCOPE_WORK_VERB_RE.search(line)
            ):
                flush()
                index += 1
                continue
            current_parts.append(line)
            awaiting_body = False
        index += 1

    flush()
    return items


_SCOPE_JUNK_ITEM_RE = re.compile(
    r"^(TYP\.?|XP|BLDG\.?\s*#?|V\.?I\.?F\.?|\d+[A-Z]?-\d+[A-Z]?|"
    r"\d{1,3}(\.\d+)?|N/?A)$|"
    r"V\.?I\.?F\.?|TYP\.\s*\d|^\(?[EN]\)\s|"
    r"\bSEE\s+PLANS\s+AND\s+SPECIFICATIONS\s+FOR\s+ALL\s+OTHER\b",
    re.IGNORECASE,
)

# Title-block / sidebar text that must never appear as scope bullets.
_SCOPE_TITLEBLOCK_JUNK_RE = re.compile(
    r"^(PHONE|TEL|FAX|AGENT|DATE|CHK\s*BY|DWG\s*BY|DRAWN\s*BY|REVISION|"
    r"SHEET|OF|SINCE\s+\d{4})\b|"
    r"\b(PHONE|FAX)\s*\(?\d{3}\)?|"
    r"\bTHE\s+GARLAND\s+COMPANY\b|\bGARLAND\b$|"
    r"\bSCHOOL\s+DISTRICT\b|"
    r"\b(ELEMENTARY|MIDDLE|HIGH)\s+SCHOOL\b|"
    r"^\d{1,6}\s+.+\b(OHIO|CA|CALIFORNIA|OH)\b.+\d{5}|"
    r"^[A-Z]{2,4}$|"  # initials like DC / GCK
    r"^\d{1,2}-\d{1,2}-\d{2,4}$",  # bare dates
    re.IGNORECASE,
)

_SCOPE_WORK_VERB_RE = re.compile(
    r"\b(REMOVE|DISPOSE|INSTALL|NAIL|FASTEN|PERFORM|APPLY|CLEAN|TEST|ENSURE|"
    r"REPLACE|PROVIDE|PATCH|PAINT|POWER\s+WASH|MECHANICALLY|FLOOD\s+COAT|"
    r"RECEIVE|COORDINATE|DEMOLISH|SEAL|PRIME|SAND|REPAIR|ESTABLISH|FURNISH|"
    r"COMPLY|REVIEW|HOLD|SLEEVED|PROHIBITED|RE-?CABLE|CUTOVER)\b",
    re.IGNORECASE,
)

# Symbol legend / product schedule cells that follow real scope lists on Architect CDs.
_EQUIPMENT_SCHEDULE_RE = re.compile(
    r"^(DATA LOCATION|SEE\s+\d{2}\s+\d{2}|PANDUIT|OFCI|ARUBA|CHATSWORTH|"
    r"CAT6A DATA DROP LOCATION|INTERIOR WIRELESS|EXTERIOR WIRELESS|"
    r"LADDER RACK|INTERCOM CONSOLE|T-\d{3}\b|NQ-CON|"
    r"TECHNOLOGY INFRASTRUCTURE\s+(FLOOR|RCP|SITE|DETAILS|SINGLE|COVER)|"
    r"TECHNOLOGY SHEET INDEX|TECHNOLOGY SYMBOL LEGEND|"
    r"Autodesk Docs:|WRITTEN DIMENSIONS ON THESE DRAWINGS|"
    r"THESE PLANS OR THE SPECIFICATIONS ARE SUITABLE)\b",
    re.IGNORECASE,
)


def _is_titleblock_scope_junk(text: str) -> bool:
    """True when a 'scope item' is actually title-block / contact chrome."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return True
    if _SCOPE_TITLEBLOCK_JUNK_RE.search(cleaned):
        return True
    if STANDALONE_PHONE_RE.search(cleaned) and not _SCOPE_WORK_VERB_RE.search(cleaned):
        return True
    if COMPANY_HINTS.search(cleaned) and not _SCOPE_WORK_VERB_RE.search(cleaned):
        return True
    if DISTRICT_RE.search(cleaned) and not _SCOPE_WORK_VERB_RE.search(cleaned):
        return True
    if SINGLE_LINE_ADDRESS_RE.match(cleaned) or (
        ADDRESS_RE.match(cleaned) and not _SCOPE_WORK_VERB_RE.search(cleaned)
    ):
        return True
    return False


def _is_scope_prose(line: str) -> bool:
    """True for sentence-like scope bullets (not keynotes / dimensions)."""
    if len(line) < 28:
        return False
    if _SCOPE_JUNK_ITEM_RE.search(line) or _is_titleblock_scope_junk(line):
        return False
    if re.search(r'[�"]?\d+\'-\d+', line):  # CAD dimensions
        return False
    if re.fullmatch(r"[\d.\sA-Z/-]{1,20}", line):
        return False
    # Prefer imperative / work-description language.
    if not re.match(r"^[A-Za-z(]", line):
        return False
    letters = sum(1 for ch in line if ch.isalpha())
    if letters < 18 or letters / max(len(line), 1) <= 0.45:
        return False
    # Require a work verb unless the line is clearly a continuation clause.
    return bool(_SCOPE_WORK_VERB_RE.search(line)) or len(line) > 60


def _collect_scope_prose(lines: list[str], limit: int = 25) -> list[str]:
    """Collect unnumbered prose under a scope heading (common on CD sets)."""
    items: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        buffer = []
        if _is_scope_prose(text):
            items.append(text)

    for line in lines:
        if _is_scope_heading_line(line):
            flush()
            break
        # Stop at title-block sidebar — never merge phone/school into scope.
        if _is_titleblock_scope_junk(line) or _looks_like_label(line) or SKIP_LINES.match(
            line
        ):
            flush()
            if _is_titleblock_scope_junk(line) or _looks_like_label(line):
                # Once we hit the stamp block, remaining lines are not scope.
                if re.search(
                    r"\b(PHONE|FAX|AGENT|GARLAND|CHK\s*BY|DWG\s*BY)\b", line, re.I
                ):
                    break
            continue
        numbered = NUMBERED_ITEM_RE.match(line)
        if numbered and not (numbered.group(2) or "").strip():
            # Bare "1." / "2." markers — body is prose below.
            flush()
            continue
        if _SCOPE_JUNK_ITEM_RE.search(line) or re.search(r'[�"]?\d+\'-\d+', line):
            flush()
            continue
        if re.match(r"^[A-Za-z(]", line) and len(line) > 15:
            # New sentence vs wrapped continuation.
            if buffer and (
                line[:1].isupper()
                and buffer[-1].endswith((".", ";", ":"))
            ):
                flush()
            buffer.append(line)
            if line.endswith((".", ";", ":")):
                flush()
        else:
            flush()
        if len(items) >= limit:
            break
    flush()
    return items[:limit]


def _looks_like_equipment_schedule(line: str) -> bool:
    """True for product/schedule/sheet-index rows that follow real scope lists."""
    text = re.sub(r"\s+", " ", line).strip()
    if not text:
        return True
    if _EQUIPMENT_SCHEDULE_RE.search(text):
        return True
    # Short manufacturer / part-number cells.
    if len(text) <= 24 and not NUMBERED_ITEM_RE.match(text):
        if re.fullmatch(r"[A-Z0-9][A-Z0-9\-_/]{1,20}", text):
            return True
        if text.upper() in {"AP", "WP", "N/A", "OFCI", "OFCI.", "EMT", "GRC"}:
            return True
    return False


def _trim_scope_body_window(lines: list[str], *, max_lines: int = 90) -> list[str]:
    """Keep leading intro + contiguous numbered scope; cut schedule / far noise."""
    if not lines:
        return []
    capped = lines[:max_lines]
    out: list[str] = []
    seen_number = False
    for line in capped:
        if _is_scope_heading_line(line):
            break
        if _looks_like_equipment_schedule(line) and seen_number:
            break
        if _looks_like_equipment_schedule(line) and not seen_number:
            # Schedule before any numbered item — not a scope body.
            break
        numbered = NUMBERED_ITEM_RE.match(line)
        if numbered:
            seen_number = True
            out.append(line)
            continue
        if not seen_number:
            # Short section intros (PRE-CON MEETING REQUIREMENTS, etc.).
            if len(line) <= 90:
                out.append(line)
            continue
        # Continuation / body under current number.
        if _is_titleblock_scope_junk(line) or SKIP_LINES.match(line):
            break
        out.append(line)
    return out


def _trailing_numbered_block(lines: list[str], *, max_lines: int = 140) -> list[str]:
    """Numbered scope list immediately above a heading (CAD reading order)."""
    if not lines:
        return []
    window = lines[-max_lines:]
    num_indexes = [i for i, line in enumerate(window) if NUMBERED_ITEM_RE.match(line)]
    if not num_indexes:
        return []
    # Start at the last "1." that still has following numbers (or is the only item).
    start = num_indexes[0]
    for i in num_indexes:
        if NUMBERED_ITEM_RE.match(window[i]).group(1) == "1":
            start = i
    # Keep through end of window so wrapped body after the last number is included.
    return window[start:]


def _filter_quality_scope_items(items: list[str]) -> list[str]:
    """Drop keynote / title-block / dimension garbage from scope lists."""
    kept: list[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", item).strip()
        if len(text) < 20:
            continue
        if _SCOPE_JUNK_ITEM_RE.search(text) or _is_titleblock_scope_junk(text):
            continue
        if _looks_like_equipment_schedule(text):
            continue
        if re.search(r'[�"]?\d+\'-\d+', text) and len(text) < 80:
            continue
        if re.search(
            r"\bTECHNOLOGY INFRASTRUCTURE\s+(FLOOR|RCP|SITE|DETAILS|SINGLE|COVER)\b",
            text,
            re.I,
        ):
            continue
        # Reject items that are mostly sheet/keynote tokens.
        tokens = text.split()
        alpha_words = [t for t in tokens if re.search(r"[A-Za-z]{3,}", t)]
        if len(alpha_words) < 3:
            continue
        # Prefer real work descriptions.
        if not _SCOPE_WORK_VERB_RE.search(text) and len(text) < 50:
            continue
        kept.append(text)
    return kept


def _score_scope_items(items: list[str]) -> int:
    """Higher = more likely a real scope-of-work list."""
    if not items:
        return -1
    score = 0
    for item in items:
        if _SCOPE_WORK_VERB_RE.search(item):
            score += 3
        if _is_titleblock_scope_junk(item):
            score -= 5
        if len(item) > 40:
            score += 1
    return score


def _normalize_scope_item_key(item: str) -> str:
    return re.sub(r"\s+", " ", item).strip().upper()[:96]


def _scope_items_overlap(a: list[str], b: list[str], *, threshold: float = 0.7) -> bool:
    """True when two scope lists are largely the same content."""
    if not a or not b:
        return False
    set_a = {_normalize_scope_item_key(x) for x in a}
    set_b = {_normalize_scope_item_key(x) for x in b}
    overlap = len(set_a & set_b)
    return overlap >= threshold * min(len(set_a), len(set_b))


def _is_generic_scope_title(title: str) -> bool:
    """Bare 'Scope of Work' / 'SCOPE OF WORK: …' without a trade-specific prefix."""
    cleaned = title.strip().rstrip(":").strip()
    return bool(
        re.match(
            r"^(?:SHT\s+\d+\s*[-–]?\s*)?SCOPE(?:\s+OF\s+WORK)?(?:\s*[:\-].*)?$",
            cleaned,
            re.I,
        )
    )


def _dedupe_scope_sections(sections: list[ScopeSection]) -> list[ScopeSection]:
    """Drop sections that only repeat another section's bullets.

    Prefer the more specific title (e.g. keep 'Liquid Applied… Scope', drop a
    bare 'Scope of Work' that copied the same items).
    """
    unique: list[ScopeSection] = []
    for section in sections:
        title_key = section.title.upper().strip()
        item_key = tuple(_normalize_scope_item_key(x) for x in section.items[:3])
        # Exact title+lead-items duplicate.
        if any(
            other.title.upper().strip() == title_key
            and tuple(_normalize_scope_item_key(x) for x in other.items[:3]) == item_key
            for other in unique
        ):
            continue

        overlap_index = next(
            (
                i
                for i, other in enumerate(unique)
                if _scope_items_overlap(section.items, other.items)
            ),
            None,
        )
        if overlap_index is None:
            unique.append(section)
            continue

        existing = unique[overlap_index]
        # Same bullets under a generic heading → skip the generic copy.
        if _is_generic_scope_title(section.title) and not _is_generic_scope_title(
            existing.title
        ):
            continue
        if _is_generic_scope_title(existing.title) and not _is_generic_scope_title(
            section.title
        ):
            unique[overlap_index] = section
            continue
        # Both specific or both generic: keep the longer / first list.
        if len(section.items) > len(existing.items):
            unique[overlap_index] = section
    return unique


def _extract_scope_sections(pages: list[PageText]) -> list[ScopeSection]:
    sections: list[ScopeSection] = []
    for page in pages:
        lines = [_clean_line(line) for line in page.lines]
        lines = [line for line in lines if line]

        heading_indexes = [
            i
            for i, line in enumerate(lines)
            if _is_scope_heading_line(line)
        ]

        if heading_indexes:
            for position, heading_index in enumerate(heading_indexes):
                title = lines[heading_index].rstrip(":").strip() or "Scope of Work"
                # Cap far-away duplicate headings (CAD often repeats titles at sheet end).
                hard_cap = min(len(lines), heading_index + 90)
                if position + 1 < len(heading_indexes):
                    next_heading = min(heading_indexes[position + 1], hard_cap)
                else:
                    next_heading = hard_cap

                raw_after = lines[heading_index + 1 : next_heading]
                window_after = _trim_scope_body_window(raw_after)

                after_items = _filter_quality_scope_items(
                    _collect_numbered_items(window_after, stop_at_heading=False)
                )
                prose_after = _filter_quality_scope_items(
                    _collect_scope_prose(window_after)
                )

                # CAD order: the visual numbered list under a trade SCOPE title is
                # often extracted ABOVE the heading. Prefer that primary 1..N list
                # exactly — do not merge later sub-lists (PRE-CON / standards) that
                # restart at 1. under the same title block.
                above_items: list[str] = []
                if position == 0 and not _is_generic_scope_title(title):
                    above_start = max(0, heading_index - 140)
                    above_block = _trailing_numbered_block(
                        lines[above_start:heading_index]
                    )
                    above_items = _filter_quality_scope_items(
                        _collect_numbered_items(above_block, stop_at_heading=False)
                    )

                candidates: list[list[str]] = [after_items, prose_after]
                if above_items:
                    candidates.append(above_items)

                sole_or_first = position == 0 or len(heading_indexes) == 1
                after_weak = _score_scope_items(after_items) < 3 and _score_scope_items(
                    prose_after
                ) < 3
                # Garland-only fallback: sole generic/replacement heading with body above.
                if sole_or_first and after_weak and not above_items:
                    window_before = _trim_scope_body_window(
                        lines[max(0, heading_index - 80) : heading_index]
                    )
                    before_items = _filter_quality_scope_items(
                        _collect_numbered_items(window_before, stop_at_heading=False)
                    )
                    prose_before = _filter_quality_scope_items(
                        _collect_scope_prose(window_before)
                    )
                    candidates.extend([before_items, prose_before])
                    if len(heading_indexes) == 1:
                        page_items = _filter_quality_scope_items(
                            _collect_numbered_items(
                                _trim_scope_body_window(lines, max_lines=120),
                                stop_at_heading=False,
                            )
                        )
                        candidates.append(page_items)

                # Prefer the primary CAD-above list when it is clearly the main body
                # for this trade heading (matches the drawing under that title).
                if (
                    above_items
                    and _score_scope_items(above_items) >= 3
                    and len(above_items) >= max(3, len(after_items))
                ):
                    items = above_items
                else:
                    items = max(
                        candidates,
                        key=lambda candidate: (
                            _score_scope_items(candidate),
                            len(candidate),
                        ),
                    )
                if _score_scope_items(items) < 3:
                    items = []
                if items:
                    sections.append(
                        ScopeSection(
                            title=title, items=items, page_number=page.page_number
                        )
                    )
        else:
            # Pages without a heading but dense numbered lists (rare) — skip.
            pass

    return _dedupe_scope_sections(sections)


def _loose_general_notes(lines: list[tuple[int, str]]) -> ScopeSection | None:
    """Last-resort: gather lines under a GENERAL NOTES header without numbering."""
    collecting = False
    page_number = 1
    items: list[str] = []
    for page, text in lines:
        if SCOPE_FALLBACK_RE.match(text):
            collecting = True
            page_number = page
            continue
        if collecting:
            if _is_scope_heading_line(text) or _looks_like_label(text):
                break
            if len(text) > 20:
                items.append(text)
            if len(items) >= 15:
                break
    if not items:
        return None
    return ScopeSection(
        title="General Notes",
        items=items,
        page_number=page_number,
        notes="Captured as free-text notes (no numbered scope list found).",
    )


def _looks_like_label(text: str) -> bool:
    return bool(
        re.match(
            r"^(DATE(\s+PLOTTED)?|JOB\s+NUMBER|SHEET(\s+TITLE)?|SHEET\s+NO\.?|"
            r"REVISION|AGENT|PHONE|TEL|FAX|SCALE|DRAWN\s+BY|CHK\s*BY|DWG\s*BY|"
            r"PROJECT\s+OWNER|DETAIL\s+VIEW|ROOM\s+NUMBER|FILE\s+LOCATION)\s*:?\s*$",
            text,
            re.I,
        )
    )


_PERSON_NAME_BLOCKLIST = re.compile(
    r"\b(ELEMENTARY|MIDDLE|HIGH|SCHOOL|DISTRICT|APPROVAL|STAMP|SHEET|TITLE|"
    r"DETAIL|VIEW|ROOM|NUMBER|AGENCY|PROFESSIONAL|COVER|PLAN|INDEX|"
    r"GARLAND|COMPANY|ENGINEER|ARCHITECT|CENTER|CENTRE|PLOTTED|DATE|"
    r"CONSTRUCTION|DOCUMENTS|INFRASTRUCTURE|UPGRADE|PROJECT|DRAFTING|"
    r"ITEMS|CONTRACTOR|FURNISHED|PROTECTION|ASSOCIATION|POINT|WORK|"
    r"ACADEMY|LANGUAGE|HOMES|VISTA|SALAZAR'S|CODE|DATA|LEGEND|NORTH|"
    r"ASSISTIVE|LISTENING|SYSTEM|DEVICE|ALARM|FIRE|SMOKE|PANEL)\b",
    re.IGNORECASE,
)


def _looks_like_person_name(text: str) -> bool:
    """Two/three-token personal names (e.g. DOUG CLARK), not sheet chrome."""
    cleaned = text.strip(" :.-")
    if not cleaned or len(cleaned) > 40:
        return False
    if _looks_like_label(cleaned) or COMPANY_HINTS.search(cleaned) or DISTRICT_RE.search(cleaned):
        return False
    if _PERSON_NAME_BLOCKLIST.search(cleaned):
        return False
    if STANDALONE_PHONE_RE.search(cleaned) or ZIP_RE.search(cleaned):
        return False
    if len(cleaned) <= 2 or not re.search(r"[A-Za-z]{2,}", cleaned):
        return False
    tokens = cleaned.split()
    if not (2 <= len(tokens) <= 3):
        return False
    # Reject school codes like "RIO VISTA ES" / "MEADOW HOMES MS".
    if tokens[-1].upper() in {"ES", "MS", "HS", "ES.", "MS.", "HS."}:
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", token) for token in tokens)


def _normalize_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return raw.strip()
