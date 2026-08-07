"""Generate a clear, human-readable requirement PDF from RequirementInfo.

Layout:
  1. Who provided the requirement (provider / AOR)
  2. Project details (work type, document status, dates)
  3. Consultants (if any)
  4. Drawing / sheet index (if any)
  5. Project / HVAC summary (if any)
  6. Applicable codes (if any)
  7. Scope of work
  8. Bid alternates (if any)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ml.requirement_extractor import Consultant, RequirementInfo, ScopeSection

# Drawing-index dumps look like "2MP-001 LEGENDS … 2MP-002 SCHEDULES …"
SHEET_CODE_RE = re.compile(r"\b\d?[A-Z]{1,3}-?\d{2,3}[A-Z]?\b")
BAD_META_RE = re.compile(
    r"^(SHEET(\s+TITLE)?|JOB\s+NUMBER|DATE(\s+PLOTTED)?|REVISION|"
    r"DETAIL\s+VIEW|OF|N/?A|—|-|CHK\s*BY|DWG\s*BY|DRAWN\s+BY|"
    r"BREAK\s+LINE|NOT\s+FOR|PRODUCT\s+SUBMITTALS|COLUMN\s+GRID|"
    r"ACOUSTICAL\s+CEILING|G\s+OR\s+GB)$",
    re.I,
)
DISCLAIMER_RE = re.compile(
    r"DSA\s+CERTIFIED|DOES\s+NOT\s+REPRESENT|SHALL\s+BE\s+EMPLOYED|"
    r"WASTE\s+MANAGEMENT|PROPERTY\s+LINE|TESTING\s+LABORATORY",
    re.I,
)


def requirement_pdf_filename(source_filename: str) -> str:
    """``abc.pdf`` -> ``abc requirement.pdf``."""
    stem = Path(source_filename).stem
    return f"{stem} requirement.pdf"


def sanitize_requirement(info: RequirementInfo) -> RequirementInfo:
    """Drop noisy fields / sheet-index dumps so the PDF stays readable."""
    site_addresses = _clean_string_list(info.site_addresses, max_len=120, max_items=4)
    site_address = _clean_field(info.site_address, max_len=120)
    if not site_address and site_addresses:
        site_address = site_addresses[0]

    cleaned = RequirementInfo(
        provider_company=_clean_field(info.provider_company, max_len=80),
        provider_agent=_clean_field(info.provider_agent, max_len=60, personish=True),
        provider_phone=_clean_field(info.provider_phone, max_len=40),
        provider_fax=_clean_field(info.provider_fax, max_len=40),
        provider_address=_clean_field(info.provider_address, max_len=120),
        aor_license=_clean_field(info.aor_license, max_len=20),
        client=_clean_field(info.client, max_len=100, require_not_disclaimer=True),
        project_title=_clean_field(info.project_title, max_len=100, require_not_disclaimer=True),
        work_type=_clean_field(info.work_type, max_len=80),
        site_address=site_address,
        site_addresses=site_addresses,
        district_address=_clean_field(info.district_address, max_len=120),
        job_number=_clean_job_number(info.job_number),
        dsa_number=_clean_dsa_number(info.dsa_number),
        date=_clean_field(info.date, max_len=40),
        document_status=_clean_field(info.document_status, max_len=60),
        consultants=_dedupe_consultants(info.consultants),
        sheet_index=_clean_string_list(info.sheet_index, max_len=100, max_items=40),
        project_summary=_clean_string_list(info.project_summary, max_len=280, max_items=12),
        applicable_codes=_clean_string_list(info.applicable_codes, max_len=140, max_items=8),
        limits_of_work=_clean_string_list(info.limits_of_work, max_len=160, max_items=8),
        alternates=_clean_string_list(info.alternates, max_len=160, max_items=10),
        scope_sections=_clean_scope_sections(info.scope_sections),
        notes=[n for n in info.notes if n and len(n) < 200][:5],
        pages_with_text=info.pages_with_text,
        total_pages=info.total_pages,
    )
    if not cleaned.scope_sections and info.scope_sections:
        cleaned.notes.append(
            "Scope text was found but looked like a drawing index, so it was omitted. "
            "Check the original PDF scope sheets."
        )
    return cleaned


def generate_requirement_pdf(
    info: RequirementInfo,
    source_filename: str,
    output_path: Path,
) -> Path:
    """Write a clean requirement PDF and return ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = sanitize_requirement(info)

    styles = _styles()
    story: list = []

    # ----- Header -----
    project_name = info.project_title or Path(source_filename).stem
    story.append(Paragraph("REQUIREMENT SUMMARY", styles["eyebrow"]))
    story.append(Paragraph(_escape(project_name), styles["title"]))
    story.append(
        Paragraph(
            f"Source drawing: {_escape(source_filename)} &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"Generated: {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0f766e"), spaceAfter=12)
    )

    section_no = 1

    # ----- 1. Who gave the requirement -----
    story.append(Paragraph(f"{section_no}. Who Provided This Requirement", styles["h1"]))
    section_no += 1
    story.append(
        _info_card(
            [
                ("Company / Firm", info.provider_company),
                ("Phone", info.provider_phone),
                ("Fax", info.provider_fax),
                ("Office Address", info.provider_address),
            ],
            styles,
        )
    )

    # ----- 2. Project -----
    site_display = info.site_address
    if len(info.site_addresses) > 1:
        site_display = "; ".join(info.site_addresses)

    story.append(Paragraph(f"{section_no}. Project Details", styles["h1"]))
    section_no += 1
    story.append(
        _info_card(
            [
                ("Client / Owner", info.client),
                ("District Office Address", info.district_address),
                ("Project Title", info.project_title),
                ("Work Type", info.work_type),
                ("Site Address", site_display),
                ("Issue / Drawing Date", info.date),
                ("Document Status", info.document_status),
            ],
            styles,
        )
    )
    if info.limits_of_work:
        story.append(Paragraph("Limits of work / increments", styles["h2"]))
        story.append(_bullet_list(info.limits_of_work, styles))

    # ----- Consultants -----
    if info.consultants:
        story.append(Paragraph(f"{section_no}. Consultants", styles["h1"]))
        section_no += 1
        story.append(_consultants_table(info.consultants, styles))

    # ----- Drawing index -----
    if info.sheet_index:
        story.append(Paragraph(f"{section_no}. Drawing / Sheet Index", styles["h1"]))
        section_no += 1
        story.append(
            Paragraph(
                f"{len(info.sheet_index)} sheet(s) listed (capped for readability).",
                styles["meta"],
            )
        )
        story.append(_bullet_list(info.sheet_index, styles, sentence_case=False))

    # ----- Project / HVAC summary -----
    if info.project_summary:
        story.append(Paragraph(f"{section_no}. Project / HVAC Summary", styles["h1"]))
        section_no += 1
        story.append(_bullet_list(info.project_summary, styles))

    # ----- Codes -----
    if info.applicable_codes:
        story.append(Paragraph(f"{section_no}. Applicable Codes", styles["h1"]))
        section_no += 1
        story.append(_bullet_list(info.applicable_codes, styles, sentence_case=False))

    # ----- Scope -----
    story.append(Paragraph(f"{section_no}. Scope of Work / Requirements", styles["h1"]))
    section_no += 1
    if not info.scope_sections:
        story.append(
            Paragraph(
                "No clear numbered scope of work was found in this drawing set. "
                "Open the original PDF and check sheets titled Scope of Work / General Notes.",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"{len(info.scope_sections)} section(s) extracted from the drawing set.",
                styles["meta"],
            )
        )
        for section_index, section in enumerate(info.scope_sections, start=1):
            block: list = []
            block.append(
                Paragraph(
                    f"{section_index}. {_escape(_pretty_title(section.title))} "
                    f"<font color='#64748b' size='8'>(source page {section.page_number})</font>",
                    styles["h2"],
                )
            )
            if section.notes:
                block.append(Paragraph(_escape(section.notes), styles["meta"]))

            items = []
            for item in section.items:
                items.append(
                    ListItem(
                        Paragraph(_escape(item), styles["listBody"]),
                        leftIndent=12,
                        value="•",
                    )
                )
            block.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="•",
                    leftIndent=18,
                    bulletFontSize=9,
                    spaceBefore=2,
                    spaceAfter=8,
                )
            )
            story.append(KeepTogether(block))

    # ----- Alternates -----
    if info.alternates:
        story.append(Paragraph(f"{section_no}. Bid Alternates", styles["h1"]))
        section_no += 1
        story.append(_bullet_list(info.alternates, styles, sentence_case=False))

    if info.notes:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Notes", styles["h2"]))
        for note in info.notes:
            story.append(Paragraph(f"• {_escape(note)}", styles["meta"]))

    story.append(Spacer(1, 0.25 * inch))
    story.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=4)
    )
    story.append(
        Paragraph(
            f"Document pages scanned: {info.pages_with_text} of {info.total_pages} "
            f"contained extractable text.",
            styles["footer"],
        )
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Requirement — {project_name}",
        author="i-TAB Mechanical Device Cost Estimator",
    )
    document.build(story)
    return output_path


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "eyebrow",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#0f766e"),
            alignment=TA_CENTER,
            spaceAfter=4,
            tracking=1,
        ),
        "title": ParagraphStyle(
            "title",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.HexColor("#0f766e"),
            spaceBefore=14,
            spaceAfter=8,
            borderPadding=3,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=8,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        ),
        "listBody": ParagraphStyle(
            "listBody",
            parent=base["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=3,
        ),
        "label": ParagraphStyle(
            "label",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.HexColor("#475569"),
            leading=11,
        ),
        "value": ParagraphStyle(
            "value",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#0f172a"),
            leading=12,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#94a3b8"),
            alignment=TA_CENTER,
            spaceBefore=6,
        ),
    }


def _info_card(
    rows: list[tuple[str, str | None]], styles: dict[str, ParagraphStyle]
) -> Table:
    data = []
    for label, value in rows:
        display = value.strip() if value and value.strip() else "Not found in drawing"
        color = "#0f172a" if value and value.strip() else "#94a3b8"
        italic = "" if value and value.strip() else "<i>"
        italic_end = "" if value and value.strip() else "</i>"
        data.append(
            [
                Paragraph(_escape(label), styles["label"]),
                Paragraph(
                    f"<font color='{color}'>{italic}{_escape(display)}{italic_end}</font>",
                    styles["value"],
                ),
            ]
        )
    table = Table(data, colWidths=[1.7 * inch, 5.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _consultants_table(
    consultants: list[Consultant], styles: dict[str, ParagraphStyle]
) -> Table:
    header = [
        Paragraph("<b>Discipline</b>", styles["label"]),
        Paragraph("<b>Firm</b>", styles["label"]),
        Paragraph("<b>Phone</b>", styles["label"]),
        Paragraph("<b>Address</b>", styles["label"]),
    ]
    rows = [header]
    for consultant in consultants:
        rows.append(
            [
                Paragraph(_escape(consultant.discipline), styles["value"]),
                Paragraph(_escape(consultant.firm), styles["value"]),
                Paragraph(_escape(consultant.phone or "—"), styles["value"]),
                Paragraph(_escape(consultant.address or "—"), styles["value"]),
            ]
        )
    table = Table(rows, colWidths=[1.4 * inch, 1.9 * inch, 1.3 * inch, 2.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ccfbf1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#99f6e4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _bullet_list(
    items: list[str],
    styles: dict[str, ParagraphStyle],
    *,
    sentence_case: bool = True,
) -> ListFlowable:
    flow_items = []
    for item in items:
        text = _sentence_case(item) if sentence_case else item
        flow_items.append(
            ListItem(
                Paragraph(_escape(text), styles["listBody"]),
                leftIndent=12,
                value="•",
            )
        )
    return ListFlowable(
        flow_items,
        bulletType="bullet",
        start="•",
        leftIndent=18,
        bulletFontSize=9,
        spaceBefore=2,
        spaceAfter=8,
    )


# ---------------------------------------------------------------------------
# Content cleanup
# ---------------------------------------------------------------------------


def _clean_field(
    value: str | None,
    *,
    max_len: int = 100,
    personish: bool = False,
    require_not_disclaimer: bool = False,
) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip(" \t\r\n:.-")
    if not text or BAD_META_RE.match(text):
        return None
    if require_not_disclaimer and DISCLAIMER_RE.search(text):
        return None
    if personish and (
        len(text.split()) > 4
        or re.search(
            r"\b(VIEW|SHEET|TITLE|PLAN|INDEX|ITEMS|LINE|BREAK|SUBMITTALS|"
            r"CEILING|GRID|COLUMN|PRODUCT)\b",
            text,
            re.I,
        )
    ):
        return None
    # Spaced CAD labels like "L A N D S C A P E"
    if re.fullmatch(r"(?:[A-Za-z]\s+){3,}[A-Za-z]", text):
        return None
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def _clean_job_number(value: str | None) -> str | None:
    text = _clean_field(value, max_len=40)
    if not text:
        return None
    if BAD_META_RE.match(text) or re.search(r"sheet|title|date", text, re.I):
        return None
    return text


def _clean_dsa_number(value: str | None) -> str | None:
    text = _clean_field(value, max_len=20)
    if not text:
        return None
    if not re.fullmatch(r"0\d-\d{5,7}", text):
        return None
    return text


def _clean_string_list(
    values: list[str],
    *,
    max_len: int,
    max_items: int,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_field(value, max_len=max_len)
        if not text:
            continue
        if DISCLAIMER_RE.search(text) and len(text) > 120:
            continue
        key = text.upper()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _dedupe_consultants(consultants: list[Consultant]) -> list[Consultant]:
    seen: set[tuple[str, str]] = set()
    result: list[Consultant] = []
    for consultant in consultants:
        firm = _clean_field(consultant.firm, max_len=80)
        discipline = _clean_field(consultant.discipline, max_len=60)
        if not firm or not discipline:
            continue
        # Skip spaced-letter fake firms
        if re.fullmatch(r"(?:[A-Za-z]\s+){3,}[A-Za-z]", firm):
            continue
        key = (discipline.upper(), firm.upper())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            Consultant(
                discipline=discipline,
                firm=firm,
                phone=_clean_field(consultant.phone, max_len=40),
                address=_clean_field(consultant.address, max_len=100),
            )
        )
    return result[:12]


def _looks_like_sheet_index(text: str) -> bool:
    """True when an 'item' is mostly a drawing sheet index, not a work item."""
    codes = SHEET_CODE_RE.findall(text)
    if len(codes) >= 3:
        return True
    if re.search(r"SHEETS?\s+TOTAL\s*:", text, re.I):
        return True
    if re.search(r"\b(DEMO\s+&?\s*NEW\s+FLOOR\s+PLANS|TITLE\s+24)\b", text, re.I) and len(codes) >= 2:
        return True
    return False


def _normalize_scope_item_key(item: str) -> str:
    return re.sub(r"\s+", " ", item).strip().upper()[:96]


def _scope_items_overlap(a: list[str], b: list[str], *, threshold: float = 0.7) -> bool:
    if not a or not b:
        return False
    set_a = {_normalize_scope_item_key(x) for x in a}
    set_b = {_normalize_scope_item_key(x) for x in b}
    overlap = len(set_a & set_b)
    return overlap >= threshold * min(len(set_a), len(set_b))


def _is_generic_scope_title(title: str) -> bool:
    """Bare 'Scope of Work' without a trade/restoration-specific prefix."""
    cleaned = title.strip().rstrip(":").strip()
    return bool(
        re.match(
            r"^(?:SHT\s+\d+\s*[-–]?\s*)?SCOPE(?:\s+OF\s+WORK)?(?:\s*[:\-].*)?$",
            cleaned,
            re.I,
        )
    )


def _drop_redundant_generic_scopes(sections: list[ScopeSection]) -> list[ScopeSection]:
    """Never print a hardcoded/empty Scope of Work that only copies another section."""
    kept: list[ScopeSection] = []
    for section in sections:
        overlap_index = next(
            (
                i
                for i, other in enumerate(kept)
                if _scope_items_overlap(section.items, other.items)
            ),
            None,
        )
        if overlap_index is None:
            kept.append(section)
            continue

        existing = kept[overlap_index]
        if _is_generic_scope_title(section.title) and not _is_generic_scope_title(
            existing.title
        ):
            continue
        if _is_generic_scope_title(existing.title) and not _is_generic_scope_title(
            section.title
        ):
            kept[overlap_index] = section
            continue
        if len(section.items) > len(existing.items):
            kept[overlap_index] = section
    return kept


def _clean_scope_sections(sections: list[ScopeSection]) -> list[ScopeSection]:
    cleaned: list[ScopeSection] = []
    for section in sections:
        title = _pretty_title(section.title)
        # Prefer real scope headings; demote bare GENERAL NOTES when they are indexes.
        items: list[str] = []
        for raw in section.items:
            item = re.sub(r"\s+", " ", raw).strip()
            if len(item) < 20:
                continue
            if _looks_like_sheet_index(item):
                continue
            if DISCLAIMER_RE.search(item) and len(item) > 160:
                continue
            # Drop title-block chrome that leaked into scope bullets.
            if re.search(
                r"\b(PHONE|FAX)\s*\(?\d{3}\)?|\bTHE\s+GARLAND\s+COMPANY\b|"
                r"\bSCHOOL\s+DISTRICT\b|^\s*AGENT\b",
                item,
                re.I,
            ) and not re.search(
                r"\b(REMOVE|INSTALL|NAIL|FASTEN|PERFORM|APPLY|CLEAN|TEST|REPLACE)\b",
                item,
                re.I,
            ):
                continue
            # Drop keynote / dimension noise.
            if re.search(r"\bTYP\.?\s*\d|V\.?I\.?F\.?|[�\"]?\d+\'-\d+", item, re.I):
                alpha = sum(1 for ch in item if ch.isalpha())
                if alpha < 40:
                    continue
            if re.search(
                r"\b(SALVAGED|FLAGPOLE|ASPHALT CONCRETE PAVING)\b", item, re.I
            ) and re.search(r"\b\d{2}\.[A-Z]\b", item):
                # Keynote callouts masquerading as scope.
                continue
            # Soft wrap length for readability
            if len(item) > 500:
                item = item[:497].rstrip() + "…"
            items.append(item)

        if not items:
            continue
        # Require at least one substantive work sentence for vague "Scope" titles.
        if title.upper() in {
            "SCOPE",
            "THE ENTIRE SCOPE OF WORK",
            "FOR THE ENTIRE SCOPE OF WORK",
            "NEW SCOPE OF WORK",
        }:
            # Keep only if items look like sentences, not keynote callouts (13.D …).
            proseish = [
                i
                for i in items
                if len(i) > 45 and not re.match(r"^\d{1,3}\.[A-Z]\b", i)
            ]
            if len(proseish) < 2:
                continue
            items = proseish
        # Cap very long sections
        if len(items) > 40:
            items = items[:40]
            note = f"{section.notes or ''} (showing first 40 items)".strip()
        else:
            note = (section.notes or "").strip() or None

        cleaned.append(
            ScopeSection(
                title=title,
                items=items,
                page_number=section.page_number,
                notes=note,
            )
        )

    cleaned = _drop_redundant_generic_scopes(cleaned)

    # Prefer named trade SCOPE OF WORK sections; keep GENERAL NOTES only as fallback.
    scope_like = [
        s
        for s in cleaned
        if "SCOPE" in s.title.upper() or "SUMMARY" in s.title.upper()
    ]
    # Rank: HVAC / mechanical / roof / restoration first, then other titled trades.
    # Bare "Scope of Work" ranks below specific * Scope titles.
    def rank(section: ScopeSection) -> tuple[int, int]:
        upper = section.title.upper()
        score = 0
        if _is_generic_scope_title(section.title):
            score -= 2
        elif "SCOPE OF WORK" in upper or upper.endswith(" SCOPE"):
            score += 5
        if re.search(r"\b(MECHANICAL|HVAC|ROOF|RESTORATION|LIQUID\s+APPLIED)\b", upper):
            score += 8
        if re.search(
            r"REPAINT|PAINT|TECHNOLOGY|ELECTRICAL|PLUMBING|DEMOLITION|FIRE\s+ALARM|INTRUSION",
            upper,
        ):
            score += 4
        if upper in {
            "SCOPE",
            "THE ENTIRE SCOPE OF WORK",
            "FOR THE ENTIRE SCOPE OF WORK",
            "NEW SCOPE OF WORK",
        }:
            score -= 3
        return (-score, -len(section.items))

    if scope_like:
        scope_like.sort(key=rank)
        return scope_like[:8]
    return cleaned[:6]


def _pretty_title(title: str) -> str:
    text = re.sub(r"\s+", " ", title).strip(" :.-")
    text = re.sub(r"^SHT\s+\d+\s*[-–]?\s*", "", text, flags=re.I)
    if text.isupper() and len(text) > 4:
        return text.title()
    return text


def _sentence_case(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text.isupper() and len(text) > 20:
        lowered = text.lower()
        return lowered[0].upper() + lowered[1:]
    return text


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
