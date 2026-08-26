"""Generate a readable sheet-notes PDF from SheetNotesInfo."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pipeline.extraction.sheet_notes_extractor import SheetNotesInfo


def sheet_notes_pdf_filename(source_filename: str) -> str:
    """``abc.pdf`` -> ``abc sheet notes.pdf``."""
    stem = Path(source_filename).stem
    return f"{stem} sheet notes.pdf"


def generate_sheet_notes_pdf(
    info: SheetNotesInfo,
    source_filename: str,
    output_path: Path,
) -> Path:
    """Write the sheet-notes PDF and return ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    story: list = []

    story.append(Paragraph("SHEET NOTES SUMMARY", styles["eyebrow"]))
    story.append(Paragraph(_escape(Path(source_filename).stem), styles["title"]))
    story.append(
        Paragraph(
            f"Source drawing: {_escape(source_filename)} &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"Generated: {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=colors.HexColor("#0f766e"),
            spaceAfter=12,
        )
    )

    pages = ", ".join(str(p) for p in info.note_pages) or "—"
    story.append(
        Paragraph(
            f"{info.item_count} note item{'s' if info.item_count != 1 else ''} "
            f"(general {info.count_of('general')}, sheet {info.count_of('sheet')}, "
            f"tags {info.count_of('sheet_tag')}) from {info.total_pages} page(s). "
            f"Note page(s): {pages}.",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    if not info.items:
        story.append(
            Paragraph(
                "No general notes, sheet notes, or demo tag notes were found "
                "in this drawing set.",
                styles["body"],
            )
        )
        for note in info.notes:
            story.append(Paragraph(f"• {_escape(note)}", styles["meta"]))
    else:
        groups = info.grouped_by_diagram()
        section = 1
        for slug, rows in groups:
            sample = rows[0]
            title = (
                sample.display_name
                or sample.sheet_number
                or f"Page {sample.source_page}"
            )
            page_label = sample.related_diagram_page or sample.source_page
            folder = f"05_metadata/page_{page_label:03d}.json"
            story.append(
                Paragraph(
                    f"{section}. {_escape(title)} "
                    f"<font color='#64748b' size='8'>(PDF page {page_label} · {folder})</font>",
                    styles["h1"],
                )
            )
            section += 1
            general = [r for r in rows if r.note_type == "general"]
            sheet = [r for r in rows if r.note_type == "sheet"]
            tags = [r for r in rows if r.note_type == "sheet_tag"]
            if general:
                story.append(Paragraph("General Notes", styles["h2"]))
                story.append(_numbered_table(general, styles, glyph=False))
                story.append(Spacer(1, 0.10 * inch))
            if sheet:
                story.append(Paragraph("Sheet Notes", styles["h2"]))
                story.append(_numbered_table(sheet, styles, glyph=True))
                story.append(Spacer(1, 0.10 * inch))
            if tags:
                story.append(Paragraph("Sheet Tag Notes", styles["h2"]))
                story.append(_tag_table(tags, styles))
                story.append(Spacer(1, 0.14 * inch))

    story.append(Spacer(1, 0.2 * inch))
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#cbd5e1"),
            spaceBefore=4,
        )
    )
    story.append(
        Paragraph(
            f"Document pages scanned: {info.total_pages} "
            f"(full PDF; not limited by CV render cap).",
            styles["footer"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"Sheet Notes — {Path(source_filename).stem}",
    )
    doc.build(story)
    return output_path


def _numbered_table(items: list[NoteItem], styles: dict, *, glyph: bool) -> Table:
    header = [
        Paragraph("<b>Symbol</b>", styles["th"]),
        Paragraph("<b>No.</b>", styles["th"]),
        Paragraph("<b>Description</b>", styles["th"]),
    ]
    data: list[list] = [header]
    for item in items:
        data.append(
            [
                _symbol_cell(item, styles) if glyph else Paragraph("—", styles["tdCenter"]),
                Paragraph(_escape(item.symbol or "—"), styles["tdCenter"]),
                Paragraph(_escape(item.text), styles["td"]),
            ]
        )
    table = Table(data, colWidths=[0.85 * inch, 0.45 * inch, 5.75 * inch], repeatRows=1)
    table.setStyle(_table_style())
    return table


def _tag_table(items: list[NoteItem], styles: dict) -> Table:
    header = [
        Paragraph("<b>Symbol</b>", styles["th"]),
        Paragraph("<b>Tag ID</b>", styles["th"]),
        Paragraph("<b>Location</b>", styles["th"]),
        Paragraph("<b>Description</b>", styles["th"]),
    ]
    data: list[list] = [header]
    for item in items:
        data.append(
            [
                _symbol_cell(item, styles),
                Paragraph(_escape(item.tag_id or item.symbol or "—"), styles["td"]),
                Paragraph(_escape(item.location or "—"), styles["td"]),
                Paragraph(_escape(item.text or "—"), styles["td"]),
            ]
        )
    table = Table(
        data,
        colWidths=[0.85 * inch, 1.15 * inch, 1.1 * inch, 3.95 * inch],
        repeatRows=1,
    )
    table.setStyle(_table_style())
    return table


def _symbol_cell(item: NoteItem, styles: dict):
    label = item.symbol or item.tag_id or "—"
    label_para = Paragraph(_escape(label), styles["tdCenter"])
    if not item.symbol_image_png:
        return label_para
    try:
        image = Image(io.BytesIO(item.symbol_image_png))
        max_w, max_h = 0.55 * inch, 0.55 * inch
        width = float(getattr(image, "imageWidth", 0) or 0)
        height = float(getattr(image, "imageHeight", 0) or 0)
        if width <= 0 or height <= 0:
            return label_para
        aspect = width / height
        if max_w / max_h > aspect:
            image.drawHeight = max_h
            image.drawWidth = max_h * aspect
        else:
            image.drawWidth = max_w
            image.drawHeight = max_w / aspect
        image.hAlign = "CENTER"
        return image
    except Exception:  # noqa: BLE001
        return label_para


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ]
    )


def _escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "notesEyebrow",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=2,
            tracking=1,
        ),
        "title": ParagraphStyle(
            "notesTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
            leading=20,
        ),
        "h1": ParagraphStyle(
            "notesH1",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=6,
            spaceAfter=6,
            leading=14,
        ),
        "h2": ParagraphStyle(
            "notesH2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#0f766e"),
            spaceBefore=4,
            spaceAfter=4,
            leading=12,
        ),
        "meta": ParagraphStyle(
            "notesMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#64748b"),
            leading=11,
        ),
        "body": ParagraphStyle(
            "notesBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#334155"),
            leading=12,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "notesTh",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=colors.white,
            leading=9,
        ),
        "td": ParagraphStyle(
            "notesTd",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#0f172a"),
            leading=9,
        ),
        "tdCenter": ParagraphStyle(
            "notesTdCenter",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.HexColor("#0f172a"),
            leading=10,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "notesFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#94a3b8"),
            spaceBefore=4,
        ),
    }
