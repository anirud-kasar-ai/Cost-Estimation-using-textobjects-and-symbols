"""Generate a readable technical-symbol PDF from SymbolTableInfo."""

from __future__ import annotations

import io
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
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo


def technical_symbol_pdf_filename(source_filename: str) -> str:
    """``abc.pdf`` -> ``abc technical symbol.pdf``."""
    stem = Path(source_filename).stem
    return f"{stem} technical symbol.pdf"


def generate_technical_symbol_pdf(
    info: SymbolTableInfo,
    source_filename: str,
    output_path: Path,
) -> Path:
    """Write the technical symbol PDF and return ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    story: list = []

    story.append(Paragraph("TECHNICAL SYMBOL SUMMARY", styles["eyebrow"]))
    story.append(
        Paragraph(_escape(Path(source_filename).stem), styles["title"])
    )
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

    pages = ", ".join(str(p) for p in info.legend_pages) or "—"
    story.append(
        Paragraph(
            f"{info.entry_count} symbol/legend entr"
            f"{'y' if info.entry_count == 1 else 'ies'} extracted from "
            f"{info.total_pages} page(s). Legend page(s): {pages}.",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    if not info.entries:
        story.append(
            Paragraph(
                "No symbol legend or drafting symbols table was "
                "found in this drawing set.",
                styles["body"],
            )
        )
        for note in info.notes:
            story.append(Paragraph(f"• {_escape(note)}", styles["meta"]))
    else:
        # Group by legend family. Numbered SYMBOL LEGEND rows from multiple
        # pages are merged into one sorted table (matches the drawing order).
        groups: dict[str, list[SymbolEntry]] = {}
        group_pages: dict[str, list[int]] = {}
        for entry in info.entries:
            title = entry.legend_title or "Symbol Legend"
            if _is_plain_numbered_symbol_legend(title):
                key = "SYMBOL LEGEND"
            elif _is_technology_legend(title):
                key = f"TECH|{title}|{entry.source_page}"
            else:
                key = f"OTHER|{title}|{entry.source_page}"
            groups.setdefault(key, []).append(entry)
            group_pages.setdefault(key, [])
            if entry.source_page not in group_pages[key]:
                group_pages[key].append(entry.source_page)

        section = 1
        for key, rows in groups.items():
            title = rows[0].legend_title or "Symbol Legend"
            pages = group_pages.get(key, [])
            page_label = ", ".join(str(p) for p in sorted(pages)) or "—"
            if _is_plain_numbered_symbol_legend(title):
                rows = sorted(
                    rows,
                    key=lambda e: int(e.symbol)
                    if e.symbol and str(e.symbol).isdigit()
                    else 10**9,
                )
                title = "SYMBOL LEGEND"
            story.append(
                Paragraph(
                    f"{section}. {_escape(_pretty_title(title))} "
                    f"<font color='#64748b' size='8'>(source page {page_label})</font>",
                    styles["h1"],
                )
            )
            section += 1
            if _is_technology_legend(title):
                story.append(_technology_entries_table(rows, styles))
            else:
                story.append(_entries_table(rows, styles))
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
        title=f"Technical Symbols — {Path(source_filename).stem}",
    )
    doc.build(story)
    return output_path


def _is_plain_numbered_symbol_legend(title: str) -> bool:
    upper = (title or "").upper()
    return "SYMBOL LEGEND" in upper and "TECHNOLOGY" not in upper


def _is_technology_legend(title: str) -> bool:
    return "TECHNOLOGY" in title.upper() and "SYMBOL" in title.upper()


def _format_entry_notes(notes: list[str] | None) -> str:
    """Join note tokens for PDF display."""
    if not notes:
        return "—"
    return " · ".join(notes)


def _technology_entries_table(entries: list[SymbolEntry], styles: dict) -> Table:
    header = [
        Paragraph("<b>Symbol</b>", styles["th"]),
        Paragraph("<b>Description</b>", styles["th"]),
        Paragraph("<b>MFG/Model</b>", styles["th"]),
        Paragraph("<b>Part</b>", styles["th"]),
        Paragraph("<b>Notes</b>", styles["th"]),
    ]
    data: list[list] = [header]
    for entry in entries:
        data.append(
            [
                _symbol_cell(entry, styles),
                Paragraph(_escape(entry.description), styles["td"]),
                Paragraph(_escape(entry.mfg_model or "—"), styles["td"]),
                Paragraph(_escape(entry.part_number or "—"), styles["td"]),
                Paragraph(_escape(_format_entry_notes(entry.notes)), styles["td"]),
            ]
        )
    table = Table(
        data,
        colWidths=[
            1.0 * inch,
            2.25 * inch,
            1.1 * inch,
            1.1 * inch,
            1.55 * inch,
        ],
        repeatRows=1,
    )
    table.setStyle(_table_style())
    return table


def _symbol_cell(entry: SymbolEntry, styles: dict):
    label = entry.symbol or "—"
    # Numbered legend items use ``N.`` on the sheet (glyph left of the index).
    display = f"{label}." if label.isdigit() else label
    label_para = Paragraph(_escape(display), styles["tdCenter"])
    if not entry.symbol_image_png:
        return label_para
    try:
        image = Image(io.BytesIO(entry.symbol_image_png))
        # Square-ish legend glyphs — keep them readable, not as tall slivers.
        max_w, max_h = 0.78 * inch, 0.78 * inch
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
        # Numbered legends: glyph left of index, same layout as the drawing.
        if label.isdigit():
            wrap = Table(
                [[image, label_para]],
                colWidths=[0.68 * inch, 0.36 * inch],
            )
            wrap.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (0, 0), "RIGHT"),
                        ("ALIGN", (1, 0), (1, 0), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 1),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                        ("TOPPADDING", (0, 0), (-1, -1), 1),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ]
                )
            )
            return wrap
        return image
    except Exception:  # noqa: BLE001
        return label_para


def _entries_table(entries: list[SymbolEntry], styles: dict) -> Table:
    header = [
        Paragraph("<b>Symbol</b>", styles["th"]),
        Paragraph("<b>Description</b>", styles["th"]),
        Paragraph("<b>Notes</b>", styles["th"]),
    ]
    data = [header]
    for entry in entries:
        data.append(
            [
                _symbol_cell(entry, styles),
                Paragraph(_escape(entry.description), styles["td"]),
                Paragraph(_escape(_format_entry_notes(entry.notes)), styles["td"]),
            ]
        )
    table = Table(
        data,
        colWidths=[1.45 * inch, 3.75 * inch, 1.9 * inch],
        repeatRows=1,
    )
    table.setStyle(_table_style())
    return table


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.HexColor("#f8fafc"), colors.white],
            ),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ]
    )


def _pretty_title(title: str) -> str:
    text = re.sub(r"\s+", " ", title).strip(" :.-")
    if text.isupper() and len(text) > 4:
        return text.title()
    return text


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "symEyebrow",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=2,
            tracking=1,
        ),
        "title": ParagraphStyle(
            "symTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
            leading=20,
        ),
        "h1": ParagraphStyle(
            "symH1",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=6,
            spaceAfter=6,
            leading=14,
        ),
        "meta": ParagraphStyle(
            "symMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#64748b"),
            leading=11,
        ),
        "body": ParagraphStyle(
            "symBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#334155"),
            leading=12,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "symTh",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=colors.white,
            leading=9,
        ),
        "td": ParagraphStyle(
            "symTd",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#0f172a"),
            leading=9,
        ),
        "tdCenter": ParagraphStyle(
            "symTdCenter",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.HexColor("#0f172a"),
            leading=10,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "symFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#94a3b8"),
            spaceBefore=4,
        ),
    }
