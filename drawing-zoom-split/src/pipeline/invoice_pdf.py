"""Generate a printable cost-invoice PDF from 06_counts/invoice.json."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline import config

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
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

COUNTS_DIR = "06_counts"
INVOICE_PDF_NAME = "invoice.pdf"

APP_NAME = "Cost Estimate Using Site Construction Drawing"
# drawing-zoom-split/assets/aziro-logo.png (this file lives in src/pipeline/)
_LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "aziro-logo.png"

_ACCENT = colors.HexColor("#1d4ed8")
_INK = colors.HexColor("#0f172a")
_MUTED = colors.HexColor("#64748b")
_LINE = colors.HexColor("#cbd5e1")
_HEAD_BG = colors.HexColor("#dbeafe")
_ROW_BG = colors.HexColor("#f8fafc")


def invoice_pdf_path(job_root: Path) -> Path:
    return job_root / COUNTS_DIR / INVOICE_PDF_NAME


def _money(value: Any) -> str:
    return f"${float(value or 0):,.2f}"


def _escape(text: str) -> str:
    return (
        str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "eyebrow", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9, textColor=_ACCENT, alignment=TA_CENTER, spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "title", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=18, leading=22, textColor=_INK, alignment=TA_CENTER, spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"], fontSize=8, leading=11,
            textColor=_MUTED, alignment=TA_CENTER, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, textColor=_ACCENT, spaceBefore=14, spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontSize=9, leading=12, textColor=_INK,
        ),
        "cellMuted": ParagraphStyle(
            "cellMuted", parent=base["Normal"], fontSize=8, leading=11,
            textColor=_MUTED,
        ),
        "cellRight": ParagraphStyle(
            "cellRight", parent=base["Normal"], fontSize=9, leading=12,
            textColor=_INK, alignment=TA_RIGHT,
        ),
        "head": ParagraphStyle(
            "head", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, textColor=_INK,
        ),
        "totalLabel": ParagraphStyle(
            "totalLabel", parent=base["Normal"], fontSize=10, alignment=TA_RIGHT,
            textColor=_MUTED,
        ),
        "totalValue": ParagraphStyle(
            "totalValue", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=10, alignment=TA_RIGHT, textColor=_INK,
        ),
        "grand": ParagraphStyle(
            "grand", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=13, alignment=TA_RIGHT, textColor=_ACCENT,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"], fontSize=7.5, textColor=_MUTED,
            alignment=TA_CENTER, spaceBefore=6,
        ),
    }


def _table_style(header_rows: int = 1) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), _HEAD_BG),
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, _ROW_BG]),
            ("BOX", (0, 0), (-1, -1), 0.8, _LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def generate_invoice_pdf(job_root: Path, job: dict[str, Any] | None = None) -> Path:
    """Render 06_counts/invoice.json (+ result.json) into invoice.pdf."""
    counts_dir = job_root / COUNTS_DIR
    invoice = json.loads((counts_dir / "invoice.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    result_path = counts_dir / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}

    job = job or {}
    styles = _styles()
    story: list = []

    filename = job.get("filename") or invoice.get("job_id") or job_root.name
    project = (job.get("metadata") or {}).get("title") or Path(filename).stem
    job_id = str(job.get("id") or invoice.get("job_id") or job_root.name)
    estimate_no = f"EST-{job_id.upper()[:24]}"
    issued = datetime.now(timezone.utc)
    valid_days = int(getattr(config, "ESTIMATE_VALID_DAYS", 30))
    valid_until = issued + timedelta(days=valid_days)

    if _LOGO_PATH.is_file():
        logo = Image(str(_LOGO_PATH), width=0.62 * inch, height=0.62 * inch)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 0.04 * inch))
    story.append(Paragraph(_escape(APP_NAME.upper()), styles["eyebrow"]))
    story.append(Paragraph("COST ESTIMATE", styles["eyebrow"]))
    story.append(Paragraph(_escape(project), styles["title"]))
    story.append(
        Paragraph(
            f"Estimate No: <b>{_escape(estimate_no)}</b> &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"Issued: {issued.strftime('%d %b %Y, %H:%M UTC')} &nbsp;&nbsp;·&nbsp;&nbsp; "
            f"Valid until: {valid_until.strftime('%d %b %Y')}",
            styles["meta"],
        )
    )
    summary_bits = [f"Source drawing: {filename}"]
    if job.get("page_count"):
        summary_bits.append(f"{job['page_count']} page(s)")
    total_count = result.get("total_count") or job.get("symbol_count_total")
    if total_count:
        summary_bits.append(f"{total_count} symbols counted")
    story.append(Paragraph(_escape(" · ".join(summary_bits)), styles["meta"]))
    story.append(Spacer(1, 0.06 * inch))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_ACCENT, spaceAfter=10))

    # ----- From / Bill To blocks (standard invoice parties) -----
    def _party_lines(title: str, lines: list[str]) -> Paragraph:
        body = "<br/>".join(_escape(ln) for ln in lines if ln)
        return Paragraph(
            f"<b>{_escape(title)}</b><br/>{body or '<i>Not specified</i>'}",
            styles["cell"],
        )

    company_lines = [
        getattr(config, "COMPANY_NAME", "") or "Aziro",
        APP_NAME,
        getattr(config, "COMPANY_ADDRESS", ""),
        getattr(config, "COMPANY_CONTACT", ""),
        (f"Tax ID: {config.COMPANY_TAX_ID}" if getattr(config, "COMPANY_TAX_ID", "") else ""),
    ]
    client = job.get("client") or {}
    client_lines = [
        str(client.get("name") or ""),
        str(client.get("address") or ""),
        str(client.get("contact") or ""),
    ]
    parties = Table(
        [[_party_lines("From", company_lines), _party_lines("Bill To", client_lines)]],
        colWidths=[3.5 * inch, 3.5 * inch],
    )
    parties.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(parties)

    # ----- Invoice lines -----
    story.append(Paragraph("Estimate Line Items", styles["h1"]))
    lines = invoice.get("lines") or []
    rows = [
        [
            Paragraph("#", styles["head"]),
            Paragraph("Description", styles["head"]),
            Paragraph("Detected as", styles["head"]),
            Paragraph("Qty", styles["head"]),
            Paragraph("Unit", styles["head"]),
            Paragraph("Unit price", styles["head"]),
            Paragraph("Line total", styles["head"]),
        ]
    ]
    for index, line in enumerate(lines, start=1):
        rows.append(
            [
                Paragraph(str(index), styles["cell"]),
                Paragraph(_escape(line.get("description")), styles["cell"]),
                Paragraph(
                    _escape(", ".join(line.get("yolo_classes") or [])),
                    styles["cellMuted"],
                ),
                Paragraph(str(line.get("quantity") or 0), styles["cellRight"]),
                Paragraph(_escape(line.get("unit") or "EA"), styles["cell"]),
                Paragraph(_money(line.get("unit_price_usd")), styles["cellRight"]),
                Paragraph(_money(line.get("line_total_usd")), styles["cellRight"]),
            ]
        )
    table = Table(
        rows,
        colWidths=[
            0.35 * inch,
            2.15 * inch,
            1.55 * inch,
            0.5 * inch,
            0.5 * inch,
            0.95 * inch,
            1.0 * inch,
        ],
        repeatRows=1,
    )
    table.setStyle(_table_style())
    story.append(table)

    # ----- Totals -----
    tax_rate = float(invoice.get("tax_rate") or 0)
    totals = Table(
        [
            [
                Paragraph("Subtotal", styles["totalLabel"]),
                Paragraph(_money(invoice.get("subtotal_usd")), styles["totalValue"]),
            ],
            [
                Paragraph(f"Tax ({tax_rate * 100:.1f}%)", styles["totalLabel"]),
                Paragraph(_money(invoice.get("tax_usd")), styles["totalValue"]),
            ],
            [
                Paragraph("GRAND TOTAL", styles["totalLabel"]),
                Paragraph(_money(invoice.get("grand_total_usd")), styles["grand"]),
            ],
        ],
        colWidths=[5.5 * inch, 1.5 * inch],
    )
    totals.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 2), (-1, 2), 0.8, _ACCENT),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(totals)

    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_LINE, spaceBefore=4))
    story.append(
        Paragraph(
            f"All amounts in USD · Estimate valid for {valid_days} days from issue date · "
            "This is a computer-generated cost estimate, not a demand for payment.",
            styles["footer"],
        )
    )
    story.append(
        Paragraph(
            f"{_escape(getattr(config, 'COMPANY_NAME', '') or 'Aziro')} — {_escape(APP_NAME)}",
            styles["footer"],
        )
    )

    output_path = invoice_pdf_path(job_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Cost Estimate {estimate_no} — {project}",
        author=APP_NAME,
    )
    document.build(story)
    return output_path
