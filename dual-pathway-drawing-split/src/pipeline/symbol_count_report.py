"""Export wing symbol count pivot tables as CSV and PDF."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pipeline.extraction.symbol_table_extractor import SymbolTableInfo, catalog_id_for_entry


def _escape(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _symbol_key(entry) -> str:
    return catalog_id_for_entry(entry)


def build_pivot_rows(
    legend: SymbolTableInfo,
    instance_results: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return column keys and row dicts for the pivot table."""
    columns = [str(item["column_key"]) for item in instance_results]
    rows: list[dict[str, Any]] = []
    for entry in legend.entries:
        key = _symbol_key(entry)
        wing_counts = {}
        total = 0
        for column in columns:
            count = 0
            for result in instance_results:
                if result.get("column_key") != column:
                    continue
                count = int((result.get("counts") or {}).get(key, 0))
                break
            wing_counts[column] = count
            total += count
        rows.append(
            {
                "symbol": entry.symbol or "",
                "description": entry.description,
                "mfg_model": entry.mfg_model or "",
                "part_number": entry.part_number or "",
                "wing_counts": wing_counts,
                "total": total,
            }
        )
    return columns, rows


def build_symbol_count_report_payload(
    legend: SymbolTableInfo,
    instance_results: list[dict[str, Any]],
    *,
    status: str = "done",
) -> dict[str, Any]:
    """Build JSON-serializable pivot report for the UI."""
    columns, rows = build_pivot_rows(legend, instance_results)
    return {
        "status": status,
        "columns": columns,
        "rows": [
            {
                "symbol": row["symbol"],
                "description": row["description"],
                "mfg_model": row["mfg_model"],
                "part_number": row["part_number"],
                "counts": row["wing_counts"],
                "total": row["total"],
            }
            for row in rows
        ],
    }


def write_symbol_count_json(
    *,
    legend: SymbolTableInfo,
    instance_results: list[dict[str, Any]],
    output_path: Path,
    status: str = "done",
) -> Path:
    payload = build_symbol_count_report_payload(
        legend,
        instance_results,
        status=status,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_symbol_count_csv(
    *,
    legend: SymbolTableInfo,
    instance_results: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    columns, rows = build_pivot_rows(legend, instance_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Symbol", "Description", "MFG/Model", "Part Number", *columns, "Total"]
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "Symbol": row["symbol"],
                "Description": row["description"],
                "MFG/Model": row["mfg_model"],
                "Part Number": row["part_number"],
                "Total": row["total"],
            }
            for column in columns:
                record[column] = row["wing_counts"].get(column, 0)
            writer.writerow(record)
    try:
        tmp_path.replace(output_path)
        return output_path
    except OSError:
        fallback = output_path.with_name(output_path.stem + ".updated.csv")
        tmp_path.replace(fallback)
        return fallback


def write_symbol_count_pdf(
    *,
    legend: SymbolTableInfo,
    instance_results: list[dict[str, Any]],
    output_path: Path,
    source_filename: str,
    project_title: str | None = None,
) -> Path:
    columns, rows = build_pivot_rows(legend, instance_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    styles = {
        "eyebrow": ParagraphStyle(
            "eyebrow",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "title",
            parent=base["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#4b5563"),
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontSize=7,
            leading=9,
            alignment=TA_LEFT,
        ),
        "cell_center": ParagraphStyle(
            "cell_center",
            parent=base["Normal"],
            fontSize=7,
            leading=9,
            alignment=TA_CENTER,
        ),
    }

    story: list = []
    title = project_title or Path(source_filename).stem
    story.append(Paragraph("SYMBOL COUNT BY WING", styles["eyebrow"]))
    story.append(Paragraph(_escape(title), styles["title"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')} "
            f"&nbsp;&nbsp;·&nbsp;&nbsp; {len(instance_results)} wing instance(s)",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=colors.HexColor("#0f766e"),
            spaceAfter=10,
        )
    )

    header = [
        Paragraph("Symbol", styles["cell"]),
        Paragraph("Description", styles["cell"]),
        Paragraph("MFG/Model", styles["cell"]),
        Paragraph("Part", styles["cell"]),
        *[Paragraph(_escape(col), styles["cell"]) for col in columns],
        Paragraph("Total", styles["cell_center"]),
    ]
    table_data = [header]
    for row in rows:
        table_data.append(
            [
                Paragraph(_escape(row["symbol"]), styles["cell_center"]),
                Paragraph(_escape(row["description"]), styles["cell"]),
                Paragraph(_escape(row["mfg_model"]), styles["cell"]),
                Paragraph(_escape(row["part_number"]), styles["cell"]),
                *[
                    Paragraph(str(row["wing_counts"].get(col, 0)), styles["cell_center"])
                    for col in columns
                ],
                Paragraph(str(row["total"]), styles["cell_center"]),
            ]
        )

    page_width, _ = landscape(letter)
    usable = page_width - 1.0 * inch
    fixed = [0.45 * inch, 1.6 * inch, 1.0 * inch, 0.7 * inch]
    remaining = max(0.5 * inch, usable - sum(fixed))
    wing_width = max(0.3 * inch, remaining / max(len(columns) + 1, 1))
    col_widths = fixed + [wing_width] * len(columns) + [max(0.4 * inch, wing_width)]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecfdf5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#065f46")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    doc.build(story)
    return output_path


def write_symbol_count_reports(
    *,
    legend: SymbolTableInfo,
    instance_results: list[dict[str, Any]],
    out_dir: Path,
    source_filename: str,
    project_title: str | None = None,
) -> dict[str, str]:
    csv_path = out_dir / "symbol_count_report.csv"
    pdf_path = out_dir / "symbol_count_report.pdf"
    json_path = out_dir / "symbol_count_report.json"
    written_csv = write_symbol_count_csv(
        legend=legend,
        instance_results=instance_results,
        output_path=csv_path,
    )
    write_symbol_count_pdf(
        legend=legend,
        instance_results=instance_results,
        output_path=pdf_path,
        source_filename=source_filename,
        project_title=project_title,
    )
    write_symbol_count_json(
        legend=legend,
        instance_results=instance_results,
        output_path=json_path,
    )
    return {
        "csv": f"{out_dir.name}/{written_csv.name}",
        "pdf": f"{out_dir.name}/symbol_count_report.pdf",
        "json": f"{out_dir.name}/symbol_count_report.json",
    }
