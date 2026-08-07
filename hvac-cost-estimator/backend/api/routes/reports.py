"""Costing report endpoints: consolidated JSON + CSV export."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from config import Settings, get_settings
from db.models import Project, ProjectStatus
from db.session import get_db
from schemas.project import ProjectMetadata
from schemas.report import CostingReport, ReportLine

router = APIRouter(prefix="/api/projects", tags=["reports"])

METADATA_ROWS: tuple[tuple[str, str], ...] = (
    ("Project Title", "title"),
    ("Client", "client"),
    ("Architect", "architect"),
    ("Engineer", "engineer"),
    ("Project Address", "project_address"),
    ("Due Date", "due_date"),
)


def _get_completed_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found.",
        )
    if project.status != ProjectStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project is not ready (status: {project.status.value}).",
        )
    return project


def _build_report(project: Project, settings: Settings) -> CostingReport:
    from api.routes.projects import build_project_detail

    detail = build_project_detail(project, settings)
    return CostingReport(
        project_id=project.id,
        filename=project.filename,
        metadata=detail.metadata,
        lines=[
            ReportLine(
                device_type=line.device_type,
                display_name=line.display_name,
                count=line.count,
                unit_cost=line.unit_cost,
                line_total=line.line_total,
                needs_review=line.needs_review,
            )
            for line in detail.device_lines
        ],
        grand_total=detail.grand_total,
        currency=detail.currency,
    )


@router.get(
    "/{project_id}/report",
    response_model=CostingReport,
    summary="Consolidated costing report (metadata + line items + grand total)",
)
def get_report(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> CostingReport:
    return _build_report(_get_completed_project(db, project_id), settings)


@router.get(
    "/{project_id}/report/csv",
    summary="Export the costing report as CSV",
    response_class=StreamingResponse,
)
def export_report_csv(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    project = _get_completed_project(db, project_id)
    report = _build_report(project, settings)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    writer.writerow(["Field", "Information"])
    metadata = ProjectMetadata.model_validate(report.metadata).model_dump()
    for label, key in METADATA_ROWS:
        writer.writerow([label, metadata.get(key) or ""])

    writer.writerow([])
    writer.writerow(["Device", "Count", f"Unit Cost ({report.currency})", "Total Cost"])
    for line in report.lines:
        writer.writerow(
            [line.display_name, line.count, f"{line.unit_cost:.2f}", f"{line.line_total:.2f}"]
        )
    writer.writerow([])
    writer.writerow(["Grand Total", "", "", f"{report.grand_total:.2f}"])

    buffer.seek(0)
    safe_name = project.filename.rsplit(".", 1)[0] or "report"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_costing_report.csv"'
        },
    )
