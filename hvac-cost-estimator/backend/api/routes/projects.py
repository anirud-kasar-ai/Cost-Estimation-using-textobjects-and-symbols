"""Project CRUD + manual override of device line counts/costs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Settings, get_settings
from db.models import DetectionInstance, DeviceLine, Project
from db.session import get_db
from ml.cost_calculator import load_cost_rates
from ml.requirement_pdf import requirement_pdf_filename
from ml.symbol_table_pdf import technical_symbol_pdf_filename
from schemas.project import (
    DetectionReviewOut,
    DeviceLineOut,
    DeviceLineUpdate,
    ProjectDetail,
    ProjectMetadata,
    ProjectSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _get_project_or_404(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found.",
        )
    return project


def build_project_detail(project: Project, settings: Settings) -> ProjectDetail:
    """Assemble the full project response, computing totals server-side."""
    currency = "USD"
    try:
        currency = load_cost_rates(settings.cost_rates_path).currency
    except Exception:  # rate table problems shouldn't break project reads
        logger.warning("Could not load cost rate table for currency", exc_info=True)

    lines = [DeviceLineOut.model_validate(line) for line in project.device_lines]
    req_path = project.requirement_pdf_path
    sym_path = project.technical_symbol_pdf_path
    return ProjectDetail(
        id=project.id,
        filename=project.filename,
        status=project.status,
        error_message=project.error_message,
        created_at=project.created_at,
        metadata=ProjectMetadata(
            title=project.title,
            client=project.client,
            architect=project.architect,
            engineer=project.engineer,
            project_address=project.project_address,
            due_date=project.due_date,
        ),
        device_lines=lines,
        grand_total=round(sum(line.line_total for line in lines), 2),
        currency=currency,
        page_count=len(project.pages),
        has_requirement_pdf=bool(req_path and Path(req_path).exists()),
        requirement_provider=project.requirement_provider,
        pages_truncated=bool(project.pages_truncated),
        has_technical_symbol_pdf=bool(sym_path and Path(sym_path).exists()),
    )


@router.get("", response_model=list[ProjectSummary], summary="List all projects")
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return list(
        db.scalars(select(Project).order_by(Project.created_at.desc())).all()
    )


@router.get("/{project_id}", response_model=ProjectDetail, summary="Get one project")
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ProjectDetail:
    return build_project_detail(_get_project_or_404(db, project_id), settings)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project and its stored files",
)
def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> None:
    project = _get_project_or_404(db, project_id)
    db.delete(project)
    db.commit()
    shutil.rmtree(settings.storage_dir / project_id, ignore_errors=True)


@router.get(
    "/{project_id}/requirement.pdf",
    summary="Download the extracted requirement PDF",
    response_class=FileResponse,
)
def download_requirement_pdf(
    project_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    project = _get_project_or_404(db, project_id)
    if not project.requirement_pdf_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No requirement PDF is available for this project.",
        )
    path = Path(project.requirement_pdf_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requirement PDF file is missing on disk.",
        )
    download_name = requirement_pdf_filename(project.filename)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=download_name,
        content_disposition_type="attachment",
    )


@router.get(
    "/{project_id}/technical-symbol.pdf",
    summary="Download the extracted technical symbol PDF",
    response_class=FileResponse,
)
def download_technical_symbol_pdf(
    project_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    project = _get_project_or_404(db, project_id)
    if not project.technical_symbol_pdf_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No technical symbol PDF is available for this project.",
        )
    path = Path(project.technical_symbol_pdf_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Technical symbol PDF file is missing on disk.",
        )
    download_name = technical_symbol_pdf_filename(project.filename)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=download_name,
        content_disposition_type="attachment",
    )


@router.patch(
    "/{project_id}/lines/{line_id}",
    response_model=ProjectDetail,
    summary="Override a device line's count and/or unit cost",
)
def update_device_line(
    project_id: str,
    line_id: str,
    payload: DeviceLineUpdate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ProjectDetail:
    project = _get_project_or_404(db, project_id)
    line = db.get(DeviceLine, line_id)
    if line is None or line.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device line {line_id} not found on project {project_id}.",
        )

    if payload.count is not None:
        line.count = payload.count
    if payload.unit_cost is not None:
        line.unit_cost = payload.unit_cost
    db.commit()
    db.refresh(project)
    return build_project_detail(project, settings)


def _detection_for_line(db: Session, project: Project, line: DeviceLine) -> DetectionInstance:
    if line.sample_detection_id:
        det = db.get(DetectionInstance, line.sample_detection_id)
        if det is not None and det.project_id == project.id:
            return det
    det = next(
        (d for d in project.detections if d.device_line_id == line.id),
        None,
    )
    if det is None:
        det = next(
            (d for d in project.detections if d.device_type == line.device_type),
            None,
        )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No detection crop is available for this line item.",
        )
    return det


@router.get(
    "/{project_id}/lines/{line_id}/review",
    response_model=DetectionReviewOut,
    summary="Review payload: sheet crop for a costing line item",
)
def review_device_line(
    project_id: str,
    line_id: str,
    db: Session = Depends(get_db),
) -> DetectionReviewOut:
    project = _get_project_or_404(db, project_id)
    line = db.get(DeviceLine, line_id)
    if line is None or line.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device line {line_id} not found on project {project_id}.",
        )
    det = _detection_for_line(db, project, line)
    page = next((p for p in project.pages if p.page_number == det.page_number), None)
    return DetectionReviewOut(
        detection_id=det.id,
        device_line_id=line.id,
        page_number=det.page_number,
        device_type=det.device_type,
        confidence=det.confidence,
        room_label=det.room_label,
        quantity=det.quantity,
        unit=det.unit,
        bbox=(det.x1, det.y1, det.x2, det.y2),
        crop_url=f"/api/projects/{project.id}/detections/{det.id}/crop",
        page_image_url=(
            f"/api/projects/{project.id}/pages/{page.page_number}/image"
            if page is not None
            else None
        ),
        needs_review=det.needs_review or line.needs_review,
    )


@router.get(
    "/{project_id}/detections/{detection_id}/crop",
    summary="PNG crop for a single detection (review view)",
    response_class=FileResponse,
)
def download_detection_crop(
    project_id: str,
    detection_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    project = _get_project_or_404(db, project_id)
    det = db.get(DetectionInstance, detection_id)
    if det is None or det.project_id != project.id:
        raise HTTPException(status_code=404, detail="Detection not found.")
    if not det.crop_path or not Path(det.crop_path).exists():
        raise HTTPException(status_code=404, detail="Detection crop file is missing.")
    return FileResponse(
        Path(det.crop_path),
        media_type="image/png",
        filename=f"{detection_id}.png",
    )


@router.get(
    "/{project_id}/pages/{page_number}/image",
    summary="Full rendered page image",
    response_class=FileResponse,
)
def download_page_image(
    project_id: str,
    page_number: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    project = _get_project_or_404(db, project_id)
    page = next((p for p in project.pages if p.page_number == page_number), None)
    if page is None or not Path(page.image_path).exists():
        raise HTTPException(status_code=404, detail="Page image not found.")
    return FileResponse(Path(page.image_path), media_type="image/png")

