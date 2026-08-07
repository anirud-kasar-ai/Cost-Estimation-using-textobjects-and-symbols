"""PDF upload endpoint. Saves the file, creates the project row, and kicks
off pipeline processing as a background task so the response returns
immediately; the frontend polls the project status."""

from __future__ import annotations

import logging
import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config import Settings, get_settings
from db.models import Project
from db.session import get_db
from ml.pdf_to_image import PdfValidationError, validate_pdf
from ml.pipeline import process_project
from schemas.project import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload an HVAC layout PDF for processing",
)
async def upload_pdf(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted (expected a .pdf extension).",
        )

    project_id = uuid.uuid4().hex
    project_dir = settings.storage_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = project_dir / "original.pdf"

    written = 0
    max_bytes = settings.max_upload_bytes  # None = no size limit
    try:
        with pdf_path.open("wb") as target:
            while chunk := await file.read(CHUNK_SIZE):
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise HTTPException(
                        status_code=413,  # Content Too Large
                        detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
                    )
                target.write(chunk)

        if written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
            )

        try:
            validate_pdf(pdf_path)
        except PdfValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
    except HTTPException:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise

    project = Project(id=project_id, filename=filename, file_path=str(pdf_path))
    db.add(project)
    db.commit()

    background_tasks.add_task(process_project, project.id)
    logger.info("Accepted upload %s as project %s", filename, project.id)
    return UploadResponse(project_id=project.id, status=project.status)
