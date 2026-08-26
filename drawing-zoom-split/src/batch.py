"""Batch runner: process every PDF in an input directory."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pipeline import config
from pipeline.job import create_job, job_id_from_filename, process_job

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str, str], None]


@dataclass
class BatchResult:
    pdf: str
    job_id: str
    status: str
    message: str
    diagrams: int = 0
    wings: int = 0
    error: str | None = None


@dataclass
class BatchReport:
    input_dir: str
    output_dir: str
    started_at: str
    finished_at: str = ""
    results: list[BatchResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == "done")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status != "done")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_pdfs(input_dir: Path) -> list[Path]:
    """Return sorted PDF paths from a directory (non-recursive)."""
    if not input_dir.is_dir():
        raise NotADirectoryError(str(input_dir))
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def process_one_pdf(
    pdf_path: Path,
    *,
    on_progress: ProgressFn | None = None,
) -> BatchResult:
    job_id = job_id_from_filename(pdf_path.name)

    def job_progress(stage: str, message: str) -> None:
        if on_progress:
            on_progress(job_id, stage, message)

    create_job(job_id, pdf_path.name, pdf_path)
    job = process_job(job_id, on_progress=job_progress)
    return BatchResult(
        pdf=pdf_path.name,
        job_id=job_id,
        status=job.get("status") or "unknown",
        message=job.get("message") or "",
        diagrams=int(job.get("drawing_count") or 0),
        wings=int(job.get("wing_count") or 0),
        error=job.get("error"),
    )


def run_batch(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    on_progress: ProgressFn | None = None,
) -> BatchReport:
    """Process all PDFs in input_dir; write one job folder per PDF."""
    if output_dir is not None:
        config.set_storage_dir(output_dir)
    output_dir = config.STORAGE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report = BatchReport(
        input_dir=str(input_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        started_at=_now(),
    )

    pdfs = discover_pdfs(input_dir)
    if not pdfs:
        logger.warning("No PDF files found in %s", input_dir)

    for pdf_path in pdfs:
        logger.info("Processing %s", pdf_path.name)
        if on_progress:
            on_progress(pdf_path.stem, "batch", f"Starting {pdf_path.name}")
        try:
            result = process_one_pdf(pdf_path, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed on %s", pdf_path.name)
            result = BatchResult(
                pdf=pdf_path.name,
                job_id=job_id_from_filename(pdf_path.name),
                status="failed",
                message=str(exc),
                error=str(exc),
            )
        report.results.append(result)
        logger.info(
            "%s -> %s (%s diagrams, %s wings)",
            pdf_path.name,
            result.status,
            result.diagrams,
            result.wings,
        )

    report.finished_at = _now()
    manifest_path = output_dir / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": report.input_dir,
                "output_dir": report.output_dir,
                "started_at": report.started_at,
                "finished_at": report.finished_at,
                "pdf_count": len(report.results),
                "ok_count": report.ok_count,
                "fail_count": report.fail_count,
                "results": [
                    {
                        "pdf": r.pdf,
                        "job_id": r.job_id,
                        "status": r.status,
                        "message": r.message,
                        "diagrams": r.diagrams,
                        "wings": r.wings,
                        "error": r.error,
                    }
                    for r in report.results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
