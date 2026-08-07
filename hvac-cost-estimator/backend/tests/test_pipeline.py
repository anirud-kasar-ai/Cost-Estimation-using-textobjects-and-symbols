"""Tests for the end-to-end pipeline orchestrator (mock model backends)."""

from __future__ import annotations

from pathlib import Path

import pytest

from db.models import Project, ProjectStatus
from ml.pipeline import run_pipeline


class TestRunPipeline:
    def test_full_flow_on_sample_pdf(self, sample_pdf: Path, settings) -> None:
        result = run_pipeline(sample_pdf, settings.storage_dir / "pages", settings)

        # Text branch produced structured metadata
        assert result.metadata.client == "Meridian Property Group"
        assert result.metadata.engineer == "Vector Building Services Ltd."
        assert result.metadata.due_date == "2026-08-15"

        # Symbol branch produced classified devices
        assert len(result.devices) > 0
        assert all(device.confidence > 0 for device in result.devices)

        # Costing joined counts against the rate table
        assert result.costing is not None
        assert result.costing.grand_total > 0
        assert sum(line.count for line in result.costing.lines) == len(result.devices)

        # Page images were written
        assert (settings.storage_dir / "pages" / "page_001.png").exists()

    def test_deterministic_across_runs(self, sample_pdf: Path, settings) -> None:
        first = run_pipeline(sample_pdf, settings.storage_dir / "run1", settings)
        second = run_pipeline(sample_pdf, settings.storage_dir / "run2", settings)

        assert first.metadata == second.metadata
        assert first.costing is not None and second.costing is not None
        assert first.costing.lines == second.costing.lines

    def test_raises_on_corrupt_pdf(self, tmp_path: Path, settings) -> None:
        from ml.pdf_to_image import PdfValidationError

        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"%PDF-1.7\nbroken")
        with pytest.raises(PdfValidationError):
            run_pipeline(corrupt, settings.storage_dir / "pages", settings)


class TestProcessProject:
    @pytest.fixture()
    def patched_session(self, db_session, monkeypatch: pytest.MonkeyPatch):
        """Point the pipeline's session factory at the test database."""
        import db.session as db_session_module

        monkeypatch.setattr(db_session_module, "SessionLocal", lambda: db_session)
        # Prevent the test session from being closed by process_project
        monkeypatch.setattr(db_session, "close", lambda: None)
        return db_session

    def test_success_path_persists_results(
        self, sample_pdf: Path, settings, patched_session
    ) -> None:
        from ml.pipeline import process_project

        project = Project(filename="sample.pdf", file_path=str(sample_pdf))
        patched_session.add(project)
        patched_session.commit()

        process_project(project.id)

        patched_session.refresh(project)
        assert project.status == ProjectStatus.DONE
        assert project.client == "Meridian Property Group"
        assert len(project.pages) == 1
        assert len(project.device_lines) > 0
        for line in project.device_lines:
            assert line.count == line.detected_count
            assert line.line_total == round(line.count * line.unit_cost, 2)

    def test_failure_path_records_error(
        self, tmp_path: Path, settings, patched_session
    ) -> None:
        from ml.pipeline import process_project

        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"not a pdf at all")
        project = Project(filename="corrupt.pdf", file_path=str(corrupt))
        patched_session.add(project)
        patched_session.commit()

        process_project(project.id)

        patched_session.refresh(project)
        assert project.status == ProjectStatus.FAILED
        assert project.error_message is not None
        assert "PDF" in project.error_message
