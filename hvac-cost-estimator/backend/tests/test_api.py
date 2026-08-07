"""API tests: upload validation, project CRUD, overrides, report + CSV export.

Uploads are processed synchronously here (TestClient runs background tasks
before returning the response), always with mock model backends.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(settings, db_session, monkeypatch: pytest.MonkeyPatch):
    import db.session as db_session_module
    import main

    # Route the app (and the background pipeline task) to the test database.
    monkeypatch.setattr(db_session_module, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)
    monkeypatch.setattr(main, "init_db", lambda: None)
    main.app.dependency_overrides[db_session_module.get_db] = lambda: db_session

    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


def _upload(client: TestClient, pdf: Path) -> str:
    response = client.post(
        "/api/upload",
        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 202, response.text
    return response.json()["project_id"]


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


class TestUpload:
    def test_rejects_non_pdf_extension(self, client: TestClient) -> None:
        response = client.post(
            "/api/upload", files={"file": ("layout.png", b"abc", "image/png")}
        )
        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]

    def test_rejects_non_pdf_content(self, client: TestClient) -> None:
        response = client.post(
            "/api/upload",
            files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        )
        assert response.status_code == 400
        assert "does not look like a PDF" in response.json()["detail"]
        assert client.get("/api/projects").json() == []

    def test_rejects_empty_file(self, client: TestClient) -> None:
        response = client.post(
            "/api/upload", files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        assert response.status_code == 400

    def test_no_size_limit_by_default(self, client: TestClient, sample_pdf: Path) -> None:
        from config import get_settings

        assert get_settings().max_upload_bytes is None
        # Upload succeeds without any size validation kicking in
        _upload(client, sample_pdf)

    def test_rejects_oversized_file_when_cap_configured(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The size cap is optional (off by default) but still enforceable."""
        from config import get_settings

        monkeypatch.setenv("HVAC_MAX_UPLOAD_MB", "0")
        get_settings.cache_clear()
        response = client.post(
            "/api/upload",
            files={"file": ("big.pdf", b"%PDF-1.7 data", "application/pdf")},
        )
        assert response.status_code == 413

    def test_happy_path_processes_to_done(
        self, client: TestClient, sample_pdf: Path
    ) -> None:
        project_id = _upload(client, sample_pdf)

        detail = client.get(f"/api/projects/{project_id}").json()
        assert detail["status"] == "done"
        assert detail["metadata"]["client"] == "Meridian Property Group"
        assert detail["metadata"]["due_date"] == "2026-08-15"
        assert detail["page_count"] == 1
        assert len(detail["device_lines"]) > 0
        assert detail["grand_total"] == round(
            sum(line["line_total"] for line in detail["device_lines"]), 2
        )


class TestProjects:
    def test_missing_project_404(self, client: TestClient) -> None:
        assert client.get("/api/projects/deadbeef").status_code == 404

    def test_list_projects(self, client: TestClient, sample_pdf: Path) -> None:
        project_id = _upload(client, sample_pdf)
        listed = client.get("/api/projects").json()
        assert [item["id"] for item in listed] == [project_id]

    def test_delete_removes_project_and_files(
        self, client: TestClient, sample_pdf: Path, settings
    ) -> None:
        project_id = _upload(client, sample_pdf)
        assert (settings.storage_dir / project_id).exists()

        assert client.delete(f"/api/projects/{project_id}").status_code == 204
        assert client.get(f"/api/projects/{project_id}").status_code == 404
        assert not (settings.storage_dir / project_id).exists()


class TestLineOverrides:
    def test_override_recalculates_totals(
        self, client: TestClient, sample_pdf: Path
    ) -> None:
        project_id = _upload(client, sample_pdf)
        detail = client.get(f"/api/projects/{project_id}").json()
        line = detail["device_lines"][0]

        response = client.patch(
            f"/api/projects/{project_id}/lines/{line['id']}",
            json={"count": line["count"] + 5, "unit_cost": 999.0},
        )
        assert response.status_code == 200
        updated = response.json()
        updated_line = next(
            item for item in updated["device_lines"] if item["id"] == line["id"]
        )
        assert updated_line["count"] == line["count"] + 5
        assert updated_line["unit_cost"] == 999.0
        assert updated_line["line_total"] == round((line["count"] + 5) * 999.0, 2)
        # Original detection values preserved for the override UX
        assert updated_line["detected_count"] == line["detected_count"]
        assert updated["grand_total"] == round(
            sum(item["line_total"] for item in updated["device_lines"]), 2
        )

    def test_negative_count_rejected(
        self, client: TestClient, sample_pdf: Path
    ) -> None:
        project_id = _upload(client, sample_pdf)
        line_id = client.get(f"/api/projects/{project_id}").json()["device_lines"][0]["id"]

        response = client.patch(
            f"/api/projects/{project_id}/lines/{line_id}", json={"count": -1}
        )
        assert response.status_code == 422

    def test_infinite_unit_cost_rejected(
        self, client: TestClient, sample_pdf: Path
    ) -> None:
        """Regression: 1e999 parses to float('inf') and used to pass ge=0."""
        project_id = _upload(client, sample_pdf)
        detail = client.get(f"/api/projects/{project_id}").json()
        line_id = detail["device_lines"][0]["id"]

        # Raw body so the literal 1e999 reaches Pydantic's JSON parser
        # (json.dumps would already have serialized Python inf differently).
        response = client.patch(
            f"/api/projects/{project_id}/lines/{line_id}",
            content=b'{"unit_cost": 1e999}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

        # Totals untouched by the rejected update
        after = client.get(f"/api/projects/{project_id}").json()
        assert after["grand_total"] == detail["grand_total"]

    def test_values_above_sanity_caps_rejected(
        self, client: TestClient, sample_pdf: Path
    ) -> None:
        project_id = _upload(client, sample_pdf)
        detail = client.get(f"/api/projects/{project_id}").json()
        line_id = detail["device_lines"][0]["id"]

        too_many = client.patch(
            f"/api/projects/{project_id}/lines/{line_id}", json={"count": 10_000_000}
        )
        assert too_many.status_code == 422

        too_expensive = client.patch(
            f"/api/projects/{project_id}/lines/{line_id}",
            json={"unit_cost": 10_000_001},
        )
        assert too_expensive.status_code == 422

        after = client.get(f"/api/projects/{project_id}").json()
        assert after["grand_total"] == detail["grand_total"]

    def test_unknown_line_404(self, client: TestClient, sample_pdf: Path) -> None:
        project_id = _upload(client, sample_pdf)
        response = client.patch(
            f"/api/projects/{project_id}/lines/nope", json={"count": 1}
        )
        assert response.status_code == 404


class TestRequirementPdf:
    def test_upload_produces_downloadable_requirement_pdf(
        self, client: TestClient, sample_pdf: Path
    ) -> None:
        project_id = _upload(client, sample_pdf)
        detail = client.get(f"/api/projects/{project_id}").json()

        assert detail["has_requirement_pdf"] is True
        assert detail["requirement_provider"] is not None

        response = client.get(f"/api/projects/{project_id}/requirement.pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "requirement.pdf" in response.headers.get("content-disposition", "")
        assert response.content[:5] == b"%PDF-"

    def test_requirement_pdf_404_when_missing(
        self, client: TestClient, db_session
    ) -> None:
        from db.models import Project, ProjectStatus

        project = Project(
            filename="no-req.pdf",
            file_path="x.pdf",
            status=ProjectStatus.DONE,
            requirement_pdf_path=None,
        )
        db_session.add(project)
        db_session.commit()

        response = client.get(f"/api/projects/{project.id}/requirement.pdf")
        assert response.status_code == 404


class TestReports:
    def test_report_json(self, client: TestClient, sample_pdf: Path) -> None:
        project_id = _upload(client, sample_pdf)
        report = client.get(f"/api/projects/{project_id}/report").json()

        assert report["project_id"] == project_id
        assert report["metadata"]["architect"] == "Atelier North Architects"
        assert report["grand_total"] == round(
            sum(line["line_total"] for line in report["lines"]), 2
        )
        assert report["currency"] == "USD"

    def test_report_conflict_when_not_done(
        self, client: TestClient, db_session
    ) -> None:
        from db.models import Project

        project = Project(filename="pending.pdf", file_path="x.pdf")
        db_session.add(project)
        db_session.commit()

        response = client.get(f"/api/projects/{project.id}/report")
        assert response.status_code == 409

    def test_csv_export(self, client: TestClient, sample_pdf: Path) -> None:
        project_id = _upload(client, sample_pdf)
        response = client.get(f"/api/projects/{project_id}/report/csv")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]

        body = response.text
        assert "Field,Information" in body
        assert "Client,Meridian Property Group" in body
        assert "Grand Total" in body
