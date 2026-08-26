"""API tests for symbol count report endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import app as fastapi_app  # noqa: E402
from pipeline import config  # noqa: E402
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo  # noqa: E402
from pipeline.symbol_count_report import write_symbol_count_json  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(fastapi_app.app)


def test_symbol_count_json_endpoint(client: TestClient, tmp_path: Path, monkeypatch):
    job_id = "test-symbol-count-api"
    job_root = tmp_path / job_id
    job_root.mkdir(parents=True)
    out_dir = job_root / config.SYMBOL_COUNT_DIR
    out_dir.mkdir(parents=True)

    legend = SymbolTableInfo(
        entries=[SymbolEntry(description="DATA RACK", symbol="R")],
    )
    write_symbol_count_json(
        legend=legend,
        instance_results=[{"column_key": "B-WING (page_001)", "counts": {"R": 1}}],
        output_path=out_dir / "symbol_count_report.json",
    )

    (job_root / "job.json").write_text(
        json.dumps(
            {
                "id": job_id,
                "status": "done",
                "filename": "test.pdf",
                "symbol_count_status": "done",
                "symbol_count_report_json": f"{config.SYMBOL_COUNT_DIR}/symbol_count_report.json",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)

    response = client.get(f"/api/jobs/{job_id}/symbol-count.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["columns"] == ["B-WING (page_001)"]
    assert payload["rows"][0]["total"] == 1


def test_symbol_count_json_endpoint_missing(client: TestClient, tmp_path: Path, monkeypatch):
    job_id = "missing-symbol-count"
    job_root = tmp_path / job_id
    job_root.mkdir(parents=True)
    (job_root / "job.json").write_text(
        json.dumps({"id": job_id, "status": "done", "filename": "test.pdf"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)

    response = client.get(f"/api/jobs/{job_id}/symbol-count.json")
    assert response.status_code == 404
