from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import app, _job_states, _running, _state_lock  # noqa: E402
from pipeline import config  # noqa: E402
from pipeline.cv.nms import BoundingBox, SymbolDetection  # noqa: E402


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_upload_and_result(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path / "storage" / "jobs", raising=False)
    config.ensure_storage_dirs()
    monkeypatch.setattr(config, "SYMBOL_COUNT_SYNC_JOBS", True, raising=False)

    fake_payload = {
        "status": "done",
        "method": "single_image_cv",
        "legend_entry_count": 2,
        "post_nms_count": 1,
        "counts": {"R": 1},
        "rows": [],
        "detections": [],
        "notes": [],
    }

    with patch("app.run_symbol_count_job", return_value=fake_payload):
        client = TestClient(app)
        legend = {
            "entry_count": 1,
            "legend_pages": [1],
            "total_pages": 1,
            "notes": [],
            "entries": [
                {
                    "symbol": "R",
                    "description": "DATA RACK",
                    "notes": None,
                    "source_page": 1,
                    "legend_title": "X",
                    "mfg_model": None,
                    "part_number": None,
                    "has_glyph": False,
                }
            ],
        }
        files = {
            "symbol_file": ("legend.json", json.dumps(legend).encode("utf-8"), "application/json"),
            "image_file": ("plan.png", _make_png_bytes(), "image/png"),
        }
        resp = client.post("/api/jobs", files=files)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "done"


def test_health_reports_cv():
    client = TestClient(app)
    data = client.get("/api/health").json()
    assert data["method"] in {"cv", "cv+llm"}
    assert data["method"] == ("cv+llm" if data["llm_enabled"] else "cv")
    assert "tesseract_available" in data
    assert data.get("folder_upload") is True


def test_api_folder_upload(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path / "storage" / "jobs", raising=False)
    config.ensure_storage_dirs()
    monkeypatch.setattr(config, "SYMBOL_COUNT_SYNC_JOBS", True, raising=False)

    fake_payload = {
        "status": "done",
        "method": "detect_merge_evaluate_count",
        "legend_entry_count": 1,
        "post_nms_count": 1,
        "tile_count": 2,
        "counts": {"#": 1},
        "rows": [],
        "detections": [],
        "notes": [],
    }

    with patch("app.run_symbol_count_folder_job", return_value=fake_payload) as mocked:
        client = TestClient(app)
        legend = {
            "entry_count": 1,
            "legend_pages": [1],
            "total_pages": 1,
            "notes": [],
            "entries": [
                {
                    "symbol": "#",
                    "description": "DATA",
                    "notes": None,
                    "source_page": 1,
                    "legend_title": "X",
                    "mfg_model": None,
                    "part_number": None,
                    "has_glyph": False,
                }
            ],
        }
        files = [
            ("symbol_file", ("legend.json", json.dumps(legend).encode("utf-8"), "application/json")),
            ("tile_files", ("plan_zoom_r00_c00.png", _make_png_bytes(), "image/png")),
            ("tile_files", ("plan_zoom_r00_c01.png", _make_png_bytes(), "image/png")),
        ]
        resp = client.post("/api/jobs", files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "folder"
        assert mocked.called
        status_resp = client.get(f"/api/jobs/{body['job_id']}")
        assert status_resp.json()["status"] == "done"


def test_resume_orphaned_processing_job(tmp_path, monkeypatch):
    import app as app_mod

    jobs = tmp_path / "jobs"
    monkeypatch.setattr(config, "STORAGE_DIR", jobs, raising=False)
    monkeypatch.setattr(config, "SYMBOL_COUNT_SYNC_JOBS", True, raising=False)
    job_id = "orphan_folder_job"
    out = jobs / job_id
    out.mkdir(parents=True)
    (out / "symbol_file.json").write_text("{}", encoding="utf-8")
    tiles = out / "tiles"
    tiles.mkdir()
    Image.new("RGB", (32, 32), (255, 255, 255)).save(tiles / "plan_zoom_r00_c00.png")
    (out / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "processing",
                "stage": "detect_tiles_then_merge",
                "error": None,
                "created_at": 1.0,
                "finished_at": None,
                "filename": "1 zoom tiles",
                "mode": "folder",
            }
        ),
        encoding="utf-8",
    )
    with _state_lock:
        _running.discard(job_id)
        _job_states.pop(job_id, None)

    called = {"n": 0}

    def fake_run(jid: str) -> None:
        called["n"] += 1
        app_mod._set_job_state(jid, status="done", stage=None, finished_at=1.0)

    monkeypatch.setattr(app_mod, "_run_job", fake_run)
    client = TestClient(app)
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert called["n"] == 1
    assert resp.json()["status"] == "done"
