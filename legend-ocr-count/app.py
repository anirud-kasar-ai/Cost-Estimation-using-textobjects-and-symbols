from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeline import config
from pipeline.count_job import run_count_job
from pipeline.legend_ocr import tesseract_available
from pipeline.yolo_detect import class_names, yolo_available

logger = logging.getLogger("legend-ocr-count")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Legend OCR + YOLO Symbol Count", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    yolo_ok = bool(config.YOLO_ENABLED and yolo_available())
    callouts_on = bool(config.CALLOUT_OCR_ENABLED)
    if yolo_ok and callouts_on:
        method = "yolo+ocr_callout"
        note = (
            "YOLO detects symbols; OCR reads circled legend numbers. "
            "Counts prefer YOLO when class names match legend descriptions."
        )
    elif yolo_ok:
        method = "yolo"
        note = (
            "YOLO-only mode — callout OCR is paused. "
            "Set CALLOUT_OCR_ENABLED=true in .env and restart to re-enable."
        )
    elif callouts_on:
        method = "ocr_callout"
        note = "YOLO model missing or disabled — OCR callouts only."
    else:
        method = "none"
        note = "Both YOLO and callout OCR are disabled."
    return {
        "ok": True,
        "method": method,
        "tesseract_available": tesseract_available(),
        "yolo": yolo_ok,
        "callout_ocr": callouts_on,
        "yolo_model": config.YOLO_MODEL_PATH.name if yolo_ok else None,
        "yolo_classes": list(class_names().values()) if yolo_ok else [],
        "yolo_conf": config.YOLO_CONF if yolo_ok else None,
        "folder_upload": True,
        "note": note,
    }


@dataclass
class JobState:
    status: str
    stage: str | None = None
    error: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None
    filename: str | None = None
    mode: str | None = None


_state_lock = threading.Lock()
_job_states: dict[str, JobState] = {}
_running: set[str] = set()


def _job_dir(job_id: str) -> Path:
    return config.STORAGE_DIR / job_id


def _job_json_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _write_job_state(job_id: str, state: JobState) -> None:
    path = _job_json_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"job_id": job_id, **asdict(state)}, indent=2), encoding="utf-8")


def _read_job_state(job_id: str) -> JobState | None:
    path = _job_json_path(job_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return JobState(
        status=str(data.get("status") or "queued"),
        stage=data.get("stage"),
        error=data.get("error"),
        created_at=float(data.get("created_at") or 0.0),
        finished_at=data.get("finished_at"),
        filename=data.get("filename"),
        mode=data.get("mode"),
    )


def _set_job_state(job_id: str, **updates: Any) -> JobState:
    with _state_lock:
        st = _job_states.get(job_id) or _read_job_state(job_id) or JobState(
            status="queued", created_at=time.time()
        )
        for key, value in updates.items():
            setattr(st, key, value)
        _job_states[job_id] = st
        _write_job_state(job_id, st)
        return st


def _job_id_from_uploads(legend_filename: str, plan_filename: str) -> str:
    leg = Path(legend_filename).stem.replace(" ", "_")
    plan = Path(plan_filename).stem.replace(" ", "_")
    base = f"{leg}__{plan}"
    h = uuid.uuid5(uuid.NAMESPACE_URL, base).hex[:8]
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in base)
    return f"{cleaned[:60]}_{h}"


def _validate_suffix(filename: str, allowed: set[str], kind: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {kind} type {suffix!r}. Allowed: {sorted(allowed)}",
        )
    return suffix


def _clear_job_artifacts(out_dir: Path) -> None:
    for pattern in ("legend_file.*", "plan_image.*", "overlay.png", "result.json", "folder_summary.png"):
        for old in out_dir.glob(pattern):
            old.unlink(missing_ok=True)
    tiles = out_dir / "plans"
    if tiles.exists():
        shutil.rmtree(tiles, ignore_errors=True)


def _run_job(job_id: str) -> None:
    out_dir = _job_dir(job_id)
    try:
        _set_job_state(job_id, status="processing", stage="starting", error=None)

        legend_files = sorted(out_dir.glob("legend_file.*"))
        if not legend_files:
            raise RuntimeError("Legend file upload missing.")
        legend_path = legend_files[0]

        plans_dir = out_dir / "plans"
        if plans_dir.is_dir() and any(plans_dir.iterdir()):
            run_count_job(
                legend_path,
                None,
                out_dir,
                folder_dir=plans_dir,
                progress=lambda stage: _set_job_state(job_id, stage=stage),
            )
        else:
            # Upload is stored as plan_image.<ext>; job also writes plan_image.png later.
            candidates = [
                p
                for p in out_dir.glob("plan_image.*")
                if p.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES
            ]
            if not candidates:
                raise RuntimeError("Plan image or folder upload missing.")
            # Prefer the original upload when both .jpg and regenerated .png exist.
            plan_path = next(
                (p for p in candidates if p.name != "plan_image.png"),
                candidates[0],
            )
            run_count_job(
                legend_path,
                plan_path,
                out_dir,
                progress=lambda stage: _set_job_state(job_id, stage=stage),
            )

        _set_job_state(
            job_id,
            status="done",
            stage=None,
            error=None,
            finished_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        _set_job_state(
            job_id,
            status="failed",
            stage=None,
            error=str(exc),
            finished_at=time.time(),
        )
    finally:
        with _state_lock:
            _running.discard(job_id)


def _job_inputs_ready(job_id: str) -> bool:
    out_dir = _job_dir(job_id)
    if not any(out_dir.glob("legend_file.*")):
        return False
    plans = out_dir / "plans"
    if plans.is_dir() and any(plans.iterdir()):
        return True
    return any(out_dir.glob("plan_image.*"))


def _start_job_thread(job_id: str) -> bool:
    with _state_lock:
        if job_id in _running:
            return False
        _running.add(job_id)
    if config.SYNC_JOBS:
        _run_job(job_id)
        return True
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return True


def _resume_orphaned_job(job_id: str, st: JobState) -> JobState:
    if st.status not in {"queued", "processing"}:
        return st
    if (_job_dir(job_id) / "result.json").is_file():
        return _set_job_state(
            job_id, status="done", stage=None, error=None, finished_at=time.time()
        )
    with _state_lock:
        if job_id in _running:
            return st
    if not _job_inputs_ready(job_id):
        return _set_job_state(
            job_id,
            status="failed",
            error="Job stopped when the server reloaded. Upload again.",
            finished_at=time.time(),
        )
    if _start_job_thread(job_id):
        logger.warning("Resuming job %s after server restart", job_id)
        if config.SYNC_JOBS:
            return _read_job_state(job_id) or st
        return _set_job_state(job_id, status="processing", error=None)
    return st


@app.on_event("startup")
def _on_startup() -> None:
    import os

    if os.getenv("PYTEST_CURRENT_TEST") or config.SYNC_JOBS:
        return
    if not config.STORAGE_DIR.is_dir():
        return
    for path in config.STORAGE_DIR.glob("*/job.json"):
        job_id = path.parent.name
        st = _read_job_state(job_id)
        if st is not None:
            _resume_orphaned_job(job_id, st)


@app.post("/api/jobs")
async def create_job(
    legend_file: UploadFile = File(...),
    image_file: UploadFile | None = File(None),
    plan_files: list[UploadFile] | None = File(None),
) -> dict[str, str]:
    if not legend_file.filename:
        raise HTTPException(status_code=400, detail="legend_file is required.")

    legend_suffix = _validate_suffix(
        legend_file.filename, config.ALLOWED_LEGEND_SUFFIXES, "legend file"
    )

    folder_uploads = [t for t in (plan_files or []) if t is not None and t.filename]
    mode = "folder" if folder_uploads else "single"

    if mode == "single":
        if not image_file or not image_file.filename:
            raise HTTPException(
                status_code=400,
                detail="Provide image_file (single plan) or plan_files (folder).",
            )
        image_suffix = _validate_suffix(
            image_file.filename, config.ALLOWED_IMAGE_SUFFIXES, "image file"
        )
        job_id = _job_id_from_uploads(legend_file.filename, image_file.filename)
        display_name = image_file.filename
    else:
        job_id = _job_id_from_uploads(
            legend_file.filename, f"folder_{len(folder_uploads)}_images"
        )
        display_name = f"{len(folder_uploads)} plan image(s)"

    with _state_lock:
        if job_id in _running:
            raise HTTPException(status_code=409, detail=f"Job '{job_id}' is already running.")
        _running.add(job_id)

    out_dir = _job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_job_artifacts(out_dir)

    legend_path = out_dir / f"legend_file{legend_suffix}"
    legend_bytes = await legend_file.read()
    if not legend_bytes:
        with _state_lock:
            _running.discard(job_id)
        raise HTTPException(status_code=400, detail="Empty legend upload")
    legend_path.write_bytes(legend_bytes)

    if mode == "single":
        assert image_file is not None and image_file.filename
        image_path = out_dir / f"plan_image{image_suffix}"
        image_bytes = await image_file.read()
        if not image_bytes:
            with _state_lock:
                _running.discard(job_id)
            raise HTTPException(status_code=400, detail="Empty image upload")
        image_path.write_bytes(image_bytes)
    else:
        plans_dir = out_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for upload in folder_uploads:
            name = Path((upload.filename or "").replace("\\", "/")).name
            suffix = Path(name).suffix.lower()
            data = await upload.read()
            if not data:
                continue
            if name.lower() == "zooms_manifest.json":
                (plans_dir / "zooms_manifest.json").write_bytes(data)
                continue
            if suffix not in config.ALLOWED_IMAGE_SUFFIXES:
                continue
            (plans_dir / name).write_bytes(data)
            saved += 1
        if saved == 0:
            with _state_lock:
                _running.discard(job_id)
            raise HTTPException(status_code=400, detail="Folder has no plan images.")

    _set_job_state(
        job_id,
        status="queued",
        stage=None,
        error=None,
        created_at=time.time(),
        finished_at=None,
        filename=display_name,
        mode=mode,
    )

    if config.SYNC_JOBS:
        _run_job(job_id)
    else:
        threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "mode": mode}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    with _state_lock:
        st = _job_states.get(job_id)
    if st is None:
        st = _read_job_state(job_id)
        if st is not None:
            with _state_lock:
                _job_states[job_id] = st
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status in {"queued", "processing"}:
        st = _resume_orphaned_job(job_id, st)
    return {
        "job_id": job_id,
        "status": st.status,
        "stage": st.stage,
        "error": st.error,
        "mode": st.mode,
    }


@app.get("/api/jobs/{job_id}/result.json")
def job_result(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/overlay.png")
def job_overlay(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "overlay.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Overlay not ready")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/plan-image")
def job_plan_image(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "plan_image.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not ready")
    return FileResponse(path)
