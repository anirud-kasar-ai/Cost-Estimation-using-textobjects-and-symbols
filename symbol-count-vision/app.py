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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeline import config
from pipeline.cv.ocr_tags import tesseract_available
from pipeline.symbol_count import run_symbol_count_job
from pipeline.tile_merge import collect_tile_uploads, run_symbol_count_folder_job

logger = logging.getLogger("symbol-count-cv")

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

ALLOWED_SYMBOL_SUFFIXES = {".json", ".pdf"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Symbol Count CV", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    enabled = config.llm_enabled()
    return {
        "ok": True,
        "method": "cv+llm" if enabled else "cv",
        "tesseract_available": tesseract_available(),
        "template_threshold": config.CV_TEMPLATE_THRESHOLD,
        "ocr_min_conf": config.CV_OCR_MIN_CONF,
        "nms_iou": config.SYMBOL_COUNT_NMS_IOU,
        "llm_enabled": enabled,
        "llm_provider": config.llm_provider() if enabled else None,
        "llm_model": config.llm_model() if enabled else None,
        "folder_upload": True,
        "tile_overlap_pct": 0.10,
    }


@dataclass
class JobState:
    status: str  # queued/processing/done/failed
    stage: str | None = None
    error: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None
    filename: str | None = None
    mode: str | None = None  # single | folder


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
    payload = {"job_id": job_id, **asdict(state)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _job_id_from_uploads(symbol_filename: str, image_filename: str) -> str:
    sym_stem = Path(symbol_filename).stem.replace(" ", "_")
    img_stem = Path(image_filename).stem.replace(" ", "_")
    base = f"{sym_stem}__{img_stem}"
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
    for old in out_dir.glob("symbol_file.*"):
        old.unlink(missing_ok=True)
    for old in out_dir.glob("plan_image.*"):
        old.unlink(missing_ok=True)
    (out_dir / "result.json").unlink(missing_ok=True)
    (out_dir / "overlay.png").unlink(missing_ok=True)
    (out_dir / "overlay_rejected.png").unlink(missing_ok=True)
    (out_dir / "overlay_with_rejected.png").unlink(missing_ok=True)
    (out_dir / "evaluation.json").unlink(missing_ok=True)
    (out_dir / "plan_image.png").unlink(missing_ok=True)
    tiles = out_dir / "tiles"
    if tiles.exists():
        shutil.rmtree(tiles, ignore_errors=True)
    (out_dir / "zooms_manifest.json").unlink(missing_ok=True)


def _run_job(job_id: str) -> None:
    out_dir = _job_dir(job_id)

    try:
        _set_job_state(job_id, status="processing", stage="counting_symbols", error=None)

        symbol_files = sorted(out_dir.glob("symbol_file.*"))
        if not symbol_files:
            raise RuntimeError("Symbol file upload missing.")
        symbol_path = symbol_files[0]

        tiles_dir = out_dir / "tiles"
        if tiles_dir.is_dir() and any(tiles_dir.iterdir()):
            manifest = out_dir / "zooms_manifest.json"
            if not manifest.is_file():
                nested = tiles_dir / "zooms_manifest.json"
                manifest = nested if nested.is_file() else None
            _set_job_state(job_id, stage="detect_tiles_then_merge")
            run_symbol_count_folder_job(
                symbol_path,
                tiles_dir,
                out_dir,
                manifest_path=manifest if manifest and manifest.is_file() else None,
            )
        else:
            image_files = sorted(out_dir.glob("plan_image.*"))
            if not image_files:
                raise RuntimeError("Plan image or zoom folder upload missing.")
            run_symbol_count_job(symbol_path, image_files[0], out_dir)

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


@app.post("/api/jobs")
async def create_job(
    symbol_file: UploadFile = File(...),
    image_file: UploadFile | None = File(None),
    tile_files: list[UploadFile] | None = File(None),
) -> dict[str, str]:
    if not symbol_file.filename:
        raise HTTPException(status_code=400, detail="symbol_file is required.")

    symbol_suffix = _validate_suffix(symbol_file.filename, ALLOWED_SYMBOL_SUFFIXES, "symbol file")

    tile_uploads = [t for t in (tile_files or []) if t is not None and t.filename]
    mode = "folder" if tile_uploads else "single"

    if mode == "single":
        if not image_file or not image_file.filename:
            raise HTTPException(
                status_code=400,
                detail="Provide image_file (single plan) or tile_files (zoom folder).",
            )
        image_suffix = _validate_suffix(image_file.filename, ALLOWED_IMAGE_SUFFIXES, "image file")
        job_id = _job_id_from_uploads(symbol_file.filename, image_file.filename)
        display_name = image_file.filename
    else:
        job_id = _job_id_from_uploads(symbol_file.filename, f"folder_{len(tile_uploads)}_tiles")
        display_name = f"{len(tile_uploads)} zoom tiles"

    with _state_lock:
        if job_id in _running:
            raise HTTPException(status_code=409, detail=f"Job '{job_id}' is already running.")
        _running.add(job_id)

    out_dir = _job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_job_artifacts(out_dir)

    symbol_path = out_dir / f"symbol_file{symbol_suffix}"
    symbol_bytes = await symbol_file.read()
    if not symbol_bytes:
        with _state_lock:
            _running.discard(job_id)
        _set_job_state(job_id, status="failed", error="Empty upload", finished_at=time.time())
        raise HTTPException(status_code=400, detail="Empty symbol upload")
    symbol_path.write_bytes(symbol_bytes)

    if mode == "single":
        assert image_file is not None and image_file.filename
        image_path = out_dir / f"plan_image{image_suffix}"
        image_bytes = await image_file.read()
        if not image_bytes:
            with _state_lock:
                _running.discard(job_id)
            _set_job_state(job_id, status="failed", error="Empty upload", finished_at=time.time())
            raise HTTPException(status_code=400, detail="Empty image upload")
        image_path.write_bytes(image_bytes)
    else:
        tiles_dir = out_dir / "tiles"
        pairs: list[tuple[str, bytes]] = []
        for upload in tile_uploads:
            name = upload.filename or ""
            data = await upload.read()
            suffix = Path(name).suffix.lower()
            base = Path(name.replace("\\", "/")).name.lower()
            if base == "zooms_manifest.json" or suffix in ALLOWED_IMAGE_SUFFIXES:
                pairs.append((name, data))
        if not any(
            Path(n.replace("\\", "/")).suffix.lower() in ALLOWED_IMAGE_SUFFIXES for n, _ in pairs
        ):
            with _state_lock:
                _running.discard(job_id)
            raise HTTPException(status_code=400, detail="Zoom folder has no image tiles.")
        manifest = collect_tile_uploads(pairs, tiles_dir)
        if manifest and manifest.is_file():
            # Also copy to job root for the runner.
            shutil.copy2(manifest, out_dir / "zooms_manifest.json")

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

    if config.SYMBOL_COUNT_SYNC_JOBS:
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


@app.get("/api/jobs/{job_id}/overlay_with_rejected.png")
def job_overlay_with_rejected(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "overlay_with_rejected.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Combined overlay not ready")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/plan-image")
def job_plan_image(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "plan_image.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not ready")
    return FileResponse(path)
