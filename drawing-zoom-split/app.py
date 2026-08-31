"""Dual-Pathway Drawing Split API + static UI."""

from __future__ import annotations

import io
import json
import logging
import shutil
import sys
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline.extraction.requirement_pdf import requirement_pdf_filename  # noqa: E402
from pipeline.extraction.sheet_notes_pdf import sheet_notes_pdf_filename  # noqa: E402
from pipeline.extraction.symbol_table_pdf import technical_symbol_pdf_filename  # noqa: E402
from pipeline.job import create_job, job_id_from_filename, process_job, read_job  # noqa: E402
from pipeline.page_render import IMAGE_SUFFIXES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("drawing-zoom-split")

STATIC_DIR = ROOT / "static"
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
UI_PORT = 8766

app = FastAPI(title="Dual-Pathway Drawing Split", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_job_lock = threading.Lock()
_running: set[str] = set()


def _list_jobs() -> list[dict]:
    config.ensure_storage_dirs()
    jobs: list[dict] = []
    for job_path in config.STORAGE_DIR.glob("*/job.json"):
        try:
            data = json.loads(job_path.read_text(encoding="utf-8"))
            jobs.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    jobs.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return jobs


def _job_pdf_path(job: dict, kind: str) -> Path | None:
    filename = job.get("filename") or job.get("id") or ""
    if kind == "requirement":
        rel = job.get("requirement_pdf_path")
        fallback = config.REQUIREMENTS_DIR / requirement_pdf_filename(filename)
    elif kind == "technical_symbol":
        rel = job.get("technical_symbol_pdf_path")
        fallback = config.TECHNICAL_SYMBOLS_DIR / technical_symbol_pdf_filename(filename)
    elif kind == "sheet_notes":
        rel = job.get("sheet_notes_pdf_path")
        fallback = config.STORAGE_DIR / job["id"] / sheet_notes_pdf_filename(filename)
    else:
        return None
    if rel:
        path = Path(rel)
        if path.is_file():
            return path
    return fallback if fallback.is_file() else None


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    model = (
        config.GROQ_MODEL_ID
        if config.VISION_LLM_RUNTIME == "groq"
        else (
            config.HF_INFERENCE_ENDPOINT or config.HF_MODEL_ID
            if config.VISION_LLM_RUNTIME == "hosted"
            else config.HF_MODEL_ID
        )
    )
    return {
        "ok": True,
        "model": model,
        "runtime": config.VISION_LLM_RUNTIME,
        "groq_key_set": bool(config.GROQ_API_KEY),
        "endpoint_configured": bool(config.HF_INFERENCE_ENDPOINT),
        "token_set": bool(config.HF_TOKEN),
        "load_4bit": config.HF_LOAD_4BIT,
    }


def _run_job(job_id: str) -> None:
    try:
        process_job(job_id)
    finally:
        with _job_lock:
            _running.discard(job_id)


@app.post("/api/jobs")
async def create_upload(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suffix}. Use PDF or an image.",
        )
    job_id = job_id_from_filename(filename)
    with _job_lock:
        if job_id in _running:
            raise HTTPException(
                status_code=409,
                detail=f"Job for '{job_id}' is already running. Wait for it to finish.",
            )
    staging = config.STORAGE_DIR / "_staging" / job_id
    staging.mkdir(parents=True, exist_ok=True)
    dest = staging / filename
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    dest.write_bytes(payload)
    try:
        job = create_job(job_id, filename, dest)
    finally:
        dest.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    with _job_lock:
        _running.add(job_id)
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return job


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return _list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return read_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found") from None


def _safe_file(job_id: str, rel: str) -> Path:
    root = (config.STORAGE_DIR / job_id).resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


@app.get("/api/jobs/{job_id}/requirement.pdf")
def job_requirement_pdf(job_id: str) -> FileResponse:
    job = get_job(job_id)
    path = _job_pdf_path(job, "requirement")
    if path is None:
        raise HTTPException(status_code=404, detail="Requirement PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/api/jobs/{job_id}/technical-symbol.pdf")
def job_technical_symbol_pdf(job_id: str) -> FileResponse:
    job = get_job(job_id)
    path = _job_pdf_path(job, "technical_symbol")
    if path is None:
        raise HTTPException(status_code=404, detail="Technical symbol PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/api/jobs/{job_id}/sheet-notes.pdf")
def job_sheet_notes_pdf(job_id: str) -> FileResponse:
    job = get_job(job_id)
    path = _job_pdf_path(job, "sheet_notes")
    if path is None:
        raise HTTPException(status_code=404, detail="Sheet notes PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/api/jobs/{job_id}/files/{rel_path:path}")
def job_file(job_id: str, rel_path: str) -> FileResponse:
    path = _safe_file(job_id, rel_path)
    media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
    if path.suffix.lower() == ".png":
        media = "image/png"
    if path.suffix.lower() == ".json":
        media = "application/json"
    return FileResponse(path, media_type=media)


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str) -> StreamingResponse:
    root = (config.STORAGE_DIR / job_id).resolve()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )


if __name__ == "__main__":
    import uvicorn

    config.ensure_storage_dirs()
    uvicorn.run("app:app", host="127.0.0.1", port=UI_PORT, reload=False)
