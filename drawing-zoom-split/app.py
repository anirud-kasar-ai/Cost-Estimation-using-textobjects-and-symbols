"""Dual-Pathway Drawing Split API + static UI."""

from __future__ import annotations

import io
import json
import os
import logging
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import zipfile
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
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
from pipeline.job import (  # noqa: E402
    create_job,
    ensure_client_info,
    job_id_from_filename,
    process_job,
    read_job,
    write_job,
)
from pipeline.page_render import IMAGE_SUFFIXES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("drawing-zoom-split")

STATIC_DIR = ROOT / "static"
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
UI_PORT = int(os.environ.get("UI_PORT", "8766"))

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
# Process at most 2 jobs concurrently; extra uploads wait in "queued" state.
_job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")


def _list_jobs() -> list[dict]:
    config.ensure_storage_dirs()
    jobs: list[dict] = []
    for job_path in config.STORAGE_DIR.glob("*/job.json"):
        try:
            data = json.loads(job_path.read_text(encoding="utf-8"))
            if not data.get("size_bytes"):
                # Jobs created before size tracking: fill in from the stored PDF.
                source = job_path.parent / "01_source"
                pdf = next(iter(sorted(source.glob("*.pdf"))), None) if source.is_dir() else None
                if pdf is not None:
                    data["size_bytes"] = pdf.stat().st_size
            # Jobs created before client extraction: read the cover sheet once.
            ensure_client_info(data)
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
    yolo_ok = False
    yolo_classes: list[str] = []
    yolo_models: list[str] = []
    try:
        from pipeline.yolo_detect import available_model_paths, class_names, yolo_available

        yolo_ok = bool(config.YOLO_ENABLED and yolo_available())
        if yolo_ok:
            yolo_classes = sorted(class_names().values())
            yolo_models = [p.name for p in available_model_paths()]
    except Exception:  # noqa: BLE001
        yolo_ok = False
    return {
        "ok": True,
        "model": model,
        "runtime": config.VISION_LLM_RUNTIME,
        "groq_key_set": bool(config.GROQ_API_KEY),
        "endpoint_configured": bool(config.HF_INFERENCE_ENDPOINT),
        "token_set": bool(config.HF_TOKEN),
        "load_4bit": config.HF_LOAD_4BIT,
        "yolo": yolo_ok,
        "yolo_model": Path(config.YOLO_MODEL_PATH).name if yolo_ok else None,
        "yolo_models": yolo_models,
        "yolo_classes": yolo_classes,
        "symbol_count_enabled": config.SYMBOL_COUNT_ENABLED,
        "pricing_csv": str(config.PRICING_CSV_PATH),
    }


@app.get("/api/jobs/{job_id}/counts")
def job_counts(job_id: str) -> dict:
    path = config.STORAGE_DIR / job_id / "06_counts" / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Symbol counts not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/overlay")
def job_overlay(job_id: str) -> dict:
    """Wing images + detection boxes (wing-image pixel space) for visual review."""
    root = config.STORAGE_DIR / job_id
    path = root / "06_counts" / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Symbol counts not found")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    dets = result.get("yolo_detections") or []
    per_wing = result.get("per_wing") or []

    # Group detections by wing folder. New jobs record wing_folder per hit;
    # older jobs stored a flat list in per_wing order, so slice sequentially.
    groups: dict[str, list[dict]] = {}
    if dets and dets[0].get("wing_folder"):
        for d in dets:
            groups.setdefault(str(d.get("wing_folder") or ""), []).append(d)
    else:
        idx = 0
        for w in per_wing:
            n = int(w.get("yolo_count") or 0)
            groups[str(w.get("folder") or "")] = dets[idx : idx + n]
            idx += n

    det_keys = ("class_name", "confidence", "x1", "y1", "x2", "y2", "legend_number")
    wings = []
    for w in per_wing:
        folder = str(w.get("folder") or "")
        image_rel = None
        for cand in ("full_wing.jpg", "full_wing.png"):
            if (root / folder / cand).is_file():
                image_rel = f"{folder}/{cand}"
                break
        wings.append(
            {
                "wing": w.get("wing"),
                "page": w.get("page"),
                "folder": folder,
                "image": image_rel,
                "yolo_count": w.get("yolo_count"),
                "matched": w.get("matched"),
                "detections": [
                    {k: d.get(k) for k in det_keys} for d in groups.get(folder, [])
                ],
            }
        )
    return {"job_id": job_id, "yolo_model": result.get("yolo_model"), "wings": wings}


@app.get("/api/jobs/{job_id}/corrections")
def get_corrections(job_id: str) -> dict:
    path = config.STORAGE_DIR / job_id / "06_counts" / "corrections.json"
    if not path.is_file():
        return {"saved_at": None, "lines": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"saved_at": None, "lines": []}


@app.put("/api/jobs/{job_id}/corrections")
def put_corrections(job_id: str, payload: dict = Body(...)) -> dict:
    """Persist human-verified counts/prices so model accuracy can be tracked."""
    from datetime import datetime, timezone

    out_dir = config.STORAGE_DIR / job_id / "06_counts"
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job has no counts yet")
    clean = []
    for ln in payload.get("lines") or []:
        if not isinstance(ln, dict):
            continue
        try:
            clean.append(
                {
                    "key": str(ln.get("key") or ""),
                    "description": str(ln.get("description") or ""),
                    "model_count": int(ln.get("model_count") or 0),
                    "corrected_count": int(ln.get("corrected_count") or 0),
                    "model_unit_price": round(float(ln.get("model_unit_price") or 0), 2),
                    "corrected_unit_price": round(
                        float(ln.get("corrected_unit_price") or 0), 2
                    ),
                    "note": str(ln.get("note") or "").strip(),
                }
            )
        except (TypeError, ValueError):
            continue
    doc = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lines": clean,
    }
    (out_dir / "corrections.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


@app.put("/api/jobs/{job_id}/tax")
def update_tax_rate(job_id: str, payload: dict = Body(...)) -> dict:
    """Set a custom tax rate (fraction, e.g. 0.045) and rebuild the invoice files."""
    out_dir = config.STORAGE_DIR / job_id / "06_counts"
    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Symbol counts not found")
    try:
        rate = float(payload.get("tax_rate"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="tax_rate must be a number") from None
    if rate < 0 or rate > 1:
        raise HTTPException(
            status_code=400, detail="tax_rate must be a fraction between 0 and 1 (4.5% = 0.045)"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    from pipeline.count_stage import write_invoice_csv
    from pipeline.pricing import build_invoice

    invoice = build_invoice(
        result.get("rows") or [], job_id=job_id, refresh_prices=False, tax_rate=rate
    )
    result["invoice"] = invoice
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "invoice.json").write_text(json.dumps(invoice, indent=2), encoding="utf-8")
    write_invoice_csv(out_dir, invoice)
    # Regenerate the printable PDF so downloads match the new tax
    try:
        from pipeline.invoice_pdf import generate_invoice_pdf

        try:
            job = read_job(job_id)
        except FileNotFoundError:
            job = None
        generate_invoice_pdf(config.STORAGE_DIR / job_id, job)
    except Exception:  # noqa: BLE001
        logger.exception("Invoice PDF regeneration failed for %s", job_id)
    return invoice


@app.put("/api/jobs/{job_id}/client")
def update_client(job_id: str, payload: dict = Body(...)) -> dict:
    """Save the Bill-To client block on the job and regenerate the estimate PDF."""
    try:
        job = read_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    job["client"] = {
        "name": str(payload.get("name") or "").strip(),
        "address": str(payload.get("address") or "").strip(),
        "contact": str(payload.get("contact") or "").strip(),
    }
    write_job(job_id, job)
    # Rebuild the PDF so the downloaded estimate shows the new Bill-To block.
    if (config.STORAGE_DIR / job_id / "06_counts" / "invoice.json").is_file():
        try:
            from pipeline.invoice_pdf import generate_invoice_pdf

            generate_invoice_pdf(config.STORAGE_DIR / job_id, job)
        except Exception:  # noqa: BLE001
            logger.exception("Estimate PDF regeneration failed for %s", job_id)
    return job


@app.get("/api/jobs/{job_id}/invoice")
def job_invoice(job_id: str) -> dict:
    path = config.STORAGE_DIR / job_id / "06_counts" / "invoice.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Invoice not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/invoice.csv")
def job_invoice_csv(job_id: str) -> FileResponse:
    path = config.STORAGE_DIR / job_id / "06_counts" / "invoice.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Invoice CSV not found")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}_invoice.csv")


@app.get("/api/jobs/{job_id}/invoice.pdf")
def job_invoice_pdf(job_id: str) -> FileResponse:
    root = config.STORAGE_DIR / job_id
    path = root / "06_counts" / "invoice.pdf"
    if not path.is_file():
        # Older jobs were counted before PDF export existed — build on demand.
        if not (root / "06_counts" / "invoice.json").is_file():
            raise HTTPException(status_code=404, detail="Invoice not found")
        from pipeline.invoice_pdf import generate_invoice_pdf

        try:
            job = read_job(job_id)
        except FileNotFoundError:
            job = None
        try:
            path = generate_invoice_pdf(root, job)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(
        path, media_type="application/pdf", filename=f"{job_id}_invoice.pdf"
    )


_train_metrics_cache: dict[str, dict | None] = {}


def _model_train_metrics(path: Path) -> dict | None:
    """Training metrics embedded in an Ultralytics checkpoint (cached by mtime)."""
    key = f"{path.resolve()}::{path.stat().st_mtime_ns}"
    if key in _train_metrics_cache:
        return _train_metrics_cache[key]
    metrics: dict | None = None
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        tm = ckpt.get("train_metrics") or {}
        ta = ckpt.get("train_args") or {}
        if tm:
            metrics = {
                "precision": round(float(tm.get("metrics/precision(B)") or 0), 4),
                "recall": round(float(tm.get("metrics/recall(B)") or 0), 4),
                "map50": round(float(tm.get("metrics/mAP50(B)") or 0), 4),
                "map50_95": round(float(tm.get("metrics/mAP50-95(B)") or 0), 4),
                "epochs": ta.get("epochs") if isinstance(ta, dict) else None,
            }
    except Exception:  # noqa: BLE001
        metrics = None
    _train_metrics_cache[key] = metrics
    return metrics


@app.get("/api/stats")
def get_stats() -> dict:
    """Aggregate detection stats across all finished jobs (for dashboards)."""
    from collections import Counter
    from datetime import datetime, timezone

    per_class: Counter[str] = Counter()
    matched_class: Counter[str] = Counter()
    confidences: dict[str, list[float]] = {}
    durations: list[float] = []
    jobs_done = 0
    total_symbols = 0
    corr_jobs = 0
    corr_lines = 0
    corr_model_total = 0
    corr_human_total = 0
    corr_abs_err = 0
    for job in _list_jobs():
        if job.get("status") != "done":
            continue
        jobs_done += 1
        if job.get("actual_seconds"):
            durations.append(float(job["actual_seconds"]))
        corr_path = config.STORAGE_DIR / job["id"] / "06_counts" / "corrections.json"
        if corr_path.is_file():
            try:
                corr = json.loads(corr_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                corr = None
            lines = (corr or {}).get("lines") or []
            if lines:
                corr_jobs += 1
                for ln in lines:
                    m = int(ln.get("model_count") or 0)
                    h = int(ln.get("corrected_count") or 0)
                    corr_lines += 1
                    corr_model_total += m
                    corr_human_total += h
                    corr_abs_err += abs(m - h)
        result_path = config.STORAGE_DIR / job["id"] / "06_counts" / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total_symbols += int(result.get("total_count") or 0)
        for det in result.get("yolo_detections") or []:
            name = str(det.get("class_name") or "")
            if not name:
                continue
            per_class[name] += 1
            if det.get("legend_number") is not None:
                matched_class[name] += 1
            confidences.setdefault(name, []).append(float(det.get("confidence") or 0))

    models = []
    trained_class_models: dict[str, list[str]] = {}
    model_train_metrics: dict[str, dict | None] = {}
    try:
        from pipeline.yolo_detect import available_model_paths, class_names

        for path in available_model_paths():
            stat = path.stat()
            cls_list = sorted(class_names(path).values())
            for name in cls_list:
                trained_class_models.setdefault(name, []).append(path.name)
            train_metrics = _model_train_metrics(path)
            model_train_metrics[path.name] = train_metrics
            models.append(
                {
                    "name": path.name,
                    "classes": cls_list,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "active": path.name == Path(config.YOLO_MODEL_PATH).name,
                    "train_metrics": train_metrics,
                }
            )
    except Exception:  # noqa: BLE001
        pass

    # One row per class the models are TRAINED on (plus any historic detected
    # classes from older weights), with observed performance signals.
    all_class_names = sorted(set(trained_class_models) | set(per_class))
    classes = []
    for name in all_class_names:
        count = int(per_class.get(name, 0))
        matched = int(matched_class.get(name, 0))
        class_models = trained_class_models.get(name, [])
        # Training metrics of the best (highest mAP50) model trained on this class
        best_tm = None
        for model_name in class_models:
            tm = model_train_metrics.get(model_name)
            if tm and tm.get("map50") is not None:
                if best_tm is None or tm["map50"] > best_tm["map50"]:
                    best_tm = tm
        classes.append(
            {
                "class_name": name,
                "models": class_models,
                "detections": count,
                "matched": matched,
                "match_rate": round(matched / count, 4) if count else None,
                "avg_confidence": (
                    round(sum(confidences[name]) / len(confidences[name]), 4)
                    if confidences.get(name)
                    else None
                ),
                "train_map50": best_tm["map50"] if best_tm else None,
                "train_precision": best_tm["precision"] if best_tm else None,
                "train_recall": best_tm["recall"] if best_tm else None,
            }
        )
    classes.sort(key=lambda c: (-c["detections"], c["class_name"]))

    return {
        "jobs_done": jobs_done,
        "total_symbols": total_symbols,
        "avg_processing_seconds": (
            round(sum(durations) / len(durations), 1) if durations else None
        ),
        "classes": classes,
        "models": models,
        "verification": {
            "jobs_verified": corr_jobs,
            "lines_verified": corr_lines,
            "model_count_total": corr_model_total,
            "human_count_total": corr_human_total,
            "count_accuracy": (
                round(1.0 - corr_abs_err / corr_human_total, 4)
                if corr_human_total > 0
                else None
            ),
        },
    }


@app.get("/api/pricing")
def get_pricing() -> dict:
    from dataclasses import asdict

    from pipeline.pricing import ensure_pricing_csv, load_price_index

    ensure_pricing_csv(refresh=False)
    index = load_price_index()
    from pipeline.pricing import load_price_overrides

    overrides = load_price_overrides()
    items = []
    for v in index.values():
        item = asdict(v)
        ov = overrides.get(v.device_key)
        item["overridden"] = ov is not None
        item["override_reason"] = (ov or {}).get("reason") or None
        items.append(item)
    return {
        "csv": str(config.PRICING_CSV_PATH),
        "count": len(index),
        "items": items,
    }


@app.put("/api/pricing/{device_key}")
def update_price(device_key: str, payload: dict = Body(...)) -> dict:
    """Manually set a unit price — persists across synthetic price refreshes."""
    from dataclasses import asdict

    from pipeline.pricing import set_price_override

    try:
        price = float(payload.get("unit_price_usd"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="unit_price_usd must be a number") from None
    if price < 0:
        raise HTTPException(status_code=400, detail="Price cannot be negative")
    try:
        row = set_price_override(
            device_key,
            price,
            description=payload.get("description"),
            unit=payload.get("unit"),
            reason=payload.get("reason"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    item = asdict(row)
    item["overridden"] = True
    item["override_reason"] = str(payload.get("reason") or "").strip() or None
    return item
def _run_job(job_id: str) -> None:
    try:
        process_job(job_id)
    finally:
        with _job_lock:
            _running.discard(job_id)


@app.on_event("startup")
def _resume_incomplete_jobs() -> None:
    """Re-enqueue jobs left in queued/processing state by a previous run.

    Without this, a backend restart orphans in-flight jobs: job.json still says
    "processing" but no worker thread exists, so they sit stuck forever.
    """
    for job in _list_jobs():
        job_id = job.get("id")
        if not job_id or job.get("status") not in ("queued", "processing"):
            continue
        with _job_lock:
            if job_id in _running:
                continue
            _running.add(job_id)
        job["status"] = "queued"
        job["stage"] = "queued"
        job["message"] = "Re-queued after server restart"
        write_job(job_id, job)
        _job_executor.submit(_run_job, job_id)
        logger.info("Resumed incomplete job %s after restart", job_id)


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
            # Same file dropped again while its job is still processing:
            # attach to the running job instead of failing the upload.
            try:
                return read_job(job_id)
            except FileNotFoundError:
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
    _job_executor.submit(_run_job, job_id)
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
