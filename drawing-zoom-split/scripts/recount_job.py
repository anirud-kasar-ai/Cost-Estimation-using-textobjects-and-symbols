"""Run the YOLO symbol-count + invoice stage on already-processed jobs.

Use case:
Older job folders were split into wings and zoom tiles before the counting
stage existed, so they have 04_wings/<wing>/zooms/<page>/plan_zoom_*.jpg but
no 06_counts/. This script walks every wing on every kept page, runs YOLO on
each wing's zoom tiles, merges all detections into one whole-PDF count, and
writes 06_counts/result.json + invoice.json + invoice.csv — exactly what
process_job() does at the end of a fresh run, minus the LLM/render stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline.count_stage import COUNTS_DIR, run_symbol_count_stage  # noqa: E402
from pipeline.job import read_job, write_job  # noqa: E402

logger = logging.getLogger("recount_job")


def recount_one_job(job_root: Path) -> dict | None:
    job_id = job_root.name
    try:
        job = read_job(job_id)
    except FileNotFoundError:
        logger.warning("[%s] missing job.json; skipping", job_id)
        return None

    pages = job.get("pages") or []
    kept = [p for p in pages if p.get("kept")]
    wings = [w for p in kept for w in (p.get("wings") or [])]
    if not wings:
        logger.warning("[%s] no wings recorded in job.json; skipping", job_id)
        return None

    logger.info(
        "[%s] counting symbols on %d wing(s) across %d page(s)",
        job_id,
        len(wings),
        len(kept),
    )

    def progress(stage: str, message: str) -> None:
        logger.info("[%s] %s: %s", job_id, stage, message)

    payload = run_symbol_count_stage(
        job_root,
        pages,
        job_id=job_id,
        progress=progress,
    )

    invoice = payload.get("invoice") or {}
    job["symbol_count_total"] = payload.get("total_count", 0)
    job["symbol_count_rows"] = len(payload.get("rows") or [])
    job["counts_path"] = f"{COUNTS_DIR}/result.json"
    job["invoice_path"] = f"{COUNTS_DIR}/invoice.json"
    job["invoice_csv_path"] = f"{COUNTS_DIR}/invoice.csv"
    job["invoice_grand_total_usd"] = invoice.get("grand_total_usd")
    job["invoice_subtotal_usd"] = invoice.get("subtotal_usd")
    job["yolo_enabled"] = bool(payload.get("yolo_enabled"))
    job["yolo_model"] = payload.get("yolo_model")
    if payload.get("method") == "yolo":
        job["message"] = (
            f"Done — {job.get('drawing_count', 0)} diagram(s), "
            f"{job.get('wing_count', 0)} wing(s), "
            f"{job.get('symbol_count_total', 0)} symbols counted, "
            f"invoice ${float(invoice.get('grand_total_usd') or 0):.2f}"
        )
    write_job(job_id, job)

    # Console summary: per-wing counts, then whole-PDF totals.
    per_wing = payload.get("per_wing") or []
    if per_wing:
        logger.info("[%s] per-wing symbol counts:", job_id)
        for row in per_wing:
            logger.info(
                "[%s]   page %-3s %-45s %4d detected (%d matched to legend)",
                job_id,
                row.get("page"),
                row.get("wing"),
                row.get("yolo_count", 0),
                row.get("matched", 0),
            )
    logger.info(
        "[%s] TOTAL: %d symbols across %d wing folder(s) → invoice $%.2f (%s)",
        job_id,
        payload.get("total_count", 0),
        payload.get("zoom_folders_processed", 0),
        float(invoice.get("grand_total_usd") or 0),
        f"{COUNTS_DIR}/invoice.csv",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run YOLO symbol count + invoice on existing jobs"
    )
    parser.add_argument(
        "jobs",
        nargs="*",
        help="Job folder names under storage/jobs/ (e.g. 1948BIDDrawings). "
        "If omitted, recount all jobs that have wings but no 06_counts/result.json.",
    )
    parser.add_argument("--all", action="store_true", help="Recount every job.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    jobs_root = config.STORAGE_DIR
    if not jobs_root.exists():
        logger.error("storage/jobs/ root not found: %s", jobs_root)
        return 2

    if args.all:
        job_ids = sorted(
            p.name
            for p in jobs_root.iterdir()
            if p.is_dir() and (p / "job.json").exists()
        )
    elif args.jobs:
        job_ids = args.jobs
    else:
        job_ids = sorted(
            p.name
            for p in jobs_root.iterdir()
            if p.is_dir()
            and (p / "job.json").exists()
            and not (p / COUNTS_DIR / "result.json").exists()
        )

    if not job_ids:
        logger.info("No jobs selected.")
        return 0

    for job_id in job_ids:
        job_root = jobs_root / job_id
        if not job_root.exists():
            logger.warning("Missing job folder: %s", job_id)
            continue
        recount_one_job(job_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
