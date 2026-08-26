#!/usr/bin/env python3
"""Backfill symbol counting for an existing job folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline.job import read_job, write_job  # noqa: E402
from pipeline.symbol_count import run_symbol_count  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wing symbol counting for a job")
    parser.add_argument(
        "--job-id",
        required=True,
        help="Job folder name under storage/jobs/",
    )
    args = parser.parse_args()

    job_root = config.STORAGE_DIR / args.job_id
    if not job_root.is_dir():
        print(f"Job folder not found: {job_root}", file=sys.stderr)
        return 1

    def progress(stage: str, message: str) -> None:
        print(f"[{stage}] {message}")

    try:
        result = run_symbol_count(job_root, on_progress=progress)
    except Exception as exc:  # noqa: BLE001
        print(f"Symbol count failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))

    try:
        job = read_job(args.job_id)
        job.update(
            {
                "symbol_count_status": result.get("status"),
                "symbol_count_instances": result.get("instances", 0),
                "symbol_count_detail_path": result.get("symbol_count_detail_path"),
                "symbol_count_report_csv": result.get("symbol_count_report_csv"),
                "symbol_count_report_pdf": result.get("symbol_count_report_pdf"),
                "symbol_count_report_json": result.get("symbol_count_report_json"),
            }
        )
        if result.get("reason"):
            job["symbol_count_reason"] = result["reason"]
        write_job(args.job_id, job)
    except FileNotFoundError:
        pass

    return 0 if result.get("status") == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
