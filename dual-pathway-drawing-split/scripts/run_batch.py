"""Batch-process every PDF in an input directory.

Usage:
  python scripts/run_batch.py --input "D:\\Cost Estimation Using Text and Object\\Real data"
  python scripts/run_batch.py --input path/to/pdfs --output storage/jobs --verify
  python scripts/run_batch.py --input path/to/pdfs --files 1948BIDDrawings.pdf 1962BIDMHESDrawings.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from batch import discover_pdfs, process_one_pdf, run_batch  # noqa: E402
from pipeline import config  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch split construction PDFs by wing")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "storage" / "jobs",
        help="Output root for job folders (default: storage/jobs)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Process only these PDF filenames from --input",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verify_output.py after batch completes",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    config.set_storage_dir(args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=True)

    if args.files:
        selected = []
        for name in args.files:
            pdf = input_dir / name
            if not pdf.is_file():
                print(f"SKIP missing {name}", file=sys.stderr)
                continue
            selected.append(pdf)
        if not selected:
            print("ERROR: no valid PDFs from --files", file=sys.stderr)
            return 2

        from datetime import datetime, timezone
        import json

        started = datetime.now(timezone.utc).isoformat()
        results = []
        for pdf_path in selected:

            def progress(job_id: str, stage: str, message: str) -> None:
                print(f"  [{job_id}] {stage}: {message}", flush=True)

            print(f"\n=== {pdf_path.name}", flush=True)
            try:
                result = process_one_pdf(pdf_path, on_progress=progress)
            except Exception as exc:  # noqa: BLE001
                from batch import BatchResult
                from pipeline.job import job_id_from_filename

                result = BatchResult(
                    pdf=pdf_path.name,
                    job_id=job_id_from_filename(pdf_path.name),
                    status="failed",
                    message=str(exc),
                    error=str(exc),
                )
            mark = "OK" if result.status == "done" else "FAIL"
            print(
                f"  {mark}  diagrams={result.diagrams}  wings={result.wings}  "
                f"{result.message}",
                flush=True,
            )
            results.append(result)

        finished = datetime.now(timezone.utc).isoformat()
        manifest = {
            "input_dir": str(input_dir),
            "output_dir": str(config.STORAGE_DIR),
            "started_at": started,
            "finished_at": finished,
            "pdf_count": len(results),
            "ok_count": sum(1 for r in results if r.status == "done"),
            "fail_count": sum(1 for r in results if r.status != "done"),
            "results": [
                {
                    "pdf": r.pdf,
                    "job_id": r.job_id,
                    "status": r.status,
                    "message": r.message,
                    "diagrams": r.diagrams,
                    "wings": r.wings,
                    "error": r.error,
                }
                for r in results
            ],
        }
        (config.STORAGE_DIR / "batch_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        report_ok = manifest["fail_count"] == 0
    else:
        report = run_batch(input_dir, config.STORAGE_DIR)
        print(
            f"\nBatch complete: {report.ok_count}/{len(report.results)} OK, "
            f"{report.fail_count} failed"
        )
        for result in report.results:
            mark = "OK" if result.status == "done" else "FAIL"
            print(
                f"  {mark}  {result.pdf:45s}  "
                f"diagrams={result.diagrams:3d}  wings={result.wings:3d}"
            )
        report_ok = report.fail_count == 0

    if args.verify:
        import subprocess

        job_ids = [
            r.job_id for r in (results if args.files else report.results)
        ]
        cmd = [sys.executable, str(ROOT / "scripts" / "verify_output.py"), *job_ids]
        verify_code = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
        if verify_code != 0:
            return verify_code

    return 0 if report_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
