"""Verify wing ROI + zoom outputs for Real data PDFs.

Maps each PDF in ``Real data/`` to a job folder under ``storage/jobs/`` (by PDF stem)
and runs ROI/zoom checks via verify_output.py logic.

Usage:
  python scripts/verify_realdata_roi_zoom.py
  python scripts/verify_realdata_roi_zoom.py --pdfs 1948BIDDrawings.pdf 1961BIDDrawings.pdf
  python scripts/verify_realdata_roi_zoom.py --backfill-missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline.job import job_id_from_filename  # noqa: E402

REAL_DATA = Path(r"D:\Cost Estimation Using Text and Object\Real data")

# Representative set used in Real data/test.md (fast regression).
DEFAULT_PDFS = [
    "1961BIDDrawings test.pdf",
    "1948BIDDrawings.pdf",
    "1954BIDDrawings-AyersES011325.pdf",
    "1962BIDMHESDrawings.pdf",
    "1952BIDDrawings-Shadeland-Sunrise010625.pdf",
]


@dataclass
class PdfRow:
    pdf: str
    job_id: str
    job_exists: bool
    wings: int = 0
    roi_ok: int = 0
    verify_pass: bool = False
    issues: list[str] = field(default_factory=list)


def _discover_pdfs(real_dir: Path) -> list[Path]:
    pdfs = sorted(
        p
        for p in real_dir.glob("*.pdf")
        if p.is_file() and "requirement" not in p.name.lower()
    )
    return pdfs


def _count_roi(job_dir: Path) -> tuple[int, int]:
    """Return (wings_total, roi_ok_on_disk)."""
    meta_dir = job_dir / "05_metadata"
    if not meta_dir.is_dir():
        return 0, 0
    wings_total = 0
    roi_ok = 0
    for page_path in meta_dir.glob("page_*.json"):
        meta = json.loads(page_path.read_text(encoding="utf-8"))
        for wing in meta.get("wings") or []:
            wings_total += 1
            roi_rel = wing.get("roi_image")
            man_rel = wing.get("zooms_manifest")
            if not roi_rel or not man_rel:
                continue
            roi_path = job_dir / roi_rel
            man_path = job_dir / man_rel
            overlay_rel = wing.get("roi_overlay_image")
            full_rel = wing.get("zooms_full_image")
            overlay_path = job_dir / overlay_rel if overlay_rel else None
            full_path = job_dir / full_rel if full_rel else None
            if (
                roi_path.is_file()
                and man_path.is_file()
                and overlay_path
                and overlay_path.is_file()
                and full_path
                and full_path.is_file()
            ):
                roi_ok += 1
    return wings_total, roi_ok


def audit_rows(pdf_paths: list[Path], jobs_root: Path) -> list[PdfRow]:
    rows: list[PdfRow] = []
    for pdf_path in pdf_paths:
        job_id = job_id_from_filename(pdf_path.name)
        job_dir = jobs_root / job_id
        row = PdfRow(pdf=pdf_path.name, job_id=job_id, job_exists=job_dir.is_dir())
        if row.job_exists:
            row.wings, row.roi_ok = _count_roi(job_dir)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ROI+zoom for Real data PDFs")
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=REAL_DATA,
        help="Real data folder containing bid PDFs",
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        help="Specific PDF filenames (default: representative set from test.md)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check every PDF in --real-dir (not just the representative set)",
    )
    parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="Run backfill_wing_roi_zooms.py for jobs with wing crops but missing ROI",
    )
    args = parser.parse_args()

    real_dir = args.real_dir.resolve()
    if not real_dir.is_dir():
        print(f"ERROR: Real data folder not found: {real_dir}", file=sys.stderr)
        return 2

    if args.all:
        pdf_paths = _discover_pdfs(real_dir)
    elif args.pdfs:
        pdf_paths = [real_dir / name for name in args.pdfs]
        pdf_paths = [p for p in pdf_paths if p.is_file()]
    else:
        pdf_paths = [real_dir / name for name in DEFAULT_PDFS]
        pdf_paths = [p for p in pdf_paths if p.is_file()]

    if not pdf_paths:
        print("ERROR: no PDFs found to verify", file=sys.stderr)
        return 2

    jobs_root = config.STORAGE_DIR
    rows = audit_rows(pdf_paths, jobs_root)

    if args.backfill_missing:
        need_backfill = [
            r.job_id
            for r in rows
            if r.job_exists and r.wings > 0 and r.roi_ok < r.wings
        ]
        if need_backfill:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "backfill_wing_roi_zooms.py"),
                *need_backfill,
                "-v",
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=False)
            rows = audit_rows(pdf_paths, jobs_root)

    existing_jobs = [r.job_id for r in rows if r.job_exists]
    verify_code = 0
    verify_report_path = ROOT / "storage" / "verify_report.json"
    if existing_jobs:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "verify_output.py"),
            *existing_jobs,
        ]
        verify_code = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
        if verify_report_path.is_file():
            report = json.loads(verify_report_path.read_text(encoding="utf-8"))
            job_reports = report if isinstance(report, list) else report.get("jobs") or []
            for job_report in job_reports:
                jid = job_report.get("job_id")
                for row in rows:
                    if row.job_id == jid:
                        row.verify_pass = int(job_report.get("issue_count") or 0) == 0
                        row.issues = [
                            f"[p{i.get('page')}] {i.get('check')}: {i.get('detail')}"
                            for i in (job_report.get("issues") or [])[:5]
                        ]

    print("Real data ROI + zoom verification")
    print(f"  folder: {real_dir}")
    print(f"  jobs root: {jobs_root}")
    print()
    print(f"{'PDF':<45} {'Job':<35} {'Wings':>5} {'ROI':>5} {'Verify':>7}")
    print("-" * 100)
    missing_jobs = 0
    roi_gaps = 0
    for row in rows:
        if not row.job_exists:
            status = "NO JOB"
            missing_jobs += 1
        elif row.wings == 0:
            status = "no wings"
        elif row.roi_ok < row.wings:
            status = "ROI GAP"
            roi_gaps += 1
        elif row.verify_pass:
            status = "PASS"
        else:
            status = "FAIL"
        print(
            f"{row.pdf:<45} {row.job_id:<35} {row.wings:>5} {row.roi_ok:>5} {status:>7}"
        )
        for issue in row.issues[:3]:
            print(f"    {issue}")

    summary = {
        "real_dir": str(real_dir),
        "pdf_count": len(rows),
        "jobs_exist": sum(1 for r in rows if r.job_exists),
        "missing_jobs": missing_jobs,
        "roi_gaps": roi_gaps,
        "verify_pass": sum(1 for r in rows if r.verify_pass),
        "rows": [
            {
                "pdf": r.pdf,
                "job_id": r.job_id,
                "job_exists": r.job_exists,
                "wings": r.wings,
                "roi_ok": r.roi_ok,
                "verify_pass": r.verify_pass,
                "issues": r.issues,
            }
            for r in rows
        ],
    }
    out_path = ROOT / "storage" / "realdata_roi_zoom_verify.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out_path}")

    if missing_jobs or roi_gaps or verify_code != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
