"""Backfill: run CV/text-based symbol counting on an existing job."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.symbol_count_cv import run_symbol_count_cv

job_id = sys.argv[1] if len(sys.argv) > 1 else "1948BIDDrawings"
job_root = Path(__file__).resolve().parent.parent / "storage" / "jobs" / job_id

t0 = time.time()
result = run_symbol_count_cv(
    job_root,
    on_progress=lambda s, m: print(f"[{s}] {m}", flush=True),
)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"Status: {result['status']}")
print(f"Instances: {result.get('instances', 0)}")

job_path = job_root / "job.json"
job = json.loads(job_path.read_text(encoding="utf-8"))
job.update(
    {
        "symbol_count_status": result.get("status"),
        "symbol_count_instances": int(result.get("instances") or 0),
        "symbol_count_detail_path": result.get("symbol_count_detail_path"),
        "symbol_count_report_csv": result.get("symbol_count_report_csv"),
        "symbol_count_report_pdf": result.get("symbol_count_report_pdf"),
        "symbol_count_report_json": result.get("symbol_count_report_json"),
    }
)
job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
print("job.json updated")
