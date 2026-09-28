"""Smoke test YOLO + OCR pipeline."""
from __future__ import annotations

from pathlib import Path

from pipeline.count_job import run_count_job
from pipeline.yolo_detect import class_names, yolo_available


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("yolo_available", yolo_available())
    print("classes", list(class_names().values())[:5], "...")
    r = run_count_job(
        root / "samples" / "symbol_legend.png",
        root / "samples" / "roof_plan.png",
        root / "storage" / "jobs" / "_yolo_smoke",
    )
    print("method", r["method"], "yolo", r["total_yolo"], "callouts", r["total_callouts"])
    print("rows>", [(x["number"], x["count"], x["method"], x["yolo_count"], x["callout_count"]) for x in r["rows"] if x["count"] or x["yolo_count"]])
    print("yolo_unmatched", r["yolo_unmatched"][:5])


if __name__ == "__main__":
    main()
