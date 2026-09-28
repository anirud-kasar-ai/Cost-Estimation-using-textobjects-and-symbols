"""Combine before/after-retrain overall business metrics into one CSV.

Before = docs/business_metrics_rescue_tuned.csv (old roof model, pre-retrain)
After  = docs/business_metrics_rescue.csv       (retrained roof model + TTA fix)

Writes docs/business_metrics_overall.csv with the 'Overall (both models)' rows
from both files, labeled so the improvement is visible side by side.

Usage (from repo root):  python scripts/business_metrics_overall.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

SOURCES = [
    (DOCS / "business_metrics_rescue_tuned.csv", "BEFORE retrain (old roof model)"),
    (DOCS / "business_metrics_rescue.csv", "AFTER retrain (new roof model + TTA fix)"),
]
OUT = DOCS / "business_metrics_overall.csv"


def main() -> None:
    header: list[str] | None = None
    rows: list[list[str]] = []
    for path, label in SOURCES:
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            head = next(reader)
            if header is None:
                header = ["Stage"] + head
            for row in reader:
                if row and row[0].startswith("Overall"):
                    rows.append([label] + row)

    assert header is not None
    try:
        with OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"Saved: {OUT}")
    except PermissionError:  # file open in Excel
        alt = OUT.with_name(OUT.stem + "_new.csv")
        with alt.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"{OUT} was locked (Excel?); saved to {alt}")

    for r in rows:
        print(
            f"{r[0]:42} {r[3]:38} recall {r[7]}  precision {r[13]}  "
            f"f1 {r[14]}  missed {r[9]}  fp {r[12]}"
        )


if __name__ == "__main__":
    main()
