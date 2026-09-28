"""Audit: which legend rows across Real data PDFs fail to match the class map.

For every PDF in "Real data": load its symbol legend (from the job folder if
already extracted, otherwise extract now), then check each legend row against
every YOLO class in the annotation map using the real matcher. Rows no class
can match are reported — candidates for new phrase variants in yolo_class_map.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import config  # noqa: E402
from pipeline.count_stage import legend_from_symbol_table  # noqa: E402
from pipeline.job import job_id_from_filename  # noqa: E402
from pipeline.legend_match import match_class_to_legend  # noqa: E402
from pipeline.yolo_class_map import mapped_class_names  # noqa: E402

REAL_DATA = ROOT.parent / "Real data"


def legend_json_for(pdf: Path) -> Path | None:
    job_id = job_id_from_filename(pdf.name)
    candidates = [
        config.STORAGE_DIR / job_id / "symbol_table.json",
        config.EXTRACTION_CACHE_DIR / job_id / "symbol_table.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def main() -> None:
    classes = mapped_class_names()
    extracted_dir = ROOT / "storage" / "_legend_audit"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for pdf in sorted(REAL_DATA.glob("*.pdf")):
        path = legend_json_for(pdf)
        source = "cached"
        if path is None:
            # Extract legend now (text-based, no LLM)
            from pipeline.extraction.symbol_table_extractor import (
                extract_symbol_table_from_pdf,
            )

            try:
                info = extract_symbol_table_from_pdf(pdf)
            except Exception as exc:  # noqa: BLE001
                print(f"=== {pdf.name}: EXTRACTION FAILED: {exc}")
                continue
            path = extracted_dir / f"{job_id_from_filename(pdf.name)}.json"
            path.write_text(
                json.dumps(info.to_json_dict(), indent=2), encoding="utf-8"
            )
            source = "extracted now"

        legend = legend_from_symbol_table(path)
        if not legend:
            print(f"=== {pdf.name} ({source}): NO LEGEND ROWS")
            continue

        matched_numbers: set[int] = set()
        for cls in classes:
            m = match_class_to_legend(cls, legend, min_score=config.YOLO_MATCH_MIN_SCORE)
            if m is not None:
                matched_numbers.add(m.legend_number)
        # A class matches only its single best row; also collect any row that at
        # least one class could match at all (per-row check).
        row_matched: dict[int, list[str]] = {}
        for entry in legend:
            for cls in classes:
                m = match_class_to_legend(
                    cls, [entry], min_score=config.YOLO_MATCH_MIN_SCORE
                )
                if m is not None:
                    row_matched.setdefault(entry.number, []).append(cls)

        print(f"=== {pdf.name} ({source}): {len(legend)} legend rows")
        for entry in legend:
            who = row_matched.get(entry.number)
            tail = f"detail={entry.detail_ref}" if entry.detail_ref else ""
            if who:
                short = ", ".join(sorted(set(who))[:3])
                print(f"    OK      {entry.description}  {tail}  <- {short}")
            else:
                print(f"    UNMATCHED  {entry.description}  {tail}")
        print()


if __name__ == "__main__":
    main()
