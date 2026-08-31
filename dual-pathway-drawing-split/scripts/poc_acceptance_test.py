"""POC acceptance test for drawing-zoom-split (1948BIDDrawings).

Runs automated QA across pipeline stages — no manual checks required.
Exit 0 = POC PASS, 1 = POC FAIL.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

import app as fastapi_app  # noqa: E402
from pipeline import config  # noqa: E402

JOB_ID = "1948BIDDrawings"
REAL_PDF = ROOT.parent / "Real data" / "1948BIDDrawings.pdf"


@dataclass
class Check:
    stage: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def ok(self, stage: str, name: str, cond: bool, detail: str = "") -> None:
        self.checks.append(Check(stage, name, cond, detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print_summary(self, job_id: str = JOB_ID) -> None:
        by_stage: dict[str, list[Check]] = {}
        for c in self.checks:
            by_stage.setdefault(c.stage, []).append(c)
        print("\n=== POC ACCEPTANCE TEST ===")
        print(f"Job: {job_id}")
        print(f"PDF: {REAL_PDF}\n")
        for stage, items in by_stage.items():
            mark = "PASS" if all(i.passed for i in items) else "FAIL"
            print(f"[{mark}] {stage}")
            for item in items:
                icon = "  ok" if item.passed else "  FAIL"
                line = f"{icon}  {item.name}"
                if item.detail:
                    line += f" — {item.detail}"
                print(line)
        total = len(self.checks)
        fails = sum(1 for c in self.checks if not c.passed)
        print(f"\n{total - fails}/{total} checks passed")
        print("POC RESULT:", "PASS" if self.passed else "FAIL")


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_acceptance(job_id: str = JOB_ID) -> Report:
    report = Report()
    job_root = config.STORAGE_DIR / job_id
    job_path = job_root / "job.json"

    # --- Stage 0: inputs ---
    report.ok(
        "0-inputs",
        "Real data test PDF exists",
        REAL_PDF.is_file(),
        str(REAL_PDF),
    )
    report.ok("0-inputs", "Job folder exists", job_root.is_dir(), str(job_root))
    report.ok("0-inputs", "job.json exists", job_path.is_file())

    job = _load_json(job_path)
    report.ok("0-inputs", "job.json parses", isinstance(job, dict))

    if not isinstance(job, dict):
        return report

    source_pdfs = list((job_root / "01_source").glob("*.pdf"))
    report.ok(
        "0-inputs",
        "Source PDF in job",
        len(source_pdfs) == 1,
        source_pdfs[0].name if source_pdfs else "missing",
    )

    # --- Stage 1: extraction ---
    report.ok(
        "1-extraction",
        "symbol_table.json present",
        (job_root / "symbol_table.json").is_file(),
    )
    legend = _load_json(job_root / "symbol_table.json")
    entry_count = int(legend.get("entry_count", 0)) if isinstance(legend, dict) else 0
    report.ok(
        "1-extraction",
        "Legend has entries",
        entry_count >= 10,
        f"{entry_count} entries",
    )
    report.ok(
        "1-extraction",
        "Requirement PDF exported",
        bool(job.get("has_requirement_pdf"))
        or (job_root / "storage").exists()
        or any((ROOT / "storage" / "requirements").glob(f"*{job_id}*")),
    )
    req_files = list((ROOT / "storage" / "requirements").glob(f"*{job_id}*"))
    report.ok(
        "1-extraction",
        "Requirement PDF on disk",
        len(req_files) >= 1,
        req_files[0].name if req_files else "none",
    )
    sym_files = list((ROOT / "storage" / "technical_symbols").glob(f"*{job_id}*"))
    report.ok(
        "1-extraction",
        "Technical symbol PDF on disk",
        len(sym_files) >= 1,
        sym_files[0].name if sym_files else "none",
    )
    report.ok(
        "1-extraction",
        "Sheet notes PDF flag or file",
        bool(job.get("has_sheet_notes_pdf"))
        or len(list((ROOT / "storage" / "sheet_notes").glob(f"*{job_id}*"))) >= 1
        if (ROOT / "storage" / "sheet_notes").exists()
        else bool(job.get("has_sheet_notes_pdf")),
    )

    # --- Stage 2: page render + classify ---
    report.ok("2-pages", "Job status done", job.get("status") == "done", job.get("status", ""))
    pages = job.get("pages") or []
    kept = [p for p in pages if p.get("kept")]
    report.ok("2-pages", "Kept plan pages", len(kept) >= 14, f"{len(kept)} kept")
    t200_pages = [
        p for p in kept if str(p.get("sheet_no", "")).startswith("T-2")
    ]
    report.ok(
        "2-pages",
        "T-200 floor plan sheets kept",
        len(t200_pages) >= 14,
        f"{len(t200_pages)} T-2xx sheets",
    )
    missing_page_img = [
        p.get("sheet_no")
        for p in kept
        if not p.get("page_image") or not (job_root / str(p.get("page_image"))).is_file()
    ]
    report.ok(
        "2-pages",
        "Kept pages have rendered JPEG",
        not missing_page_img,
        f"missing: {missing_page_img[:5]}" if missing_page_img else "all ok",
    )

    # --- Stage 3: diagram + wing crop ---
    report.ok(
        "3-wings",
        "Drawing count recorded",
        int(job.get("drawing_count") or 0) >= 14,
        str(job.get("drawing_count")),
    )
    report.ok(
        "3-wings",
        "Wing images produced",
        int(job.get("wing_count") or 0) >= 50,
        str(job.get("wing_count")),
    )
    wing_images = list((job_root / "04_wings").rglob("*.jpg"))
    report.ok(
        "3-wings",
        "Wing JPEG files on disk",
        len(wing_images) >= 50,
        f"{len(wing_images)} files",
    )
    wings_with_roi = 0
    for page in kept:
        for wing in page.get("wings") or []:
            if wing.get("roi_image") and (job_root / str(wing["roi_image"])).is_file():
                wings_with_roi += 1
    report.ok(
        "3-wings",
        "Wing ROI artifacts present",
        wings_with_roi >= 40,
        f"{wings_with_roi} wings with ROI",
    )

    # --- Stage 4: zoom tiles ---
    zoom_manifests = list((job_root / "04_wings").rglob("zooms_manifest.json"))
    report.ok(
        "4-zooms",
        "Zoom manifests present",
        len(zoom_manifests) >= 10,
        f"{len(zoom_manifests)} manifests",
    )
    plan_tiles = list((job_root / "04_wings").rglob("plan_zoom_*.jpg"))
    report.ok(
        "4-zooms",
        "Plan zoom tiles present",
        len(plan_tiles) >= 20,
        f"{len(plan_tiles)} tiles",
    )

    # --- Stage 5: API ---
    client = TestClient(fastapi_app.app)
    health = client.get("/api/health")
    report.ok(
        "5-api",
        "GET /api/health",
        health.status_code == 200,
        health.json().get("status", "") if health.status_code == 200 else str(health.status_code),
    )
    job_resp = client.get(f"/api/jobs/{job_id}")
    report.ok(
        "5-api",
        "GET /api/jobs/{id}",
        job_resp.status_code == 200 and job_resp.json().get("status") == "done",
    )
    if sym_files:
        tech = client.get(f"/api/jobs/{job_id}/technical-symbol.pdf")
        report.ok(
            "5-api",
            "GET technical-symbol.pdf",
            tech.status_code == 200 and tech.content[:4] == b"%PDF",
        )

    return report


def main() -> int:
    job_id = sys.argv[1] if len(sys.argv) > 1 else JOB_ID
    report = run_acceptance(job_id)
    report.print_summary(job_id)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
