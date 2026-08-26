"""Verify job output folders (page → diagram → wing).

Usage:
  python scripts/verify_output.py
  python scripts/verify_output.py 1948BIDDrawings 1962BIDMHESDrawings
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline.diagram_crop import ink_ratio, plan_mask  # noqa: E402
from pipeline.sheet_scan import _in_title_block  # noqa: E402


def _safe_instance_slug(page_key: str) -> str:
    """Match src/pipeline/roi_zoom.py::_safe_slug (local duplicate to avoid imports)."""
    import re

    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        (page_key or "").strip().replace(" ", "_"),
    )
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned[:80] or "roi"

MIN_PAGE_INK = 0.008
MIN_DIAGRAM_INK = 0.01
MIN_DIAGRAM_SIDE = 400
MIN_WING_INK = 0.005
MIN_WING_PLAN = 0.05
MIN_WING_SHORT = 150
MAX_WING_ASPECT = 50.0
MIN_DIAGRAM_AREA_FRAC = 0.08


@dataclass
class Issue:
    job: str
    page: str | int | None
    check: str
    detail: str


@dataclass
class JobReport:
    job_id: str
    pdf: str | None
    status: str
    kept_pages: int = 0
    checks: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def _plan_ratio(image: Image.Image) -> float:
    w, h = image.size
    scale = min(1.0, 1000.0 / max(w, h))
    small = (
        image
        if scale >= 1.0
        else image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    )
    return float((plan_mask(small) > 0).mean())


def _usable_wing(image: Image.Image) -> tuple[bool, str]:
    w, h = image.size
    short, long_ = min(w, h), max(w, h)
    if short < MIN_WING_SHORT:
        return False, f"short side {short}px < {MIN_WING_SHORT}"
    if long_ / max(1, short) > MAX_WING_ASPECT:
        return False, f"aspect {long_ / short:.1f} > {MAX_WING_ASPECT}"
    if ink_ratio(image) < MIN_WING_INK:
        return False, f"ink {ink_ratio(image):.4f} too low"
    plan = _plan_ratio(image)
    if plan < MIN_WING_PLAN:
        return False, f"plan linework {plan:.3f} < {MIN_WING_PLAN}"
    return True, "ok"


def _on_diagram_labels(meta: dict) -> list[str]:
    out: list[str] = []
    for point in meta.get("label_points") or []:
        name = point.get("name")
        pt = point.get("point")
        if not name or not pt:
            continue
        if _in_title_block([pt[0], pt[1], pt[0], pt[1]]):
            continue
        out.append(str(name))
    return out


def verify_job(job_dir: Path) -> JobReport:
    job_id = job_dir.name
    job_path = job_dir / "job.json"
    if not job_path.exists():
        return JobReport(
            job_id,
            None,
            "missing",
            issues=[Issue(job_id, None, "job.json", "missing")],
        )
    job = json.loads(job_path.read_text(encoding="utf-8"))
    report = JobReport(
        job_id,
        job.get("filename"),
        job.get("status") or "unknown",
    )
    if report.status != "done":
        report.issues.append(
            Issue(
                job_id,
                None,
                "job.status",
                f"status={report.status} error={job.get('error')}",
            )
        )
        return report

    source = list((job_dir / "01_source").glob("*"))
    report.checks += 1
    if not source:
        report.issues.append(Issue(job_id, None, "01_source", "empty"))

    pages = [p for p in job.get("pages") or [] if p.get("kept")]
    report.kept_pages = len(pages)

    for page in pages:
        page_key = page["page_key"]
        page_no = page.get("page")
        meta_path = job_dir / "05_metadata" / f"{page_key}.json"
        report.checks += 1
        if not meta_path.exists():
            report.issues.append(
                Issue(job_id, page_no, "05_metadata", f"{page_key}.json missing")
            )
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        page_rel = page.get("page_image") or meta.get("page_image")
        diagram_rel = page.get("diagram") or meta.get("diagram_image")
        if not page_rel:
            report.issues.append(Issue(job_id, page_no, "page_image", "not recorded"))
            continue
        page_path = job_dir / Path(page_rel)
        report.checks += 1
        if not page_path.is_file():
            report.issues.append(Issue(job_id, page_no, "02_pages", str(page_rel)))
            continue
        page_img: Image.Image | None = None
        try:
            page_img = Image.open(page_path).convert("RGB")
            if ink_ratio(page_img) < MIN_PAGE_INK:
                report.issues.append(
                    Issue(job_id, page_no, "02_pages.ink", f"blank page {page_rel}")
                )
        except OSError as exc:
            report.issues.append(Issue(job_id, page_no, "02_pages.read", str(exc)))

        if not diagram_rel:
            report.issues.append(Issue(job_id, page_no, "diagram_image", "not recorded"))
            continue
        diagram_path = job_dir / Path(diagram_rel)
        report.checks += 1
        if not diagram_path.is_file():
            report.issues.append(Issue(job_id, page_no, "03_drawings", str(diagram_rel)))
            continue
        try:
            diagram = Image.open(diagram_path).convert("RGB")
            dw, dh = diagram.size
            if min(dw, dh) < MIN_DIAGRAM_SIDE:
                report.issues.append(
                    Issue(job_id, page_no, "03_drawings.size", f"{dw}x{dh} too small")
                )
            if ink_ratio(diagram) < MIN_DIAGRAM_INK:
                report.issues.append(
                    Issue(job_id, page_no, "03_drawings.ink", "diagram looks blank")
                )
            cv_box = meta.get("cv_box_on_page") or {}
            if page_img and cv_box:
                page_w, page_h = page_img.size
                kept_frac = (
                    (cv_box.get("x2", 0) - cv_box.get("x1", 0))
                    * (cv_box.get("y2", 0) - cv_box.get("y1", 0))
                ) / float(page_w * page_h)
                if kept_frac < MIN_DIAGRAM_AREA_FRAC:
                    report.issues.append(
                        Issue(
                            job_id,
                            page_no,
                            "03_drawings.area",
                            f"diagram covers {kept_frac:.1%} of page",
                        )
                    )
        except OSError as exc:
            report.issues.append(Issue(job_id, page_no, "03_drawings.read", str(exc)))

        wing_map = meta.get("wing_map") or []
        report.checks += 1
        if not wing_map:
            report.issues.append(Issue(job_id, page_no, "wing_map", "empty"))

        labels = _on_diagram_labels(meta)
        wings = meta.get("wings") or []
        got = {w["name"] for w in wings}

        for name in labels:
            report.checks += 1
            if name not in got:
                report.issues.append(
                    Issue(job_id, page_no, "wing.missing", f"label {name} has no image")
                )

        for wing in wings:
            name = wing.get("name") or "?"
            img_rel = wing.get("image")
            report.checks += 1
            if not img_rel:
                report.issues.append(
                    Issue(job_id, page_no, "wing.image", f"{name} no path")
                )
                continue
            img_path = job_dir / Path(img_rel)
            if not img_path.is_file():
                report.issues.append(
                    Issue(job_id, page_no, "wing.file", f"{name} missing {img_rel}")
                )
                continue
            try:
                wing_img = Image.open(img_path).convert("RGB")
                ok, reason = _usable_wing(wing_img)
                report.checks += 1
                if not ok:
                    size = wing.get("size_pixels") or {}
                    report.issues.append(
                        Issue(
                            job_id,
                            page_no,
                            "wing.quality",
                            f"{name} {size.get('width')}x{size.get('height')}: {reason}",
                        )
                    )

                # ROI + zoom tiling outputs.
                # Newer jobs store explicit per-wing paths in metadata.
                # Older jobs may not — in that case, infer expected paths from
                # disk layout using this page's page_key.
                roi_image_rel = wing.get("roi_image")
                roi_overlay_rel = wing.get("roi_overlay_image")
                full_wing_rel = wing.get("zooms_full_image")
                manifest_rel = wing.get("zooms_manifest")

                instance_slug = _safe_instance_slug(str(page_key))
                wing_folder = img_path.parent

                if roi_image_rel and manifest_rel:
                    roi_png = job_dir / Path(roi_image_rel)
                    roi_overlay = (
                        job_dir / Path(roi_overlay_rel)
                        if roi_overlay_rel
                        else Path()
                    )
                    full_wing = (
                        job_dir / Path(full_wing_rel)
                        if full_wing_rel
                        else Path()
                    )
                    manifest_path = job_dir / Path(manifest_rel)
                    zooms_dir = manifest_path.parent
                else:
                    rois_dir = wing_folder / "rois" / instance_slug
                    zooms_dir = wing_folder / "zooms" / instance_slug
                    roi_png = rois_dir / "roi.png"
                    roi_overlay = rois_dir / "roi_overlay.png"
                    full_wing = zooms_dir / "full_wing.jpg"
                    manifest_path = zooms_dir / "zooms_manifest.json"

                report.checks += 4
                if not roi_png.is_file():
                    report.issues.append(
                        Issue(job_id, page_no, "wing.rois.roi.png", f"{name} missing")
                    )
                if not roi_overlay.is_file():
                    report.issues.append(
                        Issue(
                            job_id, page_no, "wing.rois.roi_overlay.png", f"{name} missing"
                        )
                    )
                if not full_wing.is_file():
                    report.issues.append(
                        Issue(job_id, page_no, "wing.zooms.full_wing.jpg", f"{name} missing")
                    )
                if not manifest_path.is_file():
                    report.issues.append(
                        Issue(job_id, page_no, "wing.zooms_manifest.json", f"{name} missing")
                    )
                else:
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        zooms_dir = manifest_path.parent
                        tiles = sorted(
                            p
                            for p in zooms_dir.glob("plan_zoom_r*_c*.jpg")
                            if p.suffix.lower() in {".jpg", ".jpeg"}
                        )
                        report.checks += 1
                        if int(manifest.get("count") or 0) != len(tiles):
                            report.issues.append(
                                Issue(
                                    job_id,
                                    page_no,
                                    "wing.zooms_manifest.count",
                                    f"{name} manifest count={manifest.get('count')} != tiles={len(tiles)}",
                                )
                            )
                    except Exception as exc:  # noqa: BLE001
                        report.issues.append(
                            Issue(job_id, page_no, "wing.zooms_manifest.read", f"{name}: {exc}")
                        )
            except OSError as exc:
                report.issues.append(
                    Issue(job_id, page_no, "wing.read", f"{name}: {exc}")
                )

        for name in meta.get("wings_without_image") or []:
            if name in labels:
                report.issues.append(
                    Issue(job_id, page_no, "wings_without_image", name)
                )

    summary_path = job_dir / "summary.json"
    report.checks += 1
    if not summary_path.is_file():
        report.issues.append(Issue(job_id, None, "summary.json", "missing"))
    return report


def main() -> int:
    jobs_root = config.STORAGE_DIR
    names = sys.argv[1:]
    if names:
        job_dirs = [jobs_root / n for n in names]
    else:
        job_dirs = sorted(
            p for p in jobs_root.iterdir() if p.is_dir() and (p / "job.json").exists()
        )

    reports: list[JobReport] = []
    all_issues: list[Issue] = []
    for job_dir in job_dirs:
        if not job_dir.exists():
            print(f"SKIP missing job folder {job_dir.name}")
            continue
        rep = verify_job(job_dir)
        reports.append(rep)
        all_issues.extend(rep.issues)
        mark = "PASS" if rep.ok else "FAIL"
        print(
            f"{mark}  {rep.job_id:40s}  kept={rep.kept_pages:3d}  "
            f"checks={rep.checks:4d}  issues={len(rep.issues)}"
        )
        for issue in rep.issues:
            where = f"p{issue.page}" if issue.page is not None else "-"
            print(f"      [{where}] {issue.check}: {issue.detail}")

    out = ROOT / "storage" / "verify_report.json"
    out.write_text(
        json.dumps(
            [
                {
                    "job_id": r.job_id,
                    "pdf": r.pdf,
                    "status": r.status,
                    "kept_pages": r.kept_pages,
                    "checks": r.checks,
                    "issue_count": len(r.issues),
                    "issues": [
                        {"page": i.page, "check": i.check, "detail": i.detail}
                        for i in r.issues
                    ],
                }
                for r in reports
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(reports)} jobs · {sum(r.checks for r in reports)} checks · {len(all_issues)} issues")
    print(f"wrote {out}")
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
