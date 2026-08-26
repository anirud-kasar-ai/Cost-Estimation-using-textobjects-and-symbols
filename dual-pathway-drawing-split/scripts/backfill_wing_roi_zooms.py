"""Backfill ROI + overlapping zoom tiles for existing wing crops.

Use case:
Older job folders (generated before ROI/zoom were added) may have:
  - 04_wings/<wing>/page_XXX.jpg
but missing:
  - 04_wings/<wing>/rois/<page_key>/...
  - 04_wings/<wing>/zooms/<page_key>/...
and corresponding per-wing fields in 05_metadata/page_XXX.json.

This script fills the missing outputs without re-running the LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline.roi_zoom import export_wing_roi_and_zooms  # noqa: E402
from pipeline.wing_crop import slugify  # noqa: E402


logger = logging.getLogger("backfill_wing_roi_zooms")


def _safe_instance_slug(page_key: str) -> str:
    # Keep consistent with roi_zoom._safe_slug (but without importing private fn).
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (page_key or "").strip().replace(" ", "_"))
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned[:80] or "roi"


def backfill_one_job(job_root: Path) -> int:
    job_id = job_root.name
    meta_dir = job_root / "05_metadata"
    if not meta_dir.is_dir():
        logger.warning("[%s] missing 05_metadata/; skipping", job_id)
        return 0

    page_files = sorted(meta_dir.glob("page_*.json"))
    if not page_files:
        logger.warning("[%s] missing page_*.json; skipping", job_id)
        return 0

    updated_pages = 0
    for page_path in page_files:
        page_meta = json.loads(page_path.read_text(encoding="utf-8"))
        page_key = page_meta.get("page_key") or page_path.stem

        # Wings are stored inside page_meta["wings"].
        wings = page_meta.get("wings") or []
        if not wings:
            continue

        changed = False
        for wing in wings:
            # If already present, we skip to keep the backfill fast.
            if wing.get("roi_image") and wing.get("zooms_manifest"):
                continue

            wing_rel_img = wing.get("image")
            wing_name = wing.get("name") or "UNKNOWN"
            if not wing_rel_img:
                continue

            wing_img_path = job_root / Path(wing_rel_img)
            if not wing_img_path.is_file():
                logger.warning("[%s] missing wing image on disk: %s", job_id, wing_rel_img)
                continue

            wing_folder = wing_img_path.parent
            wing_rel_prefix = Path(wing_rel_img).parent.as_posix()  # e.g. 04_wings/B-WING-EAST

            with Image.open(wing_img_path).convert("RGB") as img:
                roi_payload = export_wing_roi_and_zooms(
                    wing_image=img,
                    wing_folder=wing_folder,
                    wing_rel_prefix=wing_rel_prefix,
                    instance_dir_name=page_key,
                    wing_name=wing_name,
                    tile_size=config.ROI_ZOOM_TILE_SIZE,
                    overlap_pct=config.ROI_ZOOM_OVERLAP_PCT,
                )

            wing.update(
                {
                    "roi_box_on_wing": roi_payload.get("roi_box_on_wing"),
                    "roi_image": roi_payload.get("roi_image"),
                    "roi_overlay_image": roi_payload.get("roi_overlay_image"),
                    "zooms_manifest": roi_payload.get("zooms_manifest"),
                    "zooms_full_image": roi_payload.get("zooms_full_image"),
                }
            )
            changed = True

        if changed:
            page_path.write_text(json.dumps(page_meta, indent=2), encoding="utf-8")
            updated_pages += 1

    logger.info("[%s] backfilled %d page metadata files", job_id, updated_pages)
    return updated_pages


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill wing ROI + zoom tiles for existing jobs")
    parser.add_argument(
        "jobs",
        nargs="*",
        help="Job folder names under storage/jobs/ (e.g. 1961BIDDrawings). If omitted, backfill all jobs missing roi fields.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backfill all jobs under storage/jobs/.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
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
        job_ids = sorted(p.name for p in jobs_root.iterdir() if p.is_dir() and (p / "job.json").exists())
    elif args.jobs:
        job_ids = args.jobs
    else:
        # Heuristic: backfill jobs that are missing roi_image in at least one metadata file.
        job_ids = []
        for p in jobs_root.iterdir():
            if not p.is_dir():
                continue
            if not (p / "job.json").exists():
                continue
            meta_dir = p / "05_metadata"
            if not meta_dir.is_dir():
                continue
            sample = next(meta_dir.glob("page_*.json"), None)
            if not sample:
                continue
            meta = json.loads(sample.read_text(encoding="utf-8"))
            wings = meta.get("wings") or []
            if not wings:
                continue
            if not any(w.get("roi_image") or w.get("zooms_manifest") for w in wings):
                job_ids.append(p.name)

    if not job_ids:
        logger.info("No jobs selected for backfill.")
        return 0

    total_pages = 0
    for job_id in job_ids:
        job_root = jobs_root / job_id
        if not job_root.exists():
            logger.warning("Missing job folder: %s", job_id)
            continue
        total_pages += backfill_one_job(job_root)

    logger.info("Backfill complete. Updated %d page metadata files.", total_pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

