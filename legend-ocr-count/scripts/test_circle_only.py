"""Quick checks: example crops + roof plan circled-only detection."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline.callout_ocr import detect_callouts, is_hollow_callout_circle, _gray
from pipeline.count_job import run_count_job


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ex = root / "samples" / "callout_examples"
    for p in sorted(ex.glob("*.png")):
        img = Image.open(p).convert("RGB")
        gray = _gray(img)
        h, w = gray.shape[:2]
        ok = is_hollow_callout_circle(gray, w / 2.0, h / 2.0, min(h, w) * 0.32)
        hits = detect_callouts(img, image_name=p.name)
        print(f"{p.name}: ring_ok={ok} hits={[h.number for h in hits]}")

    r = run_count_job(
        root / "samples" / "symbol_legend.png",
        root / "samples" / "roof_plan.png",
        root / "storage" / "jobs" / "_circle_only",
    )
    print(
        "plan:",
        [(row["number"], row["count"]) for row in r["rows"] if row["count"]],
        "total",
        r["total_callouts"],
        "sources",
        sorted({d["source"] for d in r["detections"]}),
    )


if __name__ == "__main__":
    main()
