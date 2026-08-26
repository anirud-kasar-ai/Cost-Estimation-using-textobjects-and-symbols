"""One-off diagnostic: run detection + evaluator on a plan tile image."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from pipeline.cv.nms import BoundingBox, box_iou
from pipeline.cv.template_match import detect_symbols_from_glyphs
from pipeline.merge_evaluator import evaluate_merged_detections
from pipeline.symbol_count import detect_symbols_raw
from pipeline.symbol_legend import parse_symbol_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    job_dir = root / "storage/jobs/1948BIDDrawings_technical_symbol__15___plan_zoom_r00_c00_457a8cb6"
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"C:\Users\akasar\.cursor\projects\d-Cost-Estimation-Using-Text-and-Object\assets"
        r"\c__Users_akasar_AppData_Roaming_Cursor_User_workspaceStorage_360a447a511e5e52fc4ce6428cf738c9_images_image-60790d75-d9ec-4b0f-ae8c-ad23eb721589.png"
    )
    if not img_path.exists():
        print("Image missing:", img_path)
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    print("Input size:", img.size)

    legend, glyph_dir = parse_symbol_file(job_dir / "symbol_file.pdf", job_dir)
    print("Glyphs dir:", glyph_dir)
    if glyph_dir:
        names = sorted(p.name for p in glyph_dir.glob("*.png"))
        print(f"Glyph count: {len(names)}")
        print("Sample:", names[:12])

    kept, notes, meta = detect_symbols_raw(
        image=img, legend=legend, glyph_dir=glyph_dir, symbol_file_suffix=".pdf"
    )
    print("\n=== DETECTIONS (pipeline output) ===")
    for d in kept:
        print(
            f"  {d.symbol:22} src={d.source:10} score={d.score:.2f} qty={d.qty} "
            f"box=({d.box.x1:.0f},{d.box.y1:.0f},{d.box.x2:.0f},{d.box.y2:.0f})"
        )

    print("\n=== HOURGLASS vs TRIANGLE IoU (current run) ===")
    tris = [d for d in kept if d.source == "triangle"]
    cams = [d for d in kept if d.source == "hourglass"]
    for cam in cams:
        best_iou = max(box_iou(cam.box, t.box) for t in tris) if tris else 0.0
        cx = (cam.box.x1 + cam.box.x2) / 2
        cy = (cam.box.y1 + cam.box.y2) / 2
        print(f"  camera center=({cx:.0f},{cy:.0f}) best triangle IoU: {best_iou:.3f}")

    if glyph_dir:
        tmpl = detect_symbols_from_glyphs(img, legend, glyph_dir)
        print("\n=== TEMPLATE MATCH ONLY ===")
        print("  hits:", len(tmpl))
        for d in sorted(tmpl, key=lambda x: -x.score)[:15]:
            print(f"  {d.symbol:22} score={d.score:.3f}")

    accepted, summary = evaluate_merged_detections(
        roi_image=img, legend=legend, glyph_dir=glyph_dir, detections=kept
    )
    print("\n=== MERGE EVALUATOR ===")
    print("summary:", {k: v for k, v in summary.items() if k != "rejected_by_reason"})
    for ev in accepted:
        reason = ev.evaluation.get("reason", "")
        if ev.verdict != "accepted" or reason.startswith("geometry"):
            print(
                f"  {ev.detection.symbol:22} {ev.verdict:8} "
                f"glyph={ev.glyph_score} reason={reason}"
            )


if __name__ == "__main__":
    main()
