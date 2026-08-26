"""Score legend glyphs at known detection boxes."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline.cv.nms import BoundingBox
from pipeline.cv.template_match import score_glyph_on_crop
from pipeline.merge_evaluator import _crop_detection, _glyph_image_for_symbol
from pipeline.symbol_legend import parse_symbol_file

job_dir = Path("storage/jobs/1948BIDDrawings_technical_symbol__15___plan_zoom_r00_c00_457a8cb6")
img = Image.open(
    r"C:\Users\akasar\.cursor\projects\d-Cost-Estimation-Using-Text-and-Object\assets"
    r"\c__Users_akasar_AppData_Roaming_Cursor_User_workspaceStorage_360a447a511e5e52fc4ce6428cf738c9_images_image-60790d75-d9ec-4b0f-ae8c-ad23eb721589.png"
).convert("RGB")
legend, glyph_dir = parse_symbol_file(job_dir / "symbol_file.pdf", job_dir)

boxes = {
    "WP triangle area": BoundingBox(117, 210, 163, 267),
    "AP circle (hourglass false pos)": BoundingBox(643, 98, 665, 125),
    "Interior speaker triangle": BoundingBox(541, 329, 577, 369),
}

for label, symbols in [
    ("drop marks", ["#"]),
    ("cameras", ["NETWORK CAMERA", "CAT6A DATA DROP LOCATION (QTY = 1) - NETWORK CAMERA"]),
    ("AP/WP", ["AP", "WP"]),
]:
    print(f"\n=== {label} ===")
    for name, box in boxes.items():
        crop = _crop_detection(img, box, pad_px=40)
        scores = []
        for sym in symbols:
            g = _glyph_image_for_symbol(legend, glyph_dir, sym)
            if g:
                scores.append((sym, score_glyph_on_crop(crop, g)))
        best = max(scores, key=lambda x: x[1]) if scores else ("?", 0.0)
        print(f"  {name:35} best={best[0][:40]:40} score={best[1]:.3f}")
