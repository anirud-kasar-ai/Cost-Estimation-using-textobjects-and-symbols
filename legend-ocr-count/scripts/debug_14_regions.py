"""Fast ROI check: OCR crops around known callout regions on the Shadeland plan."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.callout_ocr import _gray, _hit_from_circle, _ocr_digits_in_crop, detect_hough_callouts
from pipeline.legend_ocr import load_image

plan = Path("samples/shadeland_plan.png")
img = load_image(plan)
gray = _gray(img)
h, w = gray.shape
print("plan", w, h)

# Approximate regions from the screenshot description (normalized coords)
# 14 bottom-center near W, 14 right side, 15 bottom-left, 15 right side
regions = {
    "14_right": (0.82, 0.55, 0.98, 0.78),
    "14_bottom": (0.35, 0.70, 0.55, 0.92),
    "15_right": (0.82, 0.70, 0.98, 0.90),
    "15_left": (0.05, 0.70, 0.25, 0.92),
}

for name, (x0, y0, x1, y1) in regions.items():
    x0i, y0i = int(x0 * w), int(y0 * h)
    x1i, y1i = int(x1 * w), int(y1 * h)
    crop_img = img.crop((x0i, y0i, x1i, y1i))
    hits = detect_hough_callouts(crop_img)
    print(name, "hits", [(hh.number, round(hh.confidence), round(hh.cx), round(hh.cy)) for hh in hits])

# Full-image but only report 4/14/15 (still may be slow)
print("scanning full image hough (may take ~1 min)...")
hits = detect_hough_callouts(img)
from collections import Counter

c = Counter(h.number for h in hits)
print("counts", dict(sorted(c.items())))
print("14", c.get(14, 0), "15", c.get(15, 0), "4", c.get(4, 0))
