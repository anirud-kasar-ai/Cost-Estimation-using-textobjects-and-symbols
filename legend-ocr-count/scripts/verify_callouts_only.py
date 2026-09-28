from collections import Counter
from pathlib import Path

from pipeline.callout_ocr import detect_callouts
from pipeline.legend_ocr import load_image, parse_legend_file

pdf = Path(
    r"..\drawing-zoom-split\storage\technical_symbols"
    r"\1952BIDDrawings-Shadeland-Sunrise010625 technical symbol.pdf"
)
plan = Path("samples/shadeland_plan.png")

leg = parse_legend_file(pdf)
print("legend has 14/15", {e.number for e in leg} >= {14, 15})

img = load_image(plan)
hits = detect_callouts(img, image_name=plan.name)
c = Counter(h.number for h in hits)
print("callout counts", dict(sorted(c.items())))
print("14=", c.get(14, 0), "15=", c.get(15, 0), "4=", c.get(4, 0))
for h in hits:
    if h.number in {4, 14, 15}:
        print(
            f"  #{h.number} conf={h.confidence:.0f} "
            f"box=({h.x1:.0f},{h.y1:.0f})-({h.x2:.0f},{h.y2:.0f}) src={h.source}"
        )
