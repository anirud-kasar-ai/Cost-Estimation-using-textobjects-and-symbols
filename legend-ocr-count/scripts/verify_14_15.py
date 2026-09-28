from collections import Counter
from pathlib import Path

from pipeline.callout_ocr import detect_callouts
from pipeline.count_job import run_count_job
from pipeline.legend_ocr import load_image, parse_legend_file

pdf = Path(
    r"..\drawing-zoom-split\storage\technical_symbols"
    r"\1952BIDDrawings-Shadeland-Sunrise010625 technical symbol.pdf"
)
plan = Path("samples/shadeland_plan.png")

leg = parse_legend_file(pdf)
print("legend nums", [e.number for e in leg])
print("14", next(e.description for e in leg if e.number == 14))
print("15", next(e.description for e in leg if e.number == 15))

img = load_image(plan)
hits = detect_callouts(img, image_name=plan.name)
c = Counter(h.number for h in hits)
print("callout counts", dict(sorted(c.items())))
print("has 14", c.get(14, 0), "has 15", c.get(15, 0), "orphan 4", c.get(4, 0))

r = run_count_job(pdf, plan, Path("storage/jobs/_verify_14_15"))
rows = {row["number"]: row for row in r["rows"]}
for n in (2, 6, 7, 14, 15):
    row = rows.get(n)
    if row:
        print(
            f"row {n}: count={row['count']} callout={row['callout_count']} "
            f"desc={row['description'][:40]}"
        )
    else:
        print(f"row {n}: MISSING")
print("unknown", r["unknown_numbers"])
