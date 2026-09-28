import time
from pathlib import Path

from pipeline.count_job import run_count_job
from pipeline.legend_ocr import parse_legend_file

t0 = time.time()
leg = parse_legend_file(Path("samples/symbol_legend.png"))
print("LEGEND", len(leg), "nums", [e.number for e in leg], f"{time.time() - t0:.1f}s")
t1 = time.time()
r = run_count_job(
    Path("samples/symbol_legend.png"),
    Path("samples/roof_plan.png"),
    Path("storage/jobs/_smoke3"),
)
print(
    f"DONE {time.time() - t1:.1f}s total={r['total_callouts']} dets={len(r['detections'])}"
)
print("hits", [(x["number"], x["count"]) for x in r["rows"] if x["count"]])
