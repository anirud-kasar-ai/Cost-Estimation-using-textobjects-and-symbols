"""Quick single-image count (no LLM)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["SYMBOL_COUNT_USE_LLM"] = "false"

from PIL import Image

from pipeline.symbol_count import count_symbols_in_image_cv
from pipeline.symbol_legend import parse_symbol_file


def main() -> None:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"C:\Users\akasar\.cursor\projects\d-Cost-Estimation-Using-Text-and-Object\assets"
        r"\c__Users_akasar_AppData_Roaming_Cursor_User_workspaceStorage_360a447a511e5e52fc4ce6428cf738c9_images_image-8866bb4d-4ea3-4a6b-8445-3b0850823904.png"
    )
    job = Path("storage/jobs/1948BIDDrawings_technical_symbol__15___plan_zoom_r00_c00_457a8cb6")
    img = Image.open(img_path).convert("RGB")
    leg, gd = parse_symbol_file(job / "symbol_file.pdf", job)
    p = count_symbols_in_image_cv(image=img, legend=leg, glyph_dir=gd, symbol_file_suffix=".pdf")
    print("counts:", {k: v for k, v in p["counts"].items() if v})
    print("detections:", len(p["detections"]), "rejected:", p.get("rejected_detection_count", 0))
    for d in p["detections"]:
        sym = str(d["symbol"])[:25]
        print(f"  {sym:25} {d['source']:10} qty={d.get('qty', 1)}")


if __name__ == "__main__":
    main()
