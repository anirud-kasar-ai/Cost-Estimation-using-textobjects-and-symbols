"""Label a crop PNG: click two corners per instance, type the legend key.

Usage:
    python eval/label.py eval/crops/01_hash_standalone.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    img = Image.open(path).convert("RGB")
    print(f"Image {path.name} {img.size}. Enter instances as: key x1 y1 x2 y2")
    print("Empty line to finish.")
    instances = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 5:
            print("need: key x1 y1 x2 y2")
            continue
        key, x1, y1, x2, y2 = parts
        instances.append(
            {
                "key": key,
                "qty": 1,
                "box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
            }
        )
    rec = {
        "id": path.stem,
        "image": f"crops/{path.name}",
        "width": img.size[0],
        "height": img.size[1],
        "instances": instances,
    }
    out = ROOT / "gt" / f"{path.stem}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
