"""Build a persistent glyph + description reference from a technical-symbol PDF.

Parses the legend, saves each glyph crop as PNG, asks Gemini once for a
drafter-style shape description of every glyph, and writes everything to
storage/reference_legends/<name>/ (glyphs/*.png + descriptions.json).

Usage:
    python scripts/build_reference_legend.py "<path to symbol pdf>" [name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.legend_reference import build_reference_legend  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pdf = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else None
    ref_dir, descriptions = build_reference_legend(pdf, name=name)
    print(f"Reference legend written to: {ref_dir}")
    print(json.dumps(descriptions, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
