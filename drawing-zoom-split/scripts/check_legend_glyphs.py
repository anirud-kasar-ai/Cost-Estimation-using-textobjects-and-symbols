"""Report which pricing keys resolve to a legend glyph (and which fall back)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app  # noqa: E402
from pipeline.pricing import _canon_key, load_price_index  # noqa: E402

missing = []
for key in sorted(load_price_index()):
    glyph = app._legend_glyph_sample(key, _canon_key)
    if glyph is None:
        missing.append(key)
    else:
        print(f"LEGEND  {key}  <-  {glyph.parent.parent.name}/{glyph.name}")
print(f"\nNo legend glyph ({len(missing)}):")
for key in missing:
    print(f"  {key}")
