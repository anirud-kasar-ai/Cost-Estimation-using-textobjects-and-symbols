"""Regenerate 1951 technical-symbol extract only (for crop review)."""
from __future__ import annotations

import io
import json
import re
import shutil
from pathlib import Path

from PIL import Image

from pipeline.extraction.stages import _save_glyph_templates
from pipeline.extraction.symbol_table_extractor import extract_symbol_table_from_pdf
from pipeline.extraction.symbol_table_pdf import generate_technical_symbol_pdf

ROOT = Path(__file__).resolve().parents[1]
NAME = "1951BIDDrawings-NorthgateHS"


def main() -> None:
    pdf = ROOT / "storage" / "jobs" / NAME / "01_source" / f"{NAME}.pdf"
    print("Extracting ONLY", NAME)
    info = extract_symbol_table_from_pdf(pdf)
    abbrev = sum(
        1 for e in info.entries if "ABBREV" in (e.legend_title or "").upper()
    )
    glyphs = sum(1 for e in info.entries if e.symbol_image_png)
    print(f"entries={info.entry_count} glyphs={glyphs} abbrev={abbrev}")

    out = ROOT / "storage" / "_debug_symbol_pdf" / "1951_review"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    keys = ("MOUNT", "WIRELESS", "ACCESS", "DATA LOCATION", "CAT6A DATA DROP")
    for i, entry in enumerate(info.entries):
        desc = entry.description or ""
        if not any(k in desc.upper() for k in keys):
            continue
        print(f"{i:02d}", entry.symbol or "-", "|", desc[:55], "|", len(entry.symbol_image_png or b""))
        if not entry.symbol_image_png:
            continue
        safe = re.sub(r"[^\w\-]+", "_", desc[:40]).strip("_")
        Image.open(io.BytesIO(entry.symbol_image_png)).save(out / f"{i:02d}_{safe}.png")

    job = ROOT / "storage" / "jobs" / NAME
    (job / "symbol_table.json").write_text(
        json.dumps(info.to_json_dict(), indent=2), encoding="utf-8"
    )
    gdir = job / "07_glyphs"
    if gdir.exists():
        shutil.rmtree(gdir)
    _save_glyph_templates(info, gdir)

    cache = ROOT / "storage" / "extraction_cache" / NAME / "symbol_table.json"
    if cache.parent.exists():
        shutil.copy2(job / "symbol_table.json", cache)

    out_pdf = (
        ROOT
        / "storage"
        / "technical_symbols"
        / f"{NAME} technical symbol REVIEW.pdf"
    )
    generate_technical_symbol_pdf(info, f"{NAME}.pdf", out_pdf)
    print("REVIEW PDF:", out_pdf.resolve())


if __name__ == "__main__":
    main()
