"""Symbol-extraction-only test (no UI upload, no wing/zoom pipeline).

Usage:
  cd drawing-zoom-split
  .\\.venv\\Scripts\\python.exe scripts\\test_symbol_extract.py
  .\\.venv\\Scripts\\python.exe scripts\\test_symbol_extract.py --pdf path\\to\\file.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.extraction.stages import _save_glyph_templates
from pipeline.extraction.symbol_table_extractor import extract_symbol_table_from_pdf
from pipeline.extraction.symbol_table_pdf import (
    generate_technical_symbol_pdf,
    technical_symbol_pdf_filename,
)


DEFAULT_PDF = (
    ROOT
    / "storage"
    / "jobs"
    / "1951BIDDrawings-NorthgateHS"
    / "01_source"
    / "1951BIDDrawings-NorthgateHS.pdf"
)
OUT_DIR = ROOT / "storage" / "test_symbol_extract"


def main() -> int:
    parser = argparse.ArgumentParser(description="Symbol extraction only test")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Source bid PDF (default: 1951 job source)",
    )
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    if not pdf.is_file():
        print(f"ERROR: PDF not found: {pdf}")
        return 1

    print(f"Symbol extract only: {pdf.name}")
    info = extract_symbol_table_from_pdf(pdf)
    abbrev = sum(
        1 for e in info.entries if "ABBREV" in (e.legend_title or "").upper()
    )
    glyphs = sum(1 for e in info.entries if e.symbol_image_png)
    titles: dict[str, int] = {}
    for entry in info.entries:
        title = entry.legend_title or "?"
        titles[title] = titles.get(title, 0) + 1

    print(f"  entries={info.entry_count}")
    print(f"  glyphs={glyphs}")
    print(f"  abbreviations={abbrev}")
    print(f"  legend_pages={info.legend_pages}")
    for title, count in sorted(titles.items(), key=lambda item: -item[1]):
        print(f"  - {count:3d}  {title}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    table_path = OUT_DIR / "symbol_table.json"
    table_path.write_text(
        json.dumps(info.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    glyph_count = _save_glyph_templates(info, OUT_DIR / "07_glyphs")
    pdf_name = technical_symbol_pdf_filename(pdf.name)
    symbol_pdf = OUT_DIR / pdf_name
    generate_technical_symbol_pdf(info, pdf.name, symbol_pdf)

    # Also publish beside official technical_symbols for easy open.
    pub = ROOT / "storage" / "technical_symbols" / f"{pdf.stem} technical symbol TEST.pdf"
    shutil.copy2(symbol_pdf, pub)

    print(f"  wrote {table_path}")
    print(f"  wrote {glyph_count} glyphs -> {OUT_DIR / '07_glyphs'}")
    print(f"  wrote {symbol_pdf}")
    print(f"  wrote {pub}")
    print("DONE")
    return 0 if info.entry_count > 0 and abbrev == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
