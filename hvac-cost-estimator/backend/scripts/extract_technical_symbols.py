"""Optional batch helper for technical symbol PDFs (dev / debugging).

Production path: technical symbol PDFs are created on upload
(``backend/storage/technical_symbols/``).

Usage (from backend/):

    python scripts/extract_technical_symbols.py "../../Real data/1961BIDDrawings.pdf" --out ./tmp_sym
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.symbol_table_extractor import extract_symbol_table_from_pdf  # noqa: E402
from ml.symbol_table_pdf import (  # noqa: E402
    generate_technical_symbol_pdf,
    technical_symbol_pdf_filename,
)


def _collect_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise SystemExit(f"Not a PDF: {path}")
        return [path]
    if path.is_dir():
        unique = {p.resolve(): p for p in path.glob("*.pdf")}
        return sorted(unique.values(), key=lambda p: p.name.lower())
    raise SystemExit(f"Path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF file or folder of PDFs")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for ``<stem> technical symbol.pdf`` files",
    )
    args = parser.parse_args()

    out_dir = args.out.resolve()
    pdfs = _collect_pdfs(args.input.resolve())
    if not pdfs:
        raise SystemExit(f"No PDF files found in {args.input}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting technical symbols from {len(pdfs)} PDF(s) -> {out_dir}")
    ok = 0
    for pdf in pdfs:
        try:
            info = extract_symbol_table_from_pdf(pdf)
            out_path = out_dir / technical_symbol_pdf_filename(pdf.name)
            generate_technical_symbol_pdf(info, pdf.name, out_path)
            ok += 1
            print(
                f"  OK  {pdf.name}\n"
                f"      entries={info.entry_count}  "
                f"legend_pages={info.legend_pages}  "
                f"-> {out_path.name}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {pdf.name}: {exc}")
    print(f"Done: {ok}/{len(pdfs)} succeeded")


if __name__ == "__main__":
    main()
