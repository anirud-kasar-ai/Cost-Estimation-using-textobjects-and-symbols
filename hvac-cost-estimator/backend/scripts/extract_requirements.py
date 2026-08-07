"""Optional batch helper for requirement PDFs (dev / debugging only).

Production path: requirement PDFs are created only when a drawing is uploaded
through the API (``backend/storage/requirements/``). Do not pre-generate files
under ``Real data/requirements/``.

Usage (from backend/) — ``--out`` is required:

    python scripts/extract_requirements.py "../../Real data/1945BIDDrawingsRVES.pdf" --out ./tmp_req
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python scripts/extract_requirements.py`` from backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.requirement_extractor import extract_requirement_from_pdf  # noqa: E402
from ml.requirement_pdf import (  # noqa: E402
    generate_requirement_pdf,
    requirement_pdf_filename,
)


def _collect_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise SystemExit(f"Not a PDF: {path}")
        return [path]
    if path.is_dir():
        # Case-insensitive FS (Windows) can make *.pdf and *.PDF overlap.
        unique = {p.resolve(): p for p in path.glob("*.pdf")}
        return sorted(unique.values(), key=lambda p: p.name.lower())
    raise SystemExit(f"Path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="PDF file or folder of PDFs (e.g. a single drawing under Real data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder (required; do not use Real data/requirements)",
    )
    args = parser.parse_args()

    out_dir = args.out.resolve()
    if out_dir.name.lower() == "requirements" and "real data" in str(out_dir).lower():
        raise SystemExit(
            "Refusing to write into Real data/requirements. "
            "Requirement PDFs are created only on upload. Use a temp --out folder."
        )

    pdfs = _collect_pdfs(args.input.resolve())
    if not pdfs:
        raise SystemExit(f"No PDF files found in {args.input}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting requirements from {len(pdfs)} PDF(s) -> {out_dir}")
    ok = 0
    for pdf in pdfs:
        try:
            info = extract_requirement_from_pdf(pdf)
            out_path = out_dir / requirement_pdf_filename(pdf.name)
            generate_requirement_pdf(info, pdf.name, out_path)
            ok += 1
            print(
                f"  OK  {pdf.name}\n"
                f"      provider={info.provider_summary or '—'}\n"
                f"      client={info.client or '—'}  "
                f"title={info.project_title or '—'}  "
                f"job={info.job_number or '—'}  "
                f"scopes={len(info.scope_sections)}"
            )
        except Exception as exc:  # noqa: BLE001 — CLI batch continues
            print(f"  FAIL {pdf.name}: {exc}")
    print(f"Done: {ok}/{len(pdfs)}")


if __name__ == "__main__":
    main()
