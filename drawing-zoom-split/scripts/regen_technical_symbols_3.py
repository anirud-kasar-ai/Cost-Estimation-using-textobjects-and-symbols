"""Re-extract technical-symbol PDFs for three bid sets and score glyph quality."""

from __future__ import annotations

import io
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image

from pipeline.extraction.stages import _save_glyph_templates
from pipeline.extraction.symbol_table_extractor import extract_symbol_table_from_pdf
from pipeline.extraction.symbol_table_pdf import generate_technical_symbol_pdf

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT.parent / "Real data"
OUT_TECH = ROOT / "storage" / "technical_symbols"
OUT_DEBUG = ROOT / "storage" / "_debug_symbol_pdf" / "regen_3"
NAMES = (
    "1945BIDDrawingsRVES",
    "1948BIDDrawings",
    "1951BIDDrawings-NorthgateHS",
)


def _ink_frac(png: bytes) -> float:
    img = Image.open(io.BytesIO(png)).convert("L")
    dark = sum(1 for p in img.getdata() if p < 245)
    return dark / max(1, img.width * img.height)


def _edge_crumb_score(png: bytes) -> float:
    """Fraction of dark pixels in a 4px border ring (lower is cleaner)."""
    img = Image.open(io.BytesIO(png)).convert("L")
    w, h = img.size
    if w < 12 or h < 12:
        return 0.0
    pix = img.load()
    ring = 0
    dark = 0
    for y in range(h):
        for x in range(w):
            if x < 4 or y < 4 or x >= w - 4 or y >= h - 4:
                ring += 1
                if pix[x, y] < 245:
                    dark += 1
    return dark / max(1, ring)


def process(name: str) -> dict:
    pdf = REAL / f"{name}.pdf"
    if not pdf.is_file():
        job = ROOT / "storage" / "jobs" / name / "01_source" / f"{name}.pdf"
        pdf = job if job.is_file() else pdf
    if not pdf.is_file():
        raise FileNotFoundError(name)

    t0 = time.time()
    info = extract_symbol_table_from_pdf(pdf)
    elapsed = time.time() - t0

    glyphs = [e for e in info.entries if e.symbol_image_png]
    abbrev = sum(1 for e in info.entries if "ABBREV" in (e.legend_title or "").upper())
    edge_scores = [_edge_crumb_score(e.symbol_image_png or b"") for e in glyphs]
    ink_scores = [_ink_frac(e.symbol_image_png or b"") for e in glyphs]
    clean = sum(1 for s in edge_scores if s < 0.02)
    mean_edge = sum(edge_scores) / max(1, len(edge_scores))
    mean_ink = sum(ink_scores) / max(1, len(ink_scores))

    # Persist technical symbol PDF. Prefer the canonical folder; fall back to
    # _debug if Windows has the destination locked open in a viewer.
    preferred = OUT_TECH / f"{name} technical symbol.pdf"
    fallback = OUT_DEBUG / f"{name} technical symbol.pdf"
    stamp = OUT_TECH / f"{name} technical symbol.regen.pdf"
    out_pdf = stamp
    try:
        generate_technical_symbol_pdf(info, f"{name}.pdf", stamp)
    except PermissionError:
        out_pdf = fallback
        generate_technical_symbol_pdf(info, f"{name}.pdf", out_pdf)
    for dest in (
        preferred,
        OUT_TECH / f"{name} technical symbol (updated).pdf",
        OUT_TECH / f"{name} technical symbol.new.pdf",
        fallback,
    ):
        if dest == out_pdf:
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_pdf, dest)
        except PermissionError:
            pass
    if preferred.is_file():
        out_pdf = preferred
    elif (OUT_TECH / f"{name} technical symbol (updated).pdf").is_file():
        out_pdf = OUT_TECH / f"{name} technical symbol (updated).pdf"

    # Job + cache + glyph dump for inspection.
    job_dir = ROOT / "storage" / "jobs" / name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "symbol_table.json").write_text(
        json.dumps(info.to_json_dict(), indent=2), encoding="utf-8"
    )
    gdir = job_dir / "07_glyphs"
    if gdir.exists():
        shutil.rmtree(gdir)
    _save_glyph_templates(info, gdir)

    cache = ROOT / "storage" / "extraction_cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_dir / "symbol_table.json", cache / "symbol_table.json")

    dbg = OUT_DEBUG / name
    if dbg.exists():
        shutil.rmtree(dbg)
    dbg.mkdir(parents=True)
    for i, entry in enumerate(info.entries):
        if not entry.symbol_image_png:
            continue
        safe = re.sub(r"[^\w\-]+", "_", (entry.description or entry.symbol or f"e{i}")[:42])
        Image.open(io.BytesIO(entry.symbol_image_png)).save(dbg / f"{i:02d}_{safe}.png")

    # Render first page of the generated PDF for quick visual check.
    try:
        import fitz

        doc = fitz.open(out_pdf)
        if doc.page_count:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            Image.open(io.BytesIO(pix.tobytes("png"))).save(dbg / "pdf_p1.png")
        doc.close()
    except Exception as exc:  # noqa: BLE001
        print("pdf preview failed:", exc)

    summary = {
        "name": name,
        "source": str(pdf),
        "entries": info.entry_count,
        "glyphs": len(glyphs),
        "abbrev_rows": abbrev,
        "glyph_coverage": round(len(glyphs) / max(1, info.entry_count), 3),
        "clean_glyphs": clean,
        "mean_edge_crumb": round(mean_edge, 4),
        "mean_ink_frac": round(mean_ink, 4),
        "seconds": round(elapsed, 1),
        "out_pdf": str(out_pdf),
        "notes": info.notes[:5],
        "sample": [
            {
                "i": i,
                "tag": e.symbol,
                "desc": (e.description or "")[:60],
                "has_glyph": bool(e.symbol_image_png),
                "legend": (e.legend_title or "")[:40],
            }
            for i, e in enumerate(info.entries[:12])
        ],
    }
    (dbg / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    OUT_TECH.mkdir(parents=True, exist_ok=True)
    OUT_DEBUG.mkdir(parents=True, exist_ok=True)
    results = []
    for name in NAMES:
        print("=" * 60)
        print("Extracting", name)
        summary = process(name)
        results.append(summary)
        print(
            f"  entries={summary['entries']} glyphs={summary['glyphs']} "
            f"coverage={summary['glyph_coverage']} clean={summary['clean_glyphs']} "
            f"edge={summary['mean_edge_crumb']} {summary['seconds']}s"
        )
        print("  ->", summary["out_pdf"])
    (OUT_DEBUG / "all_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("=" * 60)
    print("DONE", OUT_DEBUG)


if __name__ == "__main__":
    main()
