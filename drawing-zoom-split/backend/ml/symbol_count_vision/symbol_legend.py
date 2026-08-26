from __future__ import annotations

import re
from pathlib import Path

from pipeline.extraction.symbol_table_extractor import (
    SymbolEntry,
    SymbolTableInfo,
    catalog_id_for_entry,
    extract_symbol_table_from_pdf,
    load_symbol_table_json,
)

__all__ = [
    "SymbolEntry",
    "SymbolTableInfo",
    "load_symbol_table_json",
    "extract_symbol_table_from_pdf",
    "catalog_id_for_entry",
    "save_glyph_templates",
    "parse_symbol_file",
    "_display_key",
]


def _display_key(symbol: str | None, description: str) -> str:
    return str(symbol or description or "").strip()


# Prefer the CV count_key (includes part numbers) when available.
try:
    from pipeline.cv.tag_match import count_key as _count_key
except Exception:  # noqa: BLE001
    def _count_key(entry):  # type: ignore[no-redef]
        return _display_key(getattr(entry, "symbol", None), getattr(entry, "description", ""))


def save_glyph_templates(info: SymbolTableInfo, out_dir: Path) -> int:
    """Persist in-memory glyph PNGs to disk for CV template matching."""
    from pipeline import symbol_library
    from pipeline.cv.tag_match import count_key

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, entry in enumerate(info.entries):
        if not entry.symbol_image_png:
            continue
        key = count_key(entry) or (entry.symbol or entry.description or "").strip()
        safe = re.sub(r'[<>:"/\\|?*]', "_", key)[:80] or f"entry_{idx}"
        dest = out_dir / f"{safe}.png"
        if dest.exists():
            dest = out_dir / f"{safe}_{idx}.png"
        dest.write_bytes(entry.symbol_image_png)
        saved += 1

        if entry.glyph_source != "library":
            continue
        # Library-resolved entries may have several stored artwork variants
        # (e.g. the same symbol crops slightly differently across PDFs);
        # export them all so template matching sees every rendering.
        try:
            extras = symbol_library.all_variant_pngs(
                entry.description, entry.symbol, exclude=entry.symbol_image_png
            )
        except Exception:
            extras = []
        n = 1
        for png in extras:
            while (out_dir / f"{safe}_{n}.png").exists():
                n += 1
            (out_dir / f"{safe}_{n}.png").write_bytes(png)
            saved += 1
    return saved


def parse_symbol_file(symbol_file: Path, output_dir: Path) -> tuple[SymbolTableInfo, Path | None]:
    """Load legend from JSON or PDF; extract and save glyphs when PDF.

    Real glyphs from the uploaded file are remembered in the persistent
    symbol library; entries whose file carried no artwork get their glyph
    from the library so text-only legend PDFs remain fully detectable.
    """
    from pipeline import symbol_library

    suffix = symbol_file.suffix.lower()

    if suffix == ".json":
        legend = load_symbol_table_json(symbol_file)
    elif suffix == ".pdf":
        legend = extract_symbol_table_from_pdf(symbol_file)
    else:
        raise ValueError(f"Unsupported symbol file: {suffix}")

    if legend is None or not legend.entries:
        raise ValueError("Could not parse symbol file or it contained no entries.")

    try:
        symbol_library.ensure_seeded()
        symbol_library.add_from_info(legend, source=symbol_file.name)
        resolved = symbol_library.resolve_info(legend)
    except Exception:
        resolved = 0
    if resolved:
        legend.notes.append(
            f"{resolved} symbol graphic(s) were not present in the uploaded file "
            "and were attached from the symbol library."
        )

    glyph_dir: Path | None = None
    if any(entry.symbol_image_png for entry in legend.entries):
        glyph_dir = output_dir / "07_glyphs"
        save_glyph_templates(legend, glyph_dir)
        if not any(glyph_dir.glob("*.png")):
            glyph_dir = None

    return legend, glyph_dir
