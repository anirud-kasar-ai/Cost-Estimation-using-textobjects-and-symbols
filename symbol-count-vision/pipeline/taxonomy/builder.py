"""Build SymbolTaxonomyEntry lists from a technical-symbol PDF or in-memory legend."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.cv.tag_match import count_key
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.taxonomy.infer import infer_taxonomy_fields
from pipeline.taxonomy.schema import SymbolTaxonomy, SymbolTaxonomyEntry


def _safe_filename(key: str, idx: int) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", key).strip()[:80] or f"entry_{idx}"
    return f"{safe}.png"


def taxonomy_from_legend(
    legend: SymbolTableInfo,
    glyph_dir: Path | None = None,
    *,
    source: str = "",
) -> SymbolTaxonomy:
    """Map a parsed legend into taxonomy entries (in-memory, no disk required)."""
    entries: list[SymbolTaxonomyEntry] = []
    seen: set[str] = set()
    glyph_dir = Path(glyph_dir) if glyph_dir else None
    for idx, row in enumerate(legend.entries):
        key = count_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        shape, rule, semantics = infer_taxonomy_fields(row)
        png = row.symbol_image_png
        glyph_path = ""
        if glyph_dir is not None:
            glyph_dir.mkdir(parents=True, exist_ok=True)
            candidate = glyph_dir / _safe_filename(key, idx)
            if not candidate.is_file() and png:
                candidate.write_bytes(png)
            if candidate.is_file():
                glyph_path = str(candidate)
            else:
                stem = re.sub(r'[<>:"/\\|?*]', "_", key)[:80]
                matches = sorted(glyph_dir.glob(f"{stem}*.png"))
                if matches:
                    glyph_path = str(matches[0])
        if not png and glyph_path:
            try:
                png = Path(glyph_path).read_bytes()
            except OSError:
                png = None
        entries.append(
            SymbolTaxonomyEntry(
                key=key,
                description=(row.description or "").strip(),
                glyph_crop_path=glyph_path,
                shape_class=shape,
                context_rule=rule,
                count_semantics=semantics,
                tag=(row.symbol or "").strip() or None,
                part_number=(row.part_number or "").strip() or None,
                glyph_png=png,
            )
        )
    return SymbolTaxonomy(entries=entries, source=source)


def build_taxonomy_from_pdf(
    symbol_pdf: Path,
    output_dir: Path,
    *,
    name: str | None = None,
) -> tuple[SymbolTaxonomy, Path]:
    """Parse a technical-symbol PDF → symbol_taxonomy.json + glyphs/ folder."""
    from pipeline.symbol_legend import parse_symbol_file, save_glyph_templates

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    glyph_dir = output_dir / "glyphs"
    legend, parsed_glyphs = parse_symbol_file(symbol_pdf, output_dir)
    if parsed_glyphs and parsed_glyphs.is_dir():
        glyph_dir.mkdir(parents=True, exist_ok=True)
        for png in parsed_glyphs.glob("*.png"):
            dest = glyph_dir / png.name
            if not dest.exists():
                dest.write_bytes(png.read_bytes())
    elif any(e.symbol_image_png for e in legend.entries):
        save_glyph_templates(legend, glyph_dir)

    taxonomy = taxonomy_from_legend(
        legend,
        glyph_dir if glyph_dir.is_dir() else None,
        source=str(symbol_pdf),
    )
    json_path = output_dir / "symbol_taxonomy.json"
    taxonomy.save(json_path)
    return taxonomy, json_path


def load_or_build(
    legend: SymbolTableInfo,
    glyph_dir: Path | None = None,
    *,
    taxonomy_path: Path | None = None,
    source: str = "",
) -> SymbolTaxonomy:
    if taxonomy_path and taxonomy_path.is_file():
        return SymbolTaxonomy.load(taxonomy_path)
    return taxonomy_from_legend(legend, glyph_dir, source=source)
