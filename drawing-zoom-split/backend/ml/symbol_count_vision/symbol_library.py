"""Persistent symbol glyph library.

Legend PDFs vary: some carry real symbol artwork (embedded images or vector
cells), others are text-only summaries with no graphics at all. This module
remembers every real glyph the app has ever seen, keyed by normalized
description, so a later text-only legend for the same project still gets
usable CV templates.

Layout on disk (``storage/symbol_library/``)::

    index.json            {key: [{"file", "symbol", "mfg_model", "part_number", "source"}]}
    <sha1>.png            glyph artwork
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pipeline import config
from pipeline.extraction.symbol_table_extractor import (
    SymbolTableInfo,
    extract_symbol_table_from_pdf,
)

LIBRARY_DIR = config.ROOT / "storage" / "symbol_library"
INDEX_PATH = LIBRARY_DIR / "index.json"

# Legend PDFs already present in the workspace that carry real glyph artwork;
# used to seed the library the first time it is empty.
SEED_PDF_DIRS = [
    config.ROOT.parent / "drawing-zoom-split" / "storage" / "technical_symbols",
]


def normalize_key(description: str | None, symbol: str | None = None) -> str:
    """Punctuation/spacing-insensitive key so 'FOO - BAR' matches 'FOO BAR'."""
    base = str(description or "").strip() or str(symbol or "").strip()
    return re.sub(r"[^A-Z0-9]+", " ", base.upper()).strip()


def _load_index() -> dict[str, list[dict[str, Any]]]:
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_index(index: dict[str, list[dict[str, Any]]]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_from_info(info: SymbolTableInfo, *, source: str) -> int:
    """Store real (non-synthetic) glyphs from a parsed legend. Returns #added."""
    index = _load_index()
    added = 0
    for entry in info.entries:
        png = entry.symbol_image_png
        if not png or entry.glyph_source is not None:
            continue
        key = normalize_key(entry.description, entry.symbol)
        if not key:
            continue
        digest = hashlib.sha1(png).hexdigest()
        filename = f"{digest}.png"
        variants = index.setdefault(key, [])
        if any(v.get("file") == filename for v in variants):
            continue
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        (LIBRARY_DIR / filename).write_bytes(png)
        variants.append(
            {
                "file": filename,
                "symbol": entry.symbol,
                "mfg_model": entry.mfg_model,
                "part_number": entry.part_number,
                "source": source,
            }
        )
        added += 1
    if added:
        _save_index(index)
    return added


def _pick_variant(
    variants: list[dict[str, Any]], entry_mfg: str | None, entry_part: str | None
) -> dict[str, Any] | None:
    def norm(value: Any) -> str:
        return normalize_key(str(value or ""))

    if entry_mfg or entry_part:
        for variant in variants:
            if entry_part and norm(variant.get("part_number")) == norm(entry_part):
                return variant
        for variant in variants:
            if entry_mfg and norm(variant.get("mfg_model")) == norm(entry_mfg):
                return variant
    return variants[0] if variants else None


def resolve_info(info: SymbolTableInfo) -> int:
    """Attach library glyphs to entries lacking real artwork. Returns #resolved."""
    index = _load_index()
    if not index:
        return 0
    resolved = 0
    for entry in info.entries:
        if entry.symbol_image_png and entry.glyph_source is None:
            continue  # already has real artwork from the uploaded file
        key = normalize_key(entry.description, entry.symbol)
        variants = index.get(key)
        if not variants:
            # Loose fallback: containment either way, for wording drift
            # between PDFs describing the same symbol.
            if len(key) >= 12:
                for lib_key, lib_variants in index.items():
                    if len(lib_key) >= 12 and (key in lib_key or lib_key in key):
                        variants = lib_variants
                        break
        if not variants:
            continue
        variant = _pick_variant(variants, entry.mfg_model, entry.part_number)
        if not variant:
            continue
        try:
            png = (LIBRARY_DIR / str(variant["file"])).read_bytes()
        except OSError:
            continue
        entry.symbol_image_png = png
        entry.glyph_source = "library"
        resolved += 1
    return resolved


def all_variant_pngs(
    description: str | None,
    symbol: str | None = None,
    *,
    exclude: bytes | None = None,
) -> list[bytes]:
    """Every stored glyph variant for a symbol (minus ``exclude``)."""
    index = _load_index()
    variants = index.get(normalize_key(description, symbol)) or []
    out: list[bytes] = []
    for variant in variants:
        try:
            png = (LIBRARY_DIR / str(variant["file"])).read_bytes()
        except OSError:
            continue
        if exclude is not None and png == exclude:
            continue
        out.append(png)
    return out


def ensure_seeded() -> int:
    """One-time seed from glyph-rich legend PDFs already in the workspace."""
    if _load_index():
        return 0
    seeded = 0
    for seed_dir in SEED_PDF_DIRS:
        if not seed_dir.is_dir():
            continue
        for pdf in sorted(seed_dir.glob("*.pdf")):
            try:
                info = extract_symbol_table_from_pdf(pdf)
            except Exception:
                continue
            if info and info.entries:
                seeded += add_from_info(info, source=pdf.name)
    return seeded
