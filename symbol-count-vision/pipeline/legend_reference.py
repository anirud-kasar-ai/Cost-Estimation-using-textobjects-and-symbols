"""Persistent glyph + shape-description references for technical-symbol sheets.

A "reference legend" is a folder under storage/reference_legends/<name>/ with:
    glyphs/<key>.png     — cropped glyph artwork per legend key
    descriptions.json    — legend entries + {"<key>": "<shape description>"}

References are built once from an uploaded technical-symbol PDF (see
scripts/build_reference_legend.py). Different projects' sheets can reuse the
same key (e.g. '#', 'AP') for completely different artwork, so a stored
description is only applied to a job when the job legend's entry PROVABLY
matches the reference entry:
  1. the glyph artwork hash (sha1 of the PNG crop) is identical, or
  2. key + description + tag + part all match the stored legend row.
Anything else falls through to a fresh Gemini describe call for that sheet.
Descriptions can be edited by hand in descriptions.json; edits survive
rebuilds of the same reference.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from . import config
from .cv.tag_match import count_key
from .extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from .symbol_legend import parse_symbol_file

logger = logging.getLogger(__name__)

REFERENCE_DIR = config.RESULTS_DIR / "reference_legends"


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()[:80] or "legend"


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().upper())


def _glyph_sha1(entry: SymbolEntry) -> str | None:
    if not entry.symbol_image_png:
        return None
    return hashlib.sha1(entry.symbol_image_png).hexdigest()


def _row_fingerprint(
    key: str | None,
    description: str | None,
    tag: str | None,
    part: str | None,
) -> tuple[str, str, str, str]:
    return (_norm(key), _norm(description), _norm(tag), _norm(part))


def _load_reference_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not REFERENCE_DIR.is_dir():
        return payloads
    for desc_file in sorted(REFERENCE_DIR.glob("*/descriptions.json")):
        try:
            payload = json.loads(desc_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable reference file: %s", desc_file)
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def lookup_descriptions_for_legend(legend: SymbolTableInfo) -> dict[str, str]:
    """Stored shape descriptions for entries that provably match a reference.

    Matching is per-entry, never key-only: the glyph artwork hash must be
    identical, or the whole legend row (key + description + tag + part) must
    match. This keeps one project's '#' from borrowing another project's
    '#' description when the sheets differ.
    """
    by_hash: dict[str, str] = {}
    by_row: dict[tuple[str, str, str, str], str] = {}
    for payload in _load_reference_payloads():
        entries = payload.get("entries")
        if not isinstance(entries, list):
            continue
        for row in entries:
            if not isinstance(row, dict):
                continue
            desc = row.get("glyph")
            if not isinstance(desc, str) or not desc.strip():
                continue
            desc = desc.strip()[:120]
            sha = row.get("glyph_sha1")
            if isinstance(sha, str) and sha:
                by_hash.setdefault(sha, desc)
            by_row.setdefault(
                _row_fingerprint(
                    row.get("key"), row.get("description"),
                    row.get("tag"), row.get("part"),
                ),
                desc,
            )

    matched: dict[str, str] = {}
    for entry in legend.entries:
        key = count_key(entry)
        if not key or key in matched:
            continue
        sha = _glyph_sha1(entry)
        if sha and sha in by_hash:
            matched[key] = by_hash[sha]
            continue
        fp = _row_fingerprint(
            key, entry.description, entry.symbol, entry.part_number
        )
        if fp in by_row:
            matched[key] = by_row[fp]
    return matched


def build_reference_legend(
    symbol_pdf: Path,
    *,
    name: str | None = None,
) -> tuple[Path, dict[str, str]]:
    """Parse a technical-symbol PDF into a persistent reference folder.

    Saves every glyph crop, generates shape descriptions with the vision model
    (one call), and writes descriptions.json. Hand-edited descriptions in an
    existing descriptions.json for the same reference name are preserved.
    """
    from .llm_verify import describe_legend_glyphs_with_llm

    ref_dir = REFERENCE_DIR / _safe_name(name or symbol_pdf.stem)
    ref_dir.mkdir(parents=True, exist_ok=True)

    legend, _glyph_dir = parse_symbol_file(symbol_pdf, ref_dir)
    # parse_symbol_file writes glyphs to 07_glyphs; expose them as glyphs/.
    src = ref_dir / "07_glyphs"
    dst = ref_dir / "glyphs"
    if src.is_dir():
        dst.mkdir(exist_ok=True)
        for png in src.glob("*.png"):
            (dst / png.name).write_bytes(png.read_bytes())

    # Reuse this reference's own previous descriptions (keeps manual edits).
    existing: dict[str, str] = {}
    prior_file = ref_dir / "descriptions.json"
    if prior_file.is_file():
        try:
            prior = json.loads(prior_file.read_text(encoding="utf-8"))
            if isinstance(prior.get("glyph_descriptions"), dict):
                existing = {
                    str(k): str(v)
                    for k, v in prior["glyph_descriptions"].items()
                    if isinstance(v, str) and v.strip()
                }
        except (OSError, json.JSONDecodeError):
            pass

    descriptions: dict[str, str] = {}
    missing_keys: list[str] = []
    for entry in legend.entries:
        key = count_key(entry)
        if not key or key in descriptions:
            continue
        if key in existing:
            descriptions[key] = existing[key]
        else:
            missing_keys.append(key)

    if missing_keys and config.llm_enabled():
        generated = describe_legend_glyphs_with_llm(legend, set())
        for key in missing_keys:
            if generated.get(key):
                descriptions[key] = generated[key]

    # Keep legend metadata + artwork hash so job-time matching is per-entry,
    # not key-only (different sheets may reuse a key for different symbols).
    payload = {
        "source_pdf": str(symbol_pdf),
        "entries": [
            {
                "key": count_key(e),
                "tag": (e.symbol or "").strip() or None,
                "part": (e.part_number or "").strip() or None,
                "description": (e.description or "").strip() or None,
                "glyph": descriptions.get(count_key(e) or ""),
                "glyph_sha1": _glyph_sha1(e),
            }
            for e in legend.entries
            if count_key(e)
        ],
        "glyph_descriptions": descriptions,
    }
    (ref_dir / "descriptions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ref_dir, descriptions


__all__ = [
    "REFERENCE_DIR",
    "build_reference_legend",
    "lookup_descriptions_for_legend",
]
