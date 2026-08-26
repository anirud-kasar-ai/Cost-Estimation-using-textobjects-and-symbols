"""Parallel extraction stages: requirements, symbol legend, sheet notes."""

from __future__ import annotations

import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from pipeline import config
from pipeline.extraction.requirement_extractor import (
    RequirementInfo,
    extract_requirement_from_pdf,
)
from pipeline.extraction.requirement_pdf import (
    generate_requirement_pdf,
    requirement_pdf_filename,
)
from pipeline.extraction.sheet_notes_extractor import (
    NoteItem,
    SheetNotesInfo,
    extract_sheet_notes_from_pdf,
    write_diagram_notes_json,
)
from pipeline.extraction.sheet_notes_pdf import (
    generate_sheet_notes_pdf,
    sheet_notes_pdf_filename,
)
from pipeline.extraction.symbol_table_extractor import (
    SymbolTableInfo,
    extract_symbol_table_from_pdf,
    load_symbol_table_json,
)
from pipeline.extraction.symbol_table_pdf import (
    generate_technical_symbol_pdf,
    technical_symbol_pdf_filename,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]
_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _cache_id(source_filename: str) -> str:
    stem = Path(source_filename or "upload").stem.strip()
    stem = _INVALID_FOLDER_CHARS.sub("_", stem).strip(" .")
    stem = re.sub(r"_+", "_", stem) or "upload"
    return stem[:120]


def _pdf_fingerprint(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
    return f"{stat.st_size}-{mtime_ns}"


def _cache_dir(source_filename: str) -> Path:
    return config.EXTRACTION_CACHE_DIR / _cache_id(source_filename)


def _sheet_notes_from_json(data: dict[str, Any]) -> SheetNotesInfo:
    items: list[NoteItem] = []
    for raw in data.get("items") or []:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        items.append(
            NoteItem(
                note_type=str(raw.get("note_type") or "general"),
                text=str(raw.get("text") or ""),
                symbol=raw.get("symbol"),
                tag_id=raw.get("tag_id"),
                location=raw.get("location"),
                source_page=int(raw.get("source_page") or 1),
                sheet_number=raw.get("sheet_number"),
                slug=raw.get("slug"),
                display_name=raw.get("display_name"),
                related_diagram_page=raw.get("related_diagram_page"),
                related_diagram_sheet=raw.get("related_diagram_sheet"),
                heading=raw.get("heading"),
                bbox=tuple(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
            )
        )
    return SheetNotesInfo(
        items=items,
        total_pages=int(data.get("total_pages") or 0),
        note_pages=[int(v) for v in data.get("note_pages") or []],
        notes=[str(v) for v in data.get("notes") or []],
    )


def _load_extraction_cache(
    pdf_path: Path,
    source_filename: str,
    job_root: Path,
) -> tuple[dict[str, Any], SheetNotesInfo | None] | None:
    if not config.EXTRACTION_CACHE_ENABLED:
        return None
    cache_root = _cache_dir(source_filename)
    manifest_path = cache_root / "manifest.json"
    extraction_path = cache_root / "extraction.json"
    symbol_path = cache_root / "symbol_table.json"
    notes_path = cache_root / "sheet_notes.json"
    if not all(path.is_file() for path in (manifest_path, extraction_path, symbol_path, notes_path)):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != _pdf_fingerprint(pdf_path):
            return None
        result = json.loads(extraction_path.read_text(encoding="utf-8"))
        shutil.copy2(symbol_path, job_root / "symbol_table.json")
        shutil.copy2(notes_path, job_root / "sheet_notes.json")
        notes_data = json.loads(notes_path.read_text(encoding="utf-8"))
        sheet_notes = _sheet_notes_from_json(notes_data)
        legend = load_symbol_table_json(job_root / "symbol_table.json")
        if legend is not None:
            _save_glyph_templates(legend, job_root / "07_glyphs")
        logger.info("Using cached extraction for %s", source_filename)
        return result, sheet_notes
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.exception("Failed to load extraction cache for %s", source_filename)
        return None


def _save_extraction_cache(
    pdf_path: Path,
    source_filename: str,
    result: dict[str, Any],
    job_root: Path,
) -> None:
    if not config.EXTRACTION_CACHE_ENABLED:
        return
    cache_root = _cache_dir(source_filename)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fingerprint": _pdf_fingerprint(pdf_path),
        "source_filename": source_filename,
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (cache_root / "extraction.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    symbol_path = job_root / "symbol_table.json"
    notes_path = job_root / "sheet_notes.json"
    if symbol_path.is_file():
        shutil.copy2(symbol_path, cache_root / "symbol_table.json")
    if notes_path.is_file():
        shutil.copy2(notes_path, cache_root / "sheet_notes.json")


def _save_glyph_templates(info: SymbolTableInfo, out_dir: Path) -> int:
    """Persist in-memory glyph PNGs to disk for CV template matching."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, entry in enumerate(info.entries):
        if not entry.symbol_image_png:
            continue
        key = (entry.symbol or entry.description or "").strip()
        safe = re.sub(r'[<>:"/\\|?*]', "_", key)[:80] or f"entry_{idx}"
        dest = out_dir / f"{safe}.png"
        if dest.exists():
            dest = out_dir / f"{safe}_{idx}.png"
        dest.write_bytes(entry.symbol_image_png)
        saved += 1
    logger.info("Saved %d glyph templates to %s", saved, out_dir)
    return saved


def _extract_requirement(
    pdf_path: Path,
    source_filename: str,
) -> tuple[RequirementInfo | None, str | None]:
    try:
        info = extract_requirement_from_pdf(pdf_path)
        out_name = requirement_pdf_filename(source_filename)
        out_path = config.REQUIREMENTS_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_requirement_pdf(info, source_filename, out_path)
        logger.info(
            "Requirement PDF written to %s (provider=%s)",
            out_path,
            info.provider_summary,
        )
        return info, str(out_path)
    except Exception:
        logger.exception("Requirement extraction failed for %s", pdf_path)
        return None, None


def _extract_technical_symbol(
    pdf_path: Path,
    source_filename: str,
) -> tuple[SymbolTableInfo | None, str | None]:
    try:
        info = extract_symbol_table_from_pdf(pdf_path)
        out_name = technical_symbol_pdf_filename(source_filename)
        out_path = config.TECHNICAL_SYMBOLS_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_technical_symbol_pdf(info, source_filename, out_path)
        logger.info(
            "Technical symbol PDF written to %s (entries=%d)",
            out_path,
            info.entry_count,
        )
        return info, str(out_path)
    except Exception:
        logger.exception("Technical symbol extraction failed for %s", pdf_path)
        return None, None


def _extract_sheet_notes(
    pdf_path: Path,
    source_filename: str,
    job_root: Path,
) -> tuple[SheetNotesInfo | None, str | None]:
    try:
        info = extract_sheet_notes_from_pdf(pdf_path)
        out_name = sheet_notes_pdf_filename(source_filename)
        out_path = job_root / out_name
        generate_sheet_notes_pdf(info, source_filename, out_path)
        logger.info(
            "Sheet notes PDF written to %s (items=%d)",
            out_path,
            info.item_count,
        )
        return info, str(out_path)
    except Exception:
        logger.exception("Sheet notes extraction failed for %s", pdf_path)
        return None, None


def _requirement_metadata(info: RequirementInfo) -> dict[str, Any]:
    return {
        "provider_company": info.provider_company,
        "provider_agent": info.provider_agent,
        "provider_phone": info.provider_phone,
        "client": info.client,
        "project_title": info.project_title,
        "site_address": info.site_address,
        "job_number": info.job_number,
        "date": info.date,
        "scope_section_count": len(info.scope_sections),
    }


def _symbol_metadata(info: SymbolTableInfo) -> dict[str, Any]:
    return {
        "entry_count": info.entry_count,
        "legend_pages": info.legend_pages,
        "total_pages": info.total_pages,
    }


def _build_extraction_result(
    requirement: RequirementInfo | None,
    requirement_pdf_path: str | None,
    technical_symbol: SymbolTableInfo | None,
    technical_symbol_pdf_path: str | None,
    sheet_notes: SheetNotesInfo | None,
    sheet_notes_pdf_path: str | None,
    *,
    job_root: Path,
    symbol_table_json_path: str | None,
    notes_json_path: str | None,
) -> dict[str, Any]:
    return {
        "requirement": _requirement_metadata(requirement) if requirement else None,
        "requirement_pdf_path": requirement_pdf_path,
        "requirement_provider": requirement.provider_summary if requirement else None,
        "has_requirement_pdf": bool(requirement_pdf_path),
        "metadata": {
            "title": requirement.project_title if requirement else None,
            "client": requirement.client if requirement else None,
            "architect": requirement.provider_company if requirement else None,
            "engineer": None,
            "project_address": requirement.site_address if requirement else None,
            "due_date": requirement.date if requirement else None,
        },
        "technical_symbol": _symbol_metadata(technical_symbol) if technical_symbol else None,
        "technical_symbol_pdf_path": technical_symbol_pdf_path,
        "has_technical_symbol_pdf": bool(technical_symbol_pdf_path),
        "symbol_entry_count": technical_symbol.entry_count if technical_symbol else 0,
        "symbol_table_json_path": symbol_table_json_path,
        "sheet_notes": sheet_notes.to_json_dict() if sheet_notes else None,
        "sheet_notes_pdf_path": sheet_notes_pdf_path,
        "has_sheet_notes_pdf": bool(sheet_notes_pdf_path),
        "sheet_notes_json_path": notes_json_path,
        "sheet_notes_item_count": sheet_notes.item_count if sheet_notes else 0,
    }


def run_extraction_stages(
    pdf_path: Path,
    source_filename: str,
    job_root: Path,
    *,
    on_progress: ProgressFn | None = None,
) -> tuple[dict[str, Any], SheetNotesInfo | None]:
    """Run requirement, symbol, and sheet-notes extraction in parallel."""

    def _progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    config.ensure_storage_dirs()
    cached = _load_extraction_cache(pdf_path, source_filename, job_root)
    if cached is not None:
        _progress("Loaded cached requirements, symbols, and sheet notes")
        return cached

    _progress("Extracting requirements, symbol legend, and sheet notes…")
    tasks = {
        "requirements": lambda: _extract_requirement(pdf_path, source_filename),
        "symbol legend": lambda: _extract_technical_symbol(pdf_path, source_filename),
        "sheet notes": lambda: _extract_sheet_notes(pdf_path, source_filename, job_root),
    }
    results: dict[str, tuple[Any, str | None]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): label for label, fn in tasks.items()}
        for future in as_completed(futures):
            label = futures[future]
            _progress(f"Finished {label}…")
            results[label] = future.result()

    requirement, requirement_pdf_path = results["requirements"]
    technical_symbol, technical_symbol_pdf_path = results["symbol legend"]
    sheet_notes, sheet_notes_pdf_path = results["sheet notes"]

    notes_json_path: str | None = None
    if sheet_notes is not None:
        (job_root / "sheet_notes.json").write_text(
            json.dumps(sheet_notes.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        notes_json_path = str(job_root / "sheet_notes.json")

    symbol_table_json_path: str | None = None
    if technical_symbol is not None:
        (job_root / "symbol_table.json").write_text(
            json.dumps(technical_symbol.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        symbol_table_json_path = str(job_root / "symbol_table.json")
        _save_glyph_templates(technical_symbol, job_root / "07_glyphs")

    result = _build_extraction_result(
        requirement,
        requirement_pdf_path,
        technical_symbol,
        technical_symbol_pdf_path,
        sheet_notes,
        sheet_notes_pdf_path,
        job_root=job_root,
        symbol_table_json_path=symbol_table_json_path,
        notes_json_path=notes_json_path,
    )
    _save_extraction_cache(pdf_path, source_filename, result, job_root)
    _progress("Extraction complete")
    return result, sheet_notes


def finalize_sheet_notes(sheet_notes: SheetNotesInfo | None, job_root: Path) -> int:
    """Write per-diagram note sidecars after page metadata exists."""
    if sheet_notes is None:
        return 0
    return write_diagram_notes_json(sheet_notes, job_root)
