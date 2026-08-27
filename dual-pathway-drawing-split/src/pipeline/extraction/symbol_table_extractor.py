"""Extract symbol legends, drafting symbols, and abbreviations from bid PDFs.

Scans every page (no CV page cap). Primary input is PyMuPDF embedded text;
CAD legend layout is often non-tabular, so rows are rebuilt with heuristics
and (for SYMBOLS / ABBREVIATIONS sheets) with spatial column clustering.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

LEGEND_HEADING_RE = re.compile(
    r"^(?:TECHNOLOGY\s+)?SYMBOL\s+LEGEND\b",
    re.IGNORECASE,
)
NUMBERED_ITEM_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")
STOP_SECTION_RE = re.compile(
    r"^(CONSULTANTS|SHEET\s+INDEX|TECHNOLOGY\s+SHEET\s+INDEX|"
    r"ABBREVIATIONS\s*:|PROJECT\s+OWNER|GENERAL\s+NOTES|"
    r"DRAWING\s+INDEX|KEY\s+PLAN|LOCATION\s+MAP|"
    r"GENERAL\s+PROJECT\s+NOTES)\b",
    re.IGNORECASE,
)
SHORT_TAG_RE = re.compile(r"^(?:#|[A-Z]{1,4}\d{0,3})$")
NOTE_RE = re.compile(
    r"^(SEE\b|PANDUIT|WIREMOLD|OFCI|N/?A|EXISTING|COMMERCIAL|GENERIC|"
    r"WM\d+|POLY|ARUBA|CHATSWORTH|BOGEN|GREY\s*=)",
    re.IGNORECASE,
)
EQUIPMENT_HINT_RE = re.compile(
    r"\b(CONDUIT|RACK|BOX|CABLE|DROP|RACEWAY|JUNCTION|GROUND|SPEAKER|"
    r"CAMERA|SWITCH|PANEL|CLOCK|PHONE|ANTENNA|RECORDER|CABINET|BUS\s+BAR|"
    r"STUB|LINK|PATHWAY|HOOK|MOUNT|ENCLOSURE|AP|WAP|IDF|MDF|DETAIL|"
    r"FLASHING|DRAIN|SCUPPER|CURB|PENETRATION|EDGE)\b",
    re.IGNORECASE,
)
JUNK_LINE_RE = re.compile(
    r"^(MFG/?MODEL|ALL EQUIPMENT AND MATERIALS|UNO\)?|#\s*=\s*QTY|"
    r"GREY\s*=|SEE\s+SINGLE|LINE|N/?A|EXISTING|COMMERCIAL|GENERIC|"
    r"OFCI|POLY|\d{1,3}|T-\d+)$",
    re.IGNORECASE,
)
HEADER_NOTE_RE = re.compile(
    r"ALL EQUIPMENT AND MATERIALS|#\s*=\s*QTY|NO\s*#\s*=\s*1|"
    r"CONTRACTOR FURNISHED.*CONFIGURED",
    re.IGNORECASE,
)

# Architect cover-sheet "SYMBOLS" / "ABBREVIATIONS" layout (1945-style).
SYMBOLS_SECTION_HEADING_RE = re.compile(r"^SYMBOLS\s*$", re.IGNORECASE)
ABBREVIATIONS_HEADING_RE = re.compile(r"^ABBREVIATIONS\s*$", re.IGNORECASE)
DRAFTING_HEADER_RE = re.compile(r"^DRAFTING\s+ITEMS\b", re.IGNORECASE)
VIEW_HEADER_RE = re.compile(r"^VIEW\s+REFERENCES\b", re.IGNORECASE)
# Architect plan sheets: LEGEND - FLOOR PLAN / LEGEND - CEILING (not GENERAL NOTES).
PLAN_LEGEND_HEADING_RE = re.compile(
    r"^LEGEND\s*[-–—]\s*(FLOOR\s+PLAN|CEILING)(?:\s+DEMOLITION)?\s*$",
    re.IGNORECASE,
)
PLAN_LEGEND_STOP_RE = re.compile(
    r"^LEGEND\s*[-–—].*GENERAL\s+NOTES\b|"
    r"^KEYNOTES?\b|^NOTE:\s*|"
    r"^GENERAL\s+NOTES\b|^KEY\s+PLAN\b|"
    r"^PROFESSIONAL\s+STAMP\b|^REVISIONS\b",
    re.IGNORECASE,
)
PLAN_DESC_CONTINUATION_RE = re.compile(
    r"^(BOARD\s+CEILING|IN\s+EXISTING|REMOVED\s+WHERE|ACOUSTICAL\s+CEILING|"
    r"REINSTALLED[, ]|AND\s+TYPE|MATERIAL\b|HATCH\b|"
    r"BE\s+DEMOLISHED|WHERE\s+INDICATED|TO\s+BE\s+(REMOVED|DEMOLISHED)|"
    r"GRILLE\s+TO\s+BE|GYPSUM\s+BOARD)",
    re.IGNORECASE,
)
PLAN_GLYPH_TEXT_RE = re.compile(
    r"^(\d{1,4}|[\d.]+\s*SF|[A-Z]|ACT|GYP|"
    r"\d+['′]\s*-\s*\d+|12['′]-0[\"″]?)$",
    re.IGNORECASE,
)
SYMBOLS_SKIP_LABEL_RE = re.compile(
    r"^(SIM|TYP\.?|CMNT\s*#|PROJECT|NORTH|NAME|ELEVATION|"
    r"01\.23\.A|101AA|A-?101|A1-100|_____)$",
    re.IGNORECASE,
)
# Continuation fragments that belong to a multi-line drafting / view label.
CONTINUATION_START_RE = re.compile(
    r"^(N\.?I\.?C\.|ITEM,|REMOVED|MATERIAL|SHEET\s+REFERENCE|"
    r"AND\s+TYPE|VIEW\s+REFERENCE)$",
    re.IGNORECASE,
)


@dataclass
class SymbolEntry:
    """One legend row: optional tag, description, optional notes."""

    description: str
    symbol: str | None = None
    notes: list[str] | None = None
    source_page: int = 1
    legend_title: str = "Symbol Legend"
    mfg_model: str | None = None
    part_number: str | None = None
    symbol_image_png: bytes | None = None
    row_x: float | None = None

    def to_json_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "description": self.description,
            "notes": list(self.notes) if self.notes else None,
            "source_page": self.source_page,
            "legend_title": self.legend_title,
            "mfg_model": self.mfg_model,
            "part_number": self.part_number,
            "has_glyph": bool(self.symbol_image_png),
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> SymbolEntry:
        return cls(
            symbol=data.get("symbol"),
            description=str(data.get("description") or ""),
            notes=list(data["notes"]) if data.get("notes") else None,
            source_page=int(data.get("source_page") or 1),
            legend_title=str(data.get("legend_title") or "Symbol Legend"),
            mfg_model=data.get("mfg_model"),
            part_number=data.get("part_number"),
        )


@dataclass
class SymbolTableInfo:
    """All symbol/legend entries extracted from a drawing set."""

    entries: list[SymbolEntry] = field(default_factory=list)
    legend_pages: list[int] = field(default_factory=list)
    total_pages: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def to_json_dict(self) -> dict:
        return {
            "entry_count": self.entry_count,
            "legend_pages": list(self.legend_pages),
            "total_pages": self.total_pages,
            "notes": list(self.notes),
            "entries": [entry.to_json_dict() for entry in self.entries],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> SymbolTableInfo:
        entries = [
            SymbolEntry.from_json_dict(item)
            for item in (data.get("entries") or [])
            if isinstance(item, dict)
        ]
        return cls(
            entries=entries,
            legend_pages=[int(p) for p in (data.get("legend_pages") or [])],
            total_pages=int(data.get("total_pages") or 0),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


def load_symbol_table_json(path: Path) -> SymbolTableInfo | None:
    """Load ``symbol_table.json`` written during extraction."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return SymbolTableInfo.from_json_dict(data)


def catalog_id_for_entry(entry: SymbolEntry) -> str:
    """Stable vision/report id for a legend row."""
    sym = str(entry.symbol or "").strip()
    if sym:
        return sym
    desc = str(entry.description or "").strip()
    part = str(entry.part_number or "").strip()
    if desc and part:
        return f"{desc} ({part})"
    return desc or part


@dataclass
class _TextSpan:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def extract_symbol_table_from_pdf(pdf_path: Path) -> SymbolTableInfo:
    """Open PDF, scan all pages for symbol legends, return structured entries."""
    info = SymbolTableInfo()
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        info.notes.append(f"Could not open PDF: {exc}")
        return info

    try:
        info.total_pages = document.page_count
        for page_index in range(document.page_count):
            page = document[page_index]
            page_number = page_index + 1
            text = page.get_text("text") or ""
            if not _page_has_legend(text):
                continue

            page_entries: list[SymbolEntry] = []
            spans = _page_text_spans(page)

            # Rotated Architect-CD TECHNOLOGY SYMBOL LEGEND (spatial columns).
            spatial_tech: list[SymbolEntry] = []
            if "TECHNOLOGY SYMBOL LEGEND" in text.upper():
                spatial_tech = _parse_technology_legend_spatial(
                    page, spans, page_number
                )
                page_entries.extend(spatial_tech)

            # Garland numbered / fallback line-order legends.
            lines = [_clean(line) for line in text.splitlines()]
            lines = [line for line in lines if line]
            page_entries.extend(
                _parse_legend_page(
                    lines,
                    page_number,
                    skip_technology=bool(spatial_tech),
                    page=page,
                    spans=spans,
                )
            )

            # Architect cover: SYMBOLS (drafting / view). Abbreviations are
            # not counted as symbols.
            if _page_has_symbols_section(text):
                page_entries.extend(
                    _parse_symbols_section(spans, page_number, page=page)
                )

            # Architect plan/ceiling legends (glyph left of description).
            if _page_has_plan_legend(text):
                page_entries.extend(
                    _parse_plan_legend_sections(page, spans, page_number)
                )

            if page_entries:
                info.legend_pages.append(page_number)
                info.entries.extend(page_entries)
    finally:
        document.close()

    info.entries = _dedupe_entries(info.entries)
    info.legend_pages = sorted(set(info.legend_pages))
    if not info.entries:
        info.notes.append(
            "No symbol legend or drafting symbols table was found."
        )
    return info


def _page_has_legend(text: str) -> bool:
    upper = text.upper()
    if "TECHNOLOGY SYMBOL LEGEND" in upper:
        return True
    if re.search(r"\bSYMBOL\s+LEGEND\b", upper):
        return True
    if _page_has_symbols_section(text):
        return True
    if _page_has_plan_legend(text):
        return True
    return False


def _page_has_plan_legend(text: str) -> bool:
    """True when a sheet has LEGEND - FLOOR PLAN / LEGEND - CEILING (not notes)."""
    return bool(
        re.search(
            r"(?m)^\s*LEGEND\s*[-–—]\s*(FLOOR\s+PLAN|CEILING)"
            r"(?:\s+DEMOLITION)?\s*$",
            text,
            re.IGNORECASE,
        )
    )


def _page_has_symbols_section(text: str) -> bool:
    upper = text.upper()
    # Require the drafting/view sheet (not a stray "SYMBOLS" word elsewhere).
    if not re.search(r"(?m)^\s*SYMBOLS\s*$", text, re.IGNORECASE):
        if "\nSYMBOLS\n" not in f"\n{upper}\n" and "SYMBOLS" not in upper:
            return False
        # Fallback: heading present with drafting column.
        if "SYMBOLS" not in upper:
            return False
    return "DRAFTING ITEMS" in upper or "VIEW REFERENCES" in upper


def _page_has_abbreviations_section(text: str) -> bool:
    # Require a standalone heading — not a buried sheet-index word.
    if not re.search(r"(?m)^\s*ABBREVIATIONS\s*$", text):
        return False
    upper = text.upper()
    hits = sum(
        1
        for token in ("AFF", "CMU", "NTS", "CONC", "BLDG", "NIC", "ABOVE FINISH")
        if token in upper
    )
    return hits >= 2


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _page_text_spans(page: fitz.Page) -> list[_TextSpan]:
    spans: list[_TextSpan] = []
    data = page.get_text("dict") or {}
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            spans.append(
                _TextSpan(x0=x0, y0=y0, x1=x1, y1=y1, text=_clean(text))
            )
    return spans


def _find_heading_y(spans: list[_TextSpan], pattern: re.Pattern[str]) -> float | None:
    for span in spans:
        if pattern.match(span.text):
            return span.y0
    return None


def _parse_symbols_section(
    spans: list[_TextSpan],
    page_number: int,
    *,
    page: fitz.Page | None = None,
) -> list[SymbolEntry]:
    """Parse DRAFTING ITEMS + VIEW REFERENCES labels under SYMBOLS."""
    symbols_y = _find_heading_y(spans, SYMBOLS_SECTION_HEADING_RE)
    if symbols_y is None:
        return []

    abbrev_y = _find_heading_y(spans, ABBREVIATIONS_HEADING_RE)
    y_max = abbrev_y if abbrev_y is not None else symbols_y + 700

    drafting_headers = [
        s for s in spans if DRAFTING_HEADER_RE.match(s.text) and s.y0 >= symbols_y - 5
    ]
    view_headers = [
        s for s in spans if VIEW_HEADER_RE.match(s.text) and s.y0 >= symbols_y - 5
    ]
    if not drafting_headers and not view_headers:
        return []

    # Column split: drafting labels left of view header; view labels near it.
    view_x = min((s.x0 for s in view_headers), default=400.0)
    drafting_x_max = view_x - 20

    drafting_spans = [
        s
        for s in spans
        if symbols_y < s.y0 < y_max
        and s.x0 < drafting_x_max
        and s.x0 >= 140
        and not DRAFTING_HEADER_RE.match(s.text)
        and not SYMBOLS_SECTION_HEADING_RE.match(s.text)
        and not _is_symbols_noise(s.text)
    ]
    view_spans = [
        s
        for s in spans
        if symbols_y < s.y0 < y_max
        and s.x0 >= view_x - 5
        and s.x0 < view_x + 200
        and not VIEW_HEADER_RE.match(s.text)
        and not _is_symbols_noise(s.text)
    ]

    # Prefer description-column text (right of the glyph) over embedded glyph text.
    # Keep left-anchored full labels like "MATCHLINE - SEE: …" that sit under the glyph.
    drafting_labels = _prefer_description_column(
        drafting_spans, min_x=240, keep_if=_is_full_drafting_label
    )
    view_labels = _prefer_description_column(view_spans, min_x=view_x)

    entries: list[SymbolEntry] = []
    drafting_entries: list[SymbolEntry] = []
    view_entries: list[SymbolEntry] = []
    for label, anchor in _merge_label_runs_with_anchors(drafting_labels):
        drafting_entries.append(
            SymbolEntry(
                description=label,
                source_page=page_number,
                legend_title="SYMBOLS — Drafting Items",
                row_x=anchor.x0,
            )
        )
    for label, anchor in _merge_label_runs_with_anchors(view_labels):
        view_entries.append(
            SymbolEntry(
                description=label,
                source_page=page_number,
                legend_title="SYMBOLS — View References",
                row_x=anchor.x0,
            )
        )
    if page is not None:
        drafting_ceiling = None
        if drafting_headers:
            drafting_ceiling = min(s.y1 for s in drafting_headers) + 2
        view_ceiling = None
        if view_headers:
            view_ceiling = min(s.y1 for s in view_headers) + 2
        view_floor = (view_x - 155.0) if view_headers else None
        _attach_description_left_glyphs(
            page,
            drafting_entries,
            glyph_width=120.0,
            x_gap=2.0,
            x_left_floor=148.0,
            y_ceiling=drafting_ceiling,
            vertical_half_span=52.0,
            isolate_bands=True,
            matchline_above=True,
        )
        _attach_description_left_glyphs(
            page,
            view_entries,
            glyph_width=150.0,
            x_gap=2.0,
            x_left_floor=view_floor,
            y_ceiling=view_ceiling,
            vertical_half_span=55.0,
            isolate_bands=True,
            matchline_above=False,
        )
    entries.extend(drafting_entries)
    entries.extend(view_entries)
    return entries


def _parse_plan_legend_sections(
    page: fitz.Page,
    spans: list[_TextSpan],
    page_number: int,
) -> list[SymbolEntry]:
    """Parse Architect LEGEND - FLOOR PLAN / LEGEND - CEILING glyph+description rows."""
    headings = [
        s for s in spans if PLAN_LEGEND_HEADING_RE.match(s.text)
    ]
    if not headings:
        return []

    entries: list[SymbolEntry] = []
    for heading in headings:
        title = re.sub(r"\s+", " ", heading.text).strip().upper()
        # Stops must sit in the same right-margin legend column — ignore title-block
        # stamps (PROFESSIONAL STAMP / REVISIONS) further to the right.
        band_x1 = heading.x0 + 450
        stop_ys = sorted(
            s.y0
            for s in spans
            if s.y0 > heading.y0 + 8
            and s.x0 >= heading.x0 - 30
            and s.x0 < band_x1
            and (
                PLAN_LEGEND_STOP_RE.match(s.text)
                or PLAN_LEGEND_HEADING_RE.match(s.text)
            )
        )
        y_max = stop_ys[0] if stop_ys else heading.y0 + 650
        # Description column sits right of the glyph strip under the heading.
        band = [
            s
            for s in spans
            if heading.y1 + 2 < s.y0 < y_max
            and s.x0 >= heading.x0 - 10
            and s.x0 < band_x1
            and not PLAN_LEGEND_HEADING_RE.match(s.text)
            and not PLAN_LEGEND_STOP_RE.match(s.text)
        ]
        if not band:
            continue

        # Descriptions cluster ~120–150pt right of the heading left edge.
        desc_min_x = heading.x0 + 90
        desc_spans = [
            s
            for s in band
            if s.x0 >= desc_min_x
            and s.x0 < heading.x0 + 360
            and not PLAN_GLYPH_TEXT_RE.match(s.text)
            and not _is_plan_legend_noise(s.text)
        ]
        # Drop title-block / stamp bleed (far right of the legend).
        if desc_spans:
            xs = sorted(s.x0 for s in desc_spans)
            median_x = xs[len(xs) // 2]
            desc_spans = [s for s in desc_spans if abs(s.x0 - median_x) < 80]

        labeled = _merge_plan_legend_labels(
            desc_spans,
            page=page,
            glyph_x0=heading.x0 - 8,
            glyph_x1=desc_min_x - 2,
        )
        start = len(entries)
        for label, anchor in labeled:
            entries.append(
                SymbolEntry(
                    description=label,
                    source_page=page_number,
                    legend_title=title,
                    row_x=anchor.x0,
                )
            )
        _attach_description_left_glyphs(
            page,
            entries[start:],
            glyph_width=140.0,
            x_gap=4.0,
            x_left_floor=heading.x0 - 8,
            y_ceiling=heading.y1 + 2,
            vertical_half_span=40.0,
            isolate_bands=True,
        )
    return entries


def _is_plan_legend_noise(text: str) -> bool:
    upper = text.upper().strip()
    if not upper or len(upper) < 3:
        return True
    if upper.startswith("NOTE:"):
        return True
    if "AUTODESK" in upper or "://" in text:
        return True
    if re.fullmatch(r"\d{1,3}", upper):
        return True
    if re.fullmatch(r"\d?[A-Z]-?\d{2,4}", upper):
        return True  # sheet refs like 2A-922
    if upper in {"NO.", "DATE", "DESCRIPTION", "TEL"}:
        return True
    return False


def _glyph_strip_has_ink(
    page: fitz.Page,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> bool:
    """True when the legend glyph strip at this row has dark CAD ink."""
    left, right = (x0, x1) if x0 <= x1 else (x1, x0)
    top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
    clip = fitz.Rect(left, top, right, bottom) & page.rect
    if clip.is_empty or clip.width < 3 or clip.height < 2:
        return False
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False
        )
    except Exception:  # noqa: BLE001
        return True
    return not _pixmap_mostly_blank(pix, min_dark_pixels=6)


def _merge_plan_legend_labels(
    spans: list[_TextSpan],
    *,
    page: fitz.Page | None = None,
    glyph_x0: float | None = None,
    glyph_x1: float | None = None,
) -> list[tuple[str, _TextSpan]]:
    """Join wrapped description lines; keep separate meanings as separate rows."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.y0, s.x0))
    out: list[tuple[str, _TextSpan]] = []
    current_parts: list[str] = []
    anchor: _TextSpan | None = None
    last_y: float | None = None

    def flush() -> None:
        nonlocal current_parts, anchor
        if current_parts and anchor is not None:
            label = re.sub(r"\s+", " ", " ".join(current_parts)).strip(" ,")
            if len(label) >= 4:
                out.append((label, anchor))
        current_parts = []
        anchor = None

    def wrap_without_glyph(span: _TextSpan) -> bool:
        if page is None or glyph_x0 is None or glyph_x1 is None:
            return False
        pad = 2.0
        return not _glyph_strip_has_ink(
            page,
            glyph_x0,
            glyph_x1,
            span.y0 - pad,
            span.y1 + pad,
        )

    for span in ordered:
        text = span.text
        is_cont = bool(PLAN_DESC_CONTINUATION_RE.match(text))
        close = last_y is not None and (span.y0 - last_y) < 22
        new_item = text.upper().startswith(
            ("(E)", "(N)", "NEW ", "EXISTING ", "ROOM ", "ROLLER ", "CEILING ")
        )
        if current_parts and is_cont and close and not new_item:
            current_parts.append(text)
        elif current_parts and close and text[:1].islower() and not new_item:
            current_parts.append(text)
        elif current_parts and close and not new_item and wrap_without_glyph(span):
            current_parts.append(text)
        else:
            flush()
            current_parts = [text]
            anchor = span
        last_y = span.y0
    flush()
    return out


def _merge_label_runs_with_anchors(
    spans: list[_TextSpan],
) -> list[tuple[str, _TextSpan]]:
    """Like ``_merge_label_runs`` but keeps the first span as a crop anchor."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.y0, s.x0))
    out: list[tuple[str, _TextSpan]] = []
    current: list[str] = []
    anchor: _TextSpan | None = None
    last_y: float | None = None
    last_x: float | None = None

    def flush() -> None:
        nonlocal current, anchor
        if current and anchor is not None:
            label = re.sub(r"\s+", " ", " ".join(current)).strip(" ,")
            if not (len(label) < 4 and not re.search(r"[A-Za-z]{3,}", label)):
                out.append((label, anchor))
        current = []
        anchor = None

    for span in ordered:
        text = span.text
        if DRAFTING_HEADER_RE.match(text) or VIEW_HEADER_RE.match(text):
            continue
        is_continuation = bool(CONTINUATION_START_RE.match(text))
        close_y = last_y is not None and abs(span.y0 - last_y) < 22
        same_col = last_x is not None and abs(span.x0 - last_x) < 40

        if current and is_continuation and close_y and same_col:
            current.append(text)
        elif current and close_y and same_col and len(text.split()) <= 3 and (
            text.isupper() or CONTINUATION_START_RE.match(text)
        ):
            current.append(text)
        else:
            flush()
            current = [text]
            anchor = span
        last_y = span.y0
        last_x = span.x0
    flush()
    return out


def _attach_description_left_glyphs(
    page: fitz.Page,
    entries: list[SymbolEntry],
    *,
    glyph_width: float,
    x_gap: float,
    x_left_floor: float | None = None,
    y_ceiling: float | None = None,
    vertical_half_span: float = 28.0,
    isolate_bands: bool = False,
    matchline_above: bool = False,
) -> None:
    """Crop CAD glyphs sitting immediately left of each description row."""
    if not entries:
        return
    # Only rows that still need an image and have an x anchor from parsing.
    indexed = [
        (index, entry)
        for index, entry in enumerate(entries)
        if entry.row_x is not None and not entry.symbol_image_png
    ]
    if not indexed:
        return

    # Recover visual Y for each description via text search on the page.
    anchors: list[tuple[int, fitz.Rect, SymbolEntry]] = []
    for index, entry in indexed:
        needle = (entry.description or "").split(",")[0].strip()
        if len(needle) < 4:
            needle = (entry.description or "")[:40]
        hits = page.search_for(needle[:48]) if needle else []
        # Prefer hit near the stored description-column x.
        target_x = entry.row_x or 0.0
        best: fitz.Rect | None = None
        best_score = 1e18
        for hit in hits:
            visual = _text_to_visual_rect(page, hit)
            rect = fitz.Rect(
                min(visual.x0, visual.x1),
                min(visual.y0, visual.y1),
                max(visual.x0, visual.x1),
                max(visual.y0, visual.y1),
            )
            score = abs(rect.x0 - target_x) + 0.05 * abs(rect.y0)
            # Prefer description-column hits (not glyph-embedded sample text).
            if rect.x0 + 20 < target_x:
                score += 80
            if score < best_score:
                best_score = score
                best = rect
        if best is None:
            continue
        anchors.append((index, best, entry))

    if not anchors:
        return

    anchors.sort(key=lambda item: item[1].y0)
    for pos, (index, rect, entry) in enumerate(anchors):
        # MATCHLINE: solid wedge sits *above* the label, not left of it.
        if matchline_above and entry.description.upper().startswith("MATCHLINE"):
            clip = fitz.Rect(
                rect.x0 - 5,
                rect.y0 - 28,
                rect.x1 + 55,
                rect.y0 - 1.5,
            ) & page.rect
            png = _pixmap_to_symbol_png(
                page,
                clip,
                isolate_bands=True,
                prefer_y_frac=0.65,
            )
            if png:
                entries[index].symbol_image_png = png
            continue

        desc_mid = (rect.y0 + rect.y1) / 2.0
        y_lo = desc_mid - vertical_half_span
        y_hi = desc_mid + vertical_half_span
        # Soft clamp to neighbors — tiny overlap by default; allow more room when
        # the previous row is far away (tall glyphs like north arrow need it).
        if pos > 0:
            prev = anchors[pos - 1][1]
            mid_lo = (prev.y1 + rect.y0) / 2.0
            gap_to_prev = max(0.0, rect.y0 - prev.y1)
            slack = 3.0 if gap_to_prev < 45 else min(18.0, gap_to_prev * 0.28)
            y_lo = max(y_lo, mid_lo - slack)
        else:
            # First row: do not swallow the column header (DRAFTING ITEMS / VIEW…).
            y_lo = max(y_lo, rect.y0 - 36.0)
        if pos + 1 < len(anchors):
            nxt = anchors[pos + 1][1]
            mid_hi = (rect.y1 + nxt.y0) / 2.0
            gap_to_next = max(0.0, nxt.y0 - rect.y1)
            slack = 3.0 if gap_to_next < 45 else min(18.0, gap_to_next * 0.28)
            y_hi = min(y_hi, mid_hi + slack)
        y_lo = min(y_lo, rect.y0 - 4.0)
        y_hi = max(y_hi, rect.y1 + 4.0)
        if y_ceiling is not None:
            y_lo = max(y_lo, y_ceiling)

        x_right = rect.x0 - x_gap
        x_left = x_right - glyph_width
        if x_left_floor is not None:
            x_left = max(x_left, x_left_floor)
        clip = fitz.Rect(x_left, y_lo, x_right, y_hi) & page.rect
        if clip.is_empty or clip.width < 4 or clip.height < 3:
            continue
        prefer = 0.5
        if y_hi > y_lo:
            prefer = (desc_mid - y_lo) / (y_hi - y_lo)
            prefer = max(0.15, min(0.85, prefer))
        bound_y0 = y_lo
        bound_y1 = y_hi
        if pos > 0:
            prev = anchors[pos - 1][1]
            bound_y0 = min(bound_y0, prev.y1 + 3.0)
        else:
            bound_y0 = min(bound_y0, rect.y0 - 56.0)
        if pos + 1 < len(anchors):
            nxt = anchors[pos + 1][1]
            bound_y1 = max(bound_y1, nxt.y0 - 3.0)
        else:
            bound_y1 = max(bound_y1, rect.y1 + 56.0)
        if y_ceiling is not None:
            bound_y0 = max(bound_y0, y_ceiling)
        bound_x0 = x_left_floor if x_left_floor is not None else x_left - 24.0
        expand_bounds = fitz.Rect(bound_x0, bound_y0, x_right, bound_y1) & page.rect
        png = _pixmap_to_symbol_png(
            page,
            clip,
            isolate_bands=isolate_bands,
            prefer_y_frac=prefer,
            expand_bounds=expand_bounds,
        )
        if png:
            entries[index].symbol_image_png = png


def _image_edge_ink(
    img: PILImage.Image,
    *,
    edge_px: int = 3,
    thresh: int = 245,
) -> tuple[bool, bool, bool, bool]:
    """True when dark ink touches left, top, right, bottom crop borders."""
    gray = img.convert("L")
    pixels = gray.load()
    width, height = img.size
    if width < 2 or height < 2:
        return False, False, False, False
    edge = max(1, min(edge_px, width // 4, height // 4))

    def _dark(x: int, y: int) -> bool:
        return pixels[x, y] < thresh

    def _col_frac(x: int) -> float:
        return sum(1 for y in range(height) if _dark(x, y)) / height

    def _row_frac(y: int) -> float:
        return sum(1 for x in range(width) if _dark(x, y)) / width

    # Full-span table rules sit on cell borders; they are not clipped glyph ink.
    left = any(_dark(x, y) for x in range(edge) for y in range(height))
    if max(_col_frac(x) for x in range(edge)) >= 0.72:
        left = False
    right = any(_dark(x, y) for x in range(width - edge, width) for y in range(height))
    if max(_col_frac(x) for x in range(width - edge, width)) >= 0.72:
        right = False
    top = any(_dark(x, y) for y in range(edge) for x in range(width))
    if max(_row_frac(y) for y in range(edge)) >= 0.72:
        top = False
    bottom = any(
        _dark(x, y) for y in range(height - edge, height) for x in range(width)
    )
    if max(_row_frac(y) for y in range(height - edge, height)) >= 0.72:
        bottom = False
    return left, top, right, bottom


def _expand_clip_for_edge_ink(
    page: fitz.Page,
    clip: fitz.Rect,
    *,
    bounds: fitz.Rect | None = None,
    matrix_scale: float = 4.0,
    max_expand_pt: float = 32.0,
    step_pt: float = 8.0,
) -> fitz.Rect:
    """Grow clip while ink is cut by a border, staying inside bounds."""
    expanded = fitz.Rect(clip)
    limit = (bounds & page.rect) if bounds is not None else page.rect
    if limit.is_empty:
        return expanded
    grown = 0.0
    while grown < max_expand_pt:
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(matrix_scale, matrix_scale),
                clip=expanded,
                alpha=False,
            )
        except Exception:  # noqa: BLE001
            break
        if pix.width < 4 or pix.height < 4:
            break
        try:
            img = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        except Exception:  # noqa: BLE001
            break
        left, top, right, bottom = _image_edge_ink(img)
        if not (left or top or right or bottom):
            break
        next_clip = fitz.Rect(expanded)
        if left:
            next_clip.x0 -= step_pt
        if right:
            next_clip.x1 += step_pt
        if top:
            next_clip.y0 -= step_pt
        if bottom:
            next_clip.y1 += step_pt
        next_clip = next_clip & limit
        if (
            abs(next_clip.x0 - expanded.x0) < 0.4
            and abs(next_clip.y0 - expanded.y0) < 0.4
            and abs(next_clip.x1 - expanded.x1) < 0.4
            and abs(next_clip.y1 - expanded.y1) < 0.4
        ):
            break
        expanded = next_clip
        grown += step_pt
    return expanded


def _pixmap_to_symbol_png(
    page: fitz.Page,
    clip: fitz.Rect,
    *,
    isolate_bands: bool,
    prefer_y_frac: float,
    expand_bounds: fitz.Rect | None = None,
) -> bytes | None:
    if expand_bounds is not None:
        clip = _expand_clip_for_edge_ink(
            page, clip, bounds=expand_bounds, matrix_scale=4.0
        )
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(4.0, 4.0), clip=clip, alpha=False
        )
    except Exception:  # noqa: BLE001
        return None
    if pix.width < 4 or pix.height < 4 or _pixmap_mostly_blank(pix):
        return None
    try:
        img = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception:  # noqa: BLE001
        return _trim_pixmap_png(
            pix,
            margin=12,
            strip_grid=False,
            erase_separated_rules=True,
        )
    img = _erase_separated_edge_rules(img)
    if isolate_bands:
        img = _isolate_primary_ink_band(img, prefer_y_frac=prefer_y_frac)
    bbox = _ink_bbox(img)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    margin = 12
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    cropped = img.crop((left, top, right, bottom))
    if cropped.width < 3 or cropped.height < 3:
        return None
    png = _png_bytes(_pad_symbol_cell(cropped, min_w=120, min_h=72))
    if png and not _png_is_thin_vertical_rule(png):
        return png
    return None


def _isolate_primary_ink_band(
    img: PILImage.Image,
    *,
    prefer_y_frac: float = 0.5,
) -> PILImage.Image:
    """Keep the ink band nearest the description center; drop neighbor-row bleed."""
    width, height = img.width, img.height
    if height < 12 or width < 8:
        return img
    gray = img.convert("L")
    pixels = gray.load()

    def _col_frac(x: int) -> float:
        return sum(1 for y in range(height) if pixels[x, y] < 245) / height

    # Full-height borders/leaders bridge every row into one fake band — ignore them.
    rule_cols = {x for x in range(width) if _col_frac(x) >= 0.50}
    x0 = max(2, int(width * 0.04))
    x1 = min(width - 1, int(width * 0.99))
    usable = [x for x in range(x0, x1) if x not in rule_cols]
    if len(usable) < 8:
        usable = list(range(x0, x1))

    def _row_frac(y: int) -> float:
        return sum(1 for x in usable if pixels[x, y] < 245) / len(usable)

    ink_rows = [y for y in range(height) if _row_frac(y) >= 0.012]
    if not ink_rows:
        return img

    # Strict clusters first (do not join neighbor legend rows).
    bands: list[tuple[int, int]] = []
    start = prev = ink_rows[0]
    for row in ink_rows[1:]:
        if row <= prev + 6:
            prev = row
            continue
        bands.append((start, prev))
        start = prev = row
    bands.append((start, prev))

    target = prefer_y_frac * max(height - 1, 1)

    containing = [
        band for band in bands if band[0] - 1 <= target <= band[1] + 1
    ]
    if containing:
        best = max(containing, key=lambda b: b[1] - b[0])
    else:

        def _score(band: tuple[int, int]) -> tuple[float, float]:
            top, bot = band
            mid = (top + bot) / 2.0
            band_h = bot - top + 1
            return (abs(mid - target), -band_h)

        best = min(bands, key=_score)

    # Pull in nearby stacked *label* bands only (e.g. PROJECT/NORTH above the
    # arrow). Never absorb another full glyph row — that caused cloud/grid bleed.
    top, bot = best
    primary_h = max(1, bot - top + 1)
    changed = True
    while changed:
        changed = False
        for band in bands:
            b0, b1 = band
            band_h = b1 - b0 + 1
            # Short band immediately above = caption text belonging to this glyph.
            if (
                b1 < top
                and (top - b1) <= 52
                and band_h <= 70
                and band_h <= max(70, primary_h * 0.85)
            ):
                top = b0
                primary_h = bot - top + 1
                changed = True
            # Continuation immediately below (e.g. lower half of a ceiling-height
            # pennant). Keep the gap small so the next legend row stays out.
            elif (
                b0 > bot
                and (b0 - bot) <= 40
                and band_h <= 70
                and band_h <= max(70, primary_h * 0.85)
            ):
                bot = b1
                primary_h = bot - top + 1
                changed = True

    pad = 16
    top = max(0, top - pad)
    bot = min(height, bot + pad + 1)
    if bot - top < 6:
        return img
    cropped = img.crop((0, top, width, bot))
    # Also drop full-height rule columns from the kept band.
    if rule_cols:
        cropped = _erase_separated_edge_rules(cropped)
    return cropped


def _is_symbols_noise(text: str) -> bool:
    if SYMBOLS_SKIP_LABEL_RE.match(text):
        return True
    if re.fullmatch(r"\d+", text):
        return True
    if re.fullmatch(r"A\d{0,3}", text, re.I):
        return True
    if re.fullmatch(r"1A", text, re.I):
        return True
    # Autodesk path / notes bleed.
    if "AUTODESK" in text.upper() or "://" in text:
        return True
    if text.upper().startswith("GENERAL PROJECT"):
        return True
    return False


def _is_full_drafting_label(text: str) -> bool:
    upper = text.upper()
    return (
        upper.startswith("MATCHLINE")
        or " - " in text
        or len(text.split()) >= 3
    )


def _prefer_description_column(
    spans: list[_TextSpan],
    *,
    min_x: float,
    keep_if=None,
) -> list[_TextSpan]:
    """Keep right-hand description text; drop glyph-embedded leftovers when both exist."""
    preferred = [
        s
        for s in spans
        if s.x0 >= min_x or (keep_if is not None and keep_if(s.text))
    ]
    if preferred:
        return preferred
    return spans


def _merge_label_runs(spans: list[_TextSpan]) -> list[str]:
    """Join stacked description lines (e.g. OUTLINE… / N.I.C.… / MATERIAL)."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.y0, s.x0))
    labels: list[str] = []
    current: list[str] = []
    last_y: float | None = None
    last_x: float | None = None

    def flush() -> None:
        nonlocal current
        if current:
            labels.append(" ".join(current))
        current = []

    for span in ordered:
        text = span.text
        if DRAFTING_HEADER_RE.match(text) or VIEW_HEADER_RE.match(text):
            continue
        is_continuation = bool(CONTINUATION_START_RE.match(text))
        close_y = last_y is not None and abs(span.y0 - last_y) < 22
        same_col = last_x is not None and abs(span.x0 - last_x) < 40

        if current and is_continuation and close_y and same_col:
            current.append(text)
        elif current and close_y and same_col and len(text.split()) <= 3 and (
            text.isupper() or CONTINUATION_START_RE.match(text)
        ):
            # Short uppercase fragments stacked under a label.
            current.append(text)
        else:
            flush()
            current = [text]
        last_y = span.y0
        last_x = span.x0
    flush()

    # Drop ultra-short leftovers (lone "1", "A", etc.) that slipped through.
    cleaned: list[str] = []
    for label in labels:
        if len(label) < 4 and not re.search(r"[A-Za-z]{3,}", label):
            continue
        cleaned.append(re.sub(r"\s+", " ", label).strip(" ,"))
    return cleaned


def _parse_abbreviations_section(
    spans: list[_TextSpan], page_number: int
) -> list[SymbolEntry]:
    """Parse two-column ABBREVIATIONS lists (abbr | definition)."""
    abbrev_y = _find_heading_y(spans, ABBREVIATIONS_HEADING_RE)
    if abbrev_y is None:
        return []

    # Stay in the left legend band; project notes sit further right (~675+).
    band = [
        s
        for s in spans
        if s.y0 > abbrev_y + 5 and s.x0 < 620
    ]
    if len(band) < 8:
        return []

    # Cluster into left and right abbreviation pairs by x.
    left_abbr_x = 190.0
    mid_x = 360.0
    entries: list[SymbolEntry] = []
    entries.extend(
        _pair_abbreviation_column(
            [s for s in band if s.x0 < mid_x],
            abbr_max_x=left_abbr_x,
            page_number=page_number,
        )
    )
    entries.extend(
        _pair_abbreviation_column(
            [s for s in band if s.x0 >= mid_x],
            abbr_max_x=mid_x + 70,
            page_number=page_number,
        )
    )
    # Reject weak / mis-detected sheets (notes mistaken for abbr columns).
    solid = [e for e in entries if e.symbol and e.symbol != "—"]
    if len(solid) < 12:
        return []
    return entries


def _pair_abbreviation_column(
    spans: list[_TextSpan],
    *,
    abbr_max_x: float,
    page_number: int,
) -> list[SymbolEntry]:
    """Within one pair-column, match abbr (left) to definition (right) by Y."""
    if not spans:
        return []

    abbrs = [s for s in spans if s.x0 < abbr_max_x]
    defs = [s for s in spans if s.x0 >= abbr_max_x]
    if not defs:
        return []

    # Bucket by approximate row Y.
    rows: dict[int, dict[str, list[str]]] = {}
    for span in abbrs:
        key = int(round(span.y0 / 5.0) * 5)
        rows.setdefault(key, {"abbr": [], "def": []})
        rows[key]["abbr"].append(span.text)
    for span in defs:
        key = int(round(span.y0 / 5.0) * 5)
        # Snap to nearest existing row within 8pt if abbr already placed.
        nearest = min(rows.keys(), key=lambda y: abs(y - key), default=None)
        if nearest is not None and abs(nearest - key) <= 8:
            key = nearest
        rows.setdefault(key, {"abbr": [], "def": []})
        rows[key]["def"].append(span.text)

    entries: list[SymbolEntry] = []
    for key in sorted(rows):
        abbr = _clean(" ".join(rows[key]["abbr"]))
        definition = _clean(" ".join(rows[key]["def"]))
        if not definition:
            continue
        # Symbol-only rows (⊥, ∠) may have empty abbr in the text layer.
        if not abbr:
            abbr = "—"
        if _is_bad_abbreviation_pair(abbr, definition):
            continue
        entries.append(
            SymbolEntry(
                symbol=abbr,
                description=definition,
                source_page=page_number,
                legend_title="ABBREVIATIONS",
            )
        )
    return entries


_ABBR_TOKEN_RE = re.compile(
    r"^[@#]|[A-Za-z0-9./()#&-]{1,14}$"
)


def _is_bad_abbreviation_pair(abbr: str, definition: str) -> bool:
    upper_def = definition.upper()
    upper_abbr = abbr.upper()
    if upper_def.startswith("GENERAL PROJECT"):
        return True
    if len(definition) < 2:
        return True
    if ABBREVIATIONS_HEADING_RE.match(abbr) or ABBREVIATIONS_HEADING_RE.match(
        definition
    ):
        return True
    # Sheet-index / title-block / note bleed.
    if len(definition) > 55:
        return True
    if definition.count(".") >= 1 and len(definition) > 35:
        return True
    if len(abbr) > 14:
        return True
    if abbr != "—" and not _ABBR_TOKEN_RE.match(abbr):
        return True
    # Glyph-only rows: keep short dictionary words, drop note paragraphs.
    if abbr == "—":
        if len(definition) > 28 or definition.count(" ") > 3:
            return True
        if re.search(r"\d", definition):
            return True
        if upper_def in {"DRAWING INDEX", "PLUMBING", "MECHANICAL &"}:
            return True
    if any(
        token in upper_abbr
        for token in ("PLAN", "SCHEDULE", "DATE", "DRAWN", "SHEET", "TITLE")
    ):
        return True
    if any(
        token in upper_def
        for token in (
            "DISCLAIMS",
            "GENERAL INFORMATION",
            "FIRE ALARM",
            "LIGHTING PLAN",
            "PROJECT OWNER",
            "CONTRACTOR SHALL",
            "STRUCTURAL DRAWING",
            "SHEET TITLE",
            "MEANS, METHODS",
        )
    ):
        return True
    if re.search(r"\b\d[A-Z]\d{2,4}\b", upper_abbr):
        return True
    return False


_TECH_HEADER_RE = re.compile(
    r"^(TECHNOLOGY\s+SYMBOL\s+LEGEND:?|SYMBOL|DESCRIPTION|MFG/?MODEL|"
    r"PART(?:\s+NUMBER)?|NUMBER|NOTES(?:\s*/\s*DETAIL)?|REFERENCES|"
    r"ALL EQUIPMENT AND MATERIALS.*)$",
    re.IGNORECASE,
)
_TECH_DETAIL_MARK_RE = re.compile(r"^(T-\d+|\d{1,2})$", re.IGNORECASE)
_TECH_SUBROW_RE = re.compile(
    r"^(MOUNT|NETWORK\s+CAMERA|#\s*=\s*QTY)\b",
    re.IGNORECASE,
)
ROW_X_GAP = 15.0
ROW_X_ASSIGN = 14.0


def _parse_technology_legend_spatial(
    page: fitz.Page,
    spans: list[_TextSpan],
    page_number: int,
) -> list[SymbolEntry]:
    """Parse rotated TECHNOLOGY SYMBOL LEGEND via X-rows / Y-columns."""
    heading = next(
        (
            s
            for s in spans
            if re.match(r"^TECHNOLOGY\s+SYMBOL\s+LEGEND", s.text, re.I)
        ),
        None,
    )
    if heading is None:
        return []

    headers = _tech_column_headers(spans, heading)
    if "DESCRIPTION" not in headers or "SYMBOL" not in headers:
        return []

    bands = _tech_column_bands(headers)
    # Legend body sits left of the heading on these sheets (decreasing X).
    x_max = heading.x0 + 40
    x_min = min((s.x0 for s in spans if s.x0 < x_max), default=0) - 10
    y_min = bands["symbol"][0] - 30
    y_max = bands["notes"][1] + 40

    body = [
        s
        for s in spans
        if x_min <= s.x0 <= x_max
        and y_min <= s.y0 <= y_max
        and not _TECH_HEADER_RE.match(s.text)
        and not HEADER_NOTE_RE.search(s.text)
    ]
    if not body:
        return []

    desc_spans = [
        s
        for s in body
        if bands["description"][0] <= s.y0 < bands["description"][1]
        and _looks_like_tech_description(s.text)
    ]
    if len(desc_spans) < 3:
        return []

    row_centers = _cluster_row_centers([s.x0 for s in desc_spans], ROW_X_GAP)
    if not row_centers:
        return []

    # Visual top → bottom on sheet == descending X (unrotated text space).
    row_centers = sorted(row_centers, reverse=True)
    # Symbol / description column edges in *visual* (pixmap) coordinates.
    visual_cols = _tech_visual_column_bounds(page, headers, bands)

    entries: list[SymbolEntry] = []
    for index, center in enumerate(row_centers):
        row_spans = [s for s in body if abs(s.x0 - center) <= ROW_X_ASSIGN]
        if not row_spans:
            continue
        packed = _pack_tech_row(row_spans, bands)
        if not packed["description"]:
            continue

        desc = packed["description"] or ""
        symbol = packed["symbol"] or _infer_tech_symbol_tag(desc)
        x_right, x_left = _row_x_bounds(row_centers, index)
        # Prefer a real pixmap crop from the PDF (rotation-corrected).
        desc_visual_left = _description_visual_left(
            page, row_spans, bands["description"]
        )
        image_png = _crop_tech_symbol_cell(
            page,
            row_x_left=x_left,
            row_x_right=x_right,
            bands=bands,
            visual_cols=visual_cols,
            desc_visual_left=desc_visual_left,
        )
        if not image_png:
            image_png = _render_legend_symbol(
                symbol,
                desc,
                part_number=packed.get("part_number"),
            )
        entries.append(
            SymbolEntry(
                description=packed["description"],
                symbol=symbol,
                mfg_model=packed["mfg_model"],
                part_number=packed["part_number"],
                notes=packed["notes"],
                source_page=page_number,
                legend_title="TECHNOLOGY SYMBOL LEGEND",
                symbol_image_png=image_png,
                row_x=center,
            )
        )
    _apply_grouped_legend_symbols(entries, page=page, bands=bands, visual_cols=visual_cols)
    return entries


_INFERRED_TAG_RE = [
    (re.compile(r"^DATA RACK\b", re.I), "R"),
    (re.compile(r"^GROUND BOX\b", re.I), "G"),
    (re.compile(r"^JUNCTION BOX\b", re.I), "J"),
    (re.compile(r"^NETWORK VIDEO RECORDER\b", re.I), "NVR"),
    (re.compile(r"^SIGNAL TERMINAL CABINET\b", re.I), "STC"),
    (re.compile(r"^GROUND BUS BAR\b", re.I), "TGB"),
    (re.compile(r"^INTRUSION ALARM CONTROL PANEL\b", re.I), "IACP"),
    (re.compile(r"^FIRE ALARM CONTROL PANEL\b", re.I), "FACP"),
    (re.compile(r"^MINIMUM POINT OF ENTRY\b", re.I), "MPOE"),
    (re.compile(r"^INTERCOM CONSOLE\b", re.I), "NQ-CON"),
    (re.compile(r"^INTERIOR WIRELESS ACCESS POINT\b", re.I), "AP"),
    (re.compile(r"^DATA PERMANENT LINK\b", re.I), "#"),
]


def _infer_tech_symbol_tag(description: str) -> str | None:
    for pattern, tag in _INFERRED_TAG_RE:
        if pattern.search(description or ""):
            return tag
    if re.search(r"\bEXTERIOR INTERCOM\b", description or "", re.I):
        return "WP"
    return None


def _apply_grouped_legend_symbols(
    entries: list[SymbolEntry],
    *,
    page: fitz.Page | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    visual_cols: dict[str, float] | None = None,
) -> None:
    """Span one tall legend glyph across stacked sub-rows (WAP / camera / MOUNT)."""
    index = 0
    while index < len(entries):
        desc = (entries[index].description or "").upper()
        span = 1
        is_group = False

        if "DATA LOCATION" in desc and "CEILING" in desc:
            is_group = True
            j = index + 1
            while j < len(entries) and j < index + 3:
                nxt = (entries[j].description or "").upper()
                if "WIRELESS ACCESS POINT" in nxt or nxt.startswith("MOUNT"):
                    span = j - index + 1
                    j += 1
                    continue
                break
        elif "WIRELESS ACCESS POINT" in desc:
            is_group = True
            j = index + 1
            while j < len(entries) and j < index + 2:
                if (entries[j].description or "").upper().startswith("MOUNT"):
                    span = j - index + 1
                    j += 1
                    continue
                break
        elif "DATA DROP" in desc and "CAMERA" in desc:
            is_group = True
            j = index + 1
            while j < len(entries) and j < index + 3:
                nxt = (entries[j].description or "").upper()
                if (
                    nxt.startswith("NETWORK CAMERA")
                    or "MULTI-SENSOR" in nxt
                    or nxt.startswith("MOUNT")
                    or nxt.startswith("CAMERA (")
                ):
                    span = j - index + 1
                    j += 1
                    continue
                break

        if is_group and span > 1:
            group_img = _crop_grouped_symbol(
                page,
                entries[index : index + span],
                bands=bands,
                visual_cols=visual_cols,
            )
            if not group_img:
                group_img = entries[index].symbol_image_png
            if group_img:
                for offset in range(span):
                    entries[index + offset].symbol_image_png = group_img
            index += span
            continue
        index += 1


def _text_to_visual_rect(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect:
    """Map unrotated text-extraction coords into pixmap / page.rect space."""
    if not page.rotation:
        return fitz.Rect(rect)
    return fitz.Rect(rect) * page.rotation_matrix


def _tech_visual_column_bounds(
    page: fitz.Page,
    headers: dict[str, float],
    bands: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Unused placeholder kept for call-site compatibility."""
    del page, headers, bands
    return {}


def _description_visual_left(
    page: fitz.Page,
    row_spans: list[_TextSpan],
    desc_band: tuple[float, float],
) -> float | None:
    """Left edge of this row's description text in visual/pixmap space."""
    lo, hi = desc_band
    desc_rects = [
        fitz.Rect(s.x0, s.y0, s.x1, s.y1)
        for s in row_spans
        if lo <= s.y0 < hi and _looks_like_tech_description(s.text)
    ]
    if not desc_rects:
        return None
    visual_lefts = [
        min(r.x0, r.x1)
        for r in (_text_to_visual_rect(page, r) for r in desc_rects)
    ]
    return min(visual_lefts) if visual_lefts else None


def _crop_tech_symbol_cell(
    page: fitz.Page,
    *,
    row_x_left: float,
    row_x_right: float,
    bands: dict[str, tuple[float, float]],
    visual_cols: dict[str, float],
    desc_visual_left: float | None = None,
) -> bytes | None:
    """Crop one legend symbol cell using rotation-correct visual coordinates.

    Text extraction uses unrotated coordinates; ``page.get_pixmap`` clips in
    rotated ``page.rect`` space. Multiply by ``page.rotation_matrix`` so any
    uploaded sheet (0/90/180/270) yields the painted symbol, not a wrong strip.
    """
    del visual_cols
    sym_lo, sym_hi = bands["symbol"]
    # Slight inset from row/column midlines to reduce table-rule bleed.
    pad = 3.0
    text_rect = fitz.Rect(
        min(row_x_left, row_x_right) + pad,
        sym_lo - 1,
        max(row_x_left, row_x_right) - pad,
        sym_hi + 3,
    )
    visual = _text_to_visual_rect(page, text_rect)
    visual = fitz.Rect(
        min(visual.x0, visual.x1),
        min(visual.y0, visual.y1),
        max(visual.x0, visual.x1),
        max(visual.y0, visual.y1),
    )
    # Stop before description text so we don't pull "DA…" / "GR…" into the crop.
    if desc_visual_left is not None and desc_visual_left > visual.x0 + 8:
        visual.x1 = min(visual.x1, desc_visual_left - 3)
    visual = visual & page.rect
    if visual.is_empty or visual.width < 4 or visual.height < 4:
        return None
    # Expand along the symbol *column* (glyph width) but almost not along the
    # row axis — that direction is neighboring legend rows on rotated sheets.
    slack_row = 3.0
    slack_col = 16.0
    bound_text = fitz.Rect(
        min(row_x_left, row_x_right) - slack_row,
        sym_lo - slack_col,
        max(row_x_left, row_x_right) + slack_row,
        sym_hi + slack_col,
    )
    bound_visual = _text_to_visual_rect(page, bound_text)
    bound_visual = fitz.Rect(
        min(bound_visual.x0, bound_visual.x1),
        min(bound_visual.y0, bound_visual.y1),
        max(bound_visual.x0, bound_visual.x1),
        max(bound_visual.y0, bound_visual.y1),
    )
    if desc_visual_left is not None and desc_visual_left > bound_visual.x0 + 8:
        bound_visual.x1 = min(bound_visual.x1, desc_visual_left - 2)
    bound_visual = bound_visual & page.rect
    visual = _expand_clip_for_edge_ink(
        page, visual, bounds=bound_visual, matrix_scale=3.5
    )
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(3.5, 3.5), clip=visual, alpha=False
        )
    except Exception:  # noqa: BLE001
        return None
    if pix.width < 4 or pix.height < 4 or _pixmap_mostly_blank(pix):
        return None
    return _trim_pixmap_png(pix, strip_grid=False, erase_separated_rules=True)


def _trim_pixmap_png(
    pix: fitz.Pixmap,
    margin: int = 10,
    *,
    strip_grid: bool = True,
    erase_separated_rules: bool = False,
) -> bytes | None:
    """Tight-crop symbol ink after optional table-border cleanup."""
    try:
        img = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception:  # noqa: BLE001
        return pix.tobytes("png")
    # Wide tech-legend cells: strip full-cell L-bracket grid lines.
    # Numbered SYMBOL LEGEND glyphs are themselves boxes — never strip their
    # sides as "grid" (that deletes the right/left walls of fan/hatch/curb).
    if strip_grid:
        img = _strip_table_grid_lines(img)
    if erase_separated_rules:
        img = _erase_separated_edge_rules(img)
    bbox = _ink_bbox(img)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    cropped = img.crop((left, top, right, bottom))
    if cropped.width < 3 or cropped.height < 3:
        return None
    return _png_bytes(_pad_symbol_cell(cropped, min_w=120, min_h=72))


def _ink_bbox(img: PILImage.Image) -> tuple[int, int, int, int] | None:
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < 245 else 0)
    return mask.getbbox()


def _pad_symbol_cell(
    img: PILImage.Image, *, min_w: int, min_h: int
) -> PILImage.Image:
    """Center the glyph on a white cell so the PDF Symbol column stays roomy."""
    width = max(img.width, min_w)
    height = max(img.height, min_h)
    if width == img.width and height == img.height:
        return img
    canvas = PILImage.new("RGB", (width, height), (255, 255, 255))
    ox = (width - img.width) // 2
    oy = (height - img.height) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def _erase_separated_edge_rules(img: PILImage.Image) -> PILImage.Image:
    """Drop thin full-height column rules that sit apart from the main glyph.

    Numbered legends often place a table border left of the CAD symbol with a
    white gap between them. Erase only those isolated rules — never a box side
    that is attached to the rest of the glyph.
    """
    width, height = img.width, img.height
    if width < 12 or height < 8:
        return img

    pixels = img.load()

    def _col_frac(x: int) -> float:
        return (
            sum(
                1
                for y in range(height)
                if pixels[x, y][0] < 245
                and pixels[x, y][1] < 245
                and pixels[x, y][2] < 245
            )
            / height
        )

    def _paint_white_band(x0: int, x1: int) -> None:
        for y in range(height):
            for x in range(max(0, x0), min(width, x1)):
                pixels[x, y] = (255, 255, 255)

    def _gap_before_next_ink(start: int) -> int:
        gap = 0
        x = start
        while x < width and _col_frac(x) < 0.12:
            gap += 1
            x += 1
        return gap

    def _gap_after_prev_ink(start: int) -> int:
        gap = 0
        x = start
        while x >= 0 and _col_frac(x) < 0.12:
            gap += 1
            x -= 1
        return gap

    x = 0
    while x < width:
        if _col_frac(x) < 0.75:
            x += 1
            continue
        x1 = x
        while x1 + 1 < width and _col_frac(x1 + 1) >= 0.75:
            x1 += 1
        run_w = x1 - x + 1
        # Thin full-height rule only.
        if run_w <= 5:
            left_edge = x <= max(4, int(width * 0.35))
            right_edge = x1 >= width - max(4, int(width * 0.35)) - 1
            if left_edge and _gap_before_next_ink(x1 + 1) >= 6:
                _paint_white_band(x, x1 + 1)
            elif right_edge and _gap_after_prev_ink(x - 1) >= 6:
                _paint_white_band(x, x1 + 1)
        x = x1 + 1
    return img


def _strip_table_grid_lines(img: PILImage.Image) -> PILImage.Image:
    """Erase full-cell table rules (left/bottom L-brackets), not symbol ink."""
    width, height = img.width, img.height
    if width < 8 or height < 8:
        return img

    pixels = img.load()

    def _is_dark(x: int, y: int) -> bool:
        r, g, b = pixels[x, y]
        return r < 245 and g < 245 and b < 245

    def _paint_white_band(x0: int, y0: int, x1: int, y1: int) -> None:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                pixels[x, y] = (255, 255, 255)

    def _col_frac(x: int) -> float:
        return sum(1 for y in range(height) if _is_dark(x, y)) / height

    def _row_frac(y: int) -> float:
        return sum(1 for x in range(width) if _is_dark(x, y)) / width

    # Table rules sit ~15–20% in from the cell edge on these legends.
    x_edge = max(6, int(width * 0.25))
    y_edge = max(6, int(height * 0.25))

    for x0 in range(0, x_edge):
        if _col_frac(x0) >= 0.55:
            # Clear the whole anti-aliased cluster of the rule.
            x1 = x0
            while x1 + 1 < width and _col_frac(x1 + 1) >= 0.55:
                x1 += 1
            _paint_white_band(x0, 0, x1 + 1, height)
    for x0 in range(width - x_edge, width):
        if _col_frac(x0) >= 0.55:
            x1 = x0
            while x1 - 1 >= 0 and _col_frac(x1 - 1) >= 0.55:
                x1 -= 1
            _paint_white_band(x1, 0, x0 + 1, height)

    for y0 in range(0, y_edge):
        if _row_frac(y0) >= 0.55:
            y1 = y0
            while y1 + 1 < height and _row_frac(y1 + 1) >= 0.55:
                y1 += 1
            _paint_white_band(0, y0, width, y1 + 1)
    for y0 in range(height - y_edge, height):
        if _row_frac(y0) >= 0.55:
            y1 = y0
            while y1 - 1 >= 0 and _row_frac(y1 - 1) >= 0.55:
                y1 -= 1
            _paint_white_band(0, y1, width, y0 + 1)

    return img


def _crop_grouped_symbol(
    page: fitz.Page | None,
    group: list[SymbolEntry],
    *,
    bands: dict[str, tuple[float, float]] | None,
    visual_cols: dict[str, float] | None,
) -> bytes | None:
    """Crop a tall symbol that spans several stacked legend rows."""
    if page is None or bands is None or not group:
        return None
    xs = [e.row_x for e in group if e.row_x is not None]
    if not xs:
        return None
    desc_left: float | None = None
    for entry in group:
        needle = (entry.description or "").split(",")[0].strip()
        if len(needle) < 4:
            needle = (entry.description or "")[:40]
        if not needle:
            continue
        for hit in page.search_for(needle[:48]):
            visual = _text_to_visual_rect(page, hit)
            left = min(visual.x0, visual.x1)
            desc_left = left if desc_left is None else min(desc_left, left)
    return _crop_tech_symbol_cell(
        page,
        row_x_left=min(xs) - 10,
        row_x_right=max(xs) + 10,
        bands=bands,
        visual_cols=visual_cols or {},
        desc_visual_left=desc_left,
    )


def _tech_column_headers(
    spans: list[_TextSpan], heading: _TextSpan
) -> dict[str, float]:
    """Map column name → header y0 near the legend heading.

    Architect-CD sheets repeat labels elsewhere; anchor on SYMBOL then require
    other headers at greater Y in the same vertical strip.
    """
    near = [s for s in spans if abs(s.x0 - heading.x0) <= 90]
    symbol_span = next(
        (s for s in near if s.text.upper().strip() == "SYMBOL" and s.y0 > 1000),
        None,
    )
    if symbol_span is None:
        symbol_span = next(
            (s for s in near if s.text.upper().strip() == "SYMBOL"),
            None,
        )
    if symbol_span is None:
        return {}

    found: dict[str, float] = {"SYMBOL": symbol_span.y0}
    # Only headers below SYMBOL (increasing Y) belong to this legend.
    below = [s for s in near if s.y0 >= symbol_span.y0 - 5]
    for span in sorted(below, key=lambda s: s.y0):
        upper = span.text.upper().strip()
        if upper == "DESCRIPTION" and "DESCRIPTION" not in found:
            found["DESCRIPTION"] = span.y0
        elif upper.startswith("MFG") and "MFG" not in found:
            found["MFG"] = span.y0
        elif upper in {"PART", "PART NUMBER"} and "PART" not in found:
            found["PART"] = span.y0
        elif upper.startswith("NOTES") and "NOTES" not in found:
            found["NOTES"] = span.y0
    return found


def _tech_column_bands(headers: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Y ranges for each column (page coords on rotated Architect-CD legends)."""
    sym = headers["SYMBOL"]
    desc = headers["DESCRIPTION"]
    mfg = headers.get("MFG", desc + 150)
    part = headers.get("PART", mfg + 120)
    notes = headers.get("NOTES", part + 120)
    if not (sym < desc < mfg < part < notes):
        # Fallback fixed offsets from SYMBOL when headers are inconsistent.
        desc = max(desc, sym + 130)
        mfg = max(mfg, desc + 140)
        part = max(part, mfg + 110)
        notes = max(notes, part + 110)

    # Description text (~1307) sits between SYMBOL header (~1248) and
    # DESCRIPTION header (~1379); glyph art sits in the SYMBOL strip.
    # Keep symbol band clear of DESCRIPTION text (~1307 on Architect-CD sheets).
    symbol_hi = min(sym + 48, desc - 70)
    return {
        "symbol": (sym - 30, symbol_hi),
        "description": (symbol_hi, (desc + mfg) / 2.0),
        "mfg": ((desc + mfg) / 2.0, (mfg + part) / 2.0),
        "part": ((mfg + part) / 2.0, (part + notes) / 2.0),
        "notes": ((part + notes) / 2.0, notes + 90),
    }


def _cluster_row_centers(xs: list[float], gap: float) -> list[float]:
    if not xs:
        return []
    ordered = sorted(xs)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(c) / len(c) for c in clusters]


def _looks_like_tech_description(text: str) -> bool:
    if _TECH_HEADER_RE.match(text):
        return False
    if _TECH_DETAIL_MARK_RE.match(text):
        return False
    if text.upper() in {
        "N/A",
        "EXISTING",
        "OFCI",
        "GENERIC",
        "COMMERCIAL",
        "**=HEIGHT AS REQ'D",
        "**=HEIGHT AS REQ'D.",
    }:
        return False
    if text.startswith("**="):
        return False
    if re.match(r"^#\s*=\s*QTY", text, re.I):
        return False
    if re.fullmatch(r"WM\d+", text, re.I):
        return False
    if _looks_like_tag(text) and len(text) <= 4:
        return False
    if EQUIPMENT_HINT_RE.search(text):
        return True
    words = text.split()
    if len(words) >= 2 and len(text) >= 8:
        return True
    if text.upper() in {"MOUNT", "AP"}:
        return True
    if text.startswith("-") or text.startswith("–"):
        return True
    # Continuation fragments for multi-line description cells.
    if re.match(r"^CAMERA\s*\(", text, re.I):
        return True
    if re.match(r"^CEILING\b", text, re.I):
        return True
    return False


def _pack_tech_row(
    row_spans: list[_TextSpan],
    bands: dict[str, tuple[float, float]],
) -> dict[str, str | list[str] | None]:
    buckets: dict[str, list[_TextSpan]] = {
        "symbol": [],
        "description": [],
        "mfg": [],
        "part": [],
        "notes": [],
    }
    for span in row_spans:
        for name, (lo, hi) in bands.items():
            if lo <= span.y0 < hi:
                buckets[name].append(span)
                break

    def join_band(name: str) -> str | None:
        # Descending X == top-to-bottom reading order on rotated legends.
        items = sorted(buckets[name], key=lambda s: (-s.x0, s.y0))
        parts: list[str] = []
        for item in items:
            text = item.text.strip()
            if not text:
                continue
            if name != "notes" and _TECH_DETAIL_MARK_RE.match(text):
                continue
            if name == "description" and re.match(r"^#\s*=\s*QTY", text, re.I):
                continue
            parts.append(text)
        if not parts:
            return None
        merged = _clean(" ".join(parts))
        merged = re.sub(
            r"\bCOMMERCIAL\s+GENERIC\b",
            "COMMERCIAL GENERIC",
            merged,
            flags=re.I,
        )
        merged = re.sub(
            r"\bCHATSWORTH\s+PRODUCTS\b|\bPRODUCTS\s+CHATSWORTH\b",
            "CHATSWORTH PRODUCTS",
            merged,
            flags=re.I,
        )
        return merged or None

    def notes_parts() -> list[str]:
        items = sorted(buckets["notes"], key=lambda s: (-s.x0, s.y0))
        return [item.text.strip() for item in items if item.text.strip()]

    description = join_band("description")
    symbol = join_band("symbol")
    if symbol:
        tags = [
            t
            for t in re.split(r"\s+", symbol)
            if re.fullmatch(r"(?:#|[A-Z]{1,6}\d{0,3}|[A-Z]+-[A-Z0-9]+)", t)
            and ":" not in t
        ]
        symbol = tags[0] if tags else None

    mfg = join_band("mfg")
    part = join_band("part")
    notes = _tokenize_tech_notes(notes_parts())
    if part and part.upper() in {"N/A", "N/A."}:
        part = "N/A"

    return {
        "description": description,
        "symbol": symbol,
        "mfg_model": mfg,
        "part_number": part,
        "notes": notes,
    }


def _tokenize_tech_notes(parts: list[str]) -> list[str] | None:
    """Turn Notes-column spans into distinct callout / phrase items.

    Bare elevation indices (``2``, ``12``) are dropped. Bubble refs like
    ``T-800`` stay as separate list items (unique, order-preserving).
    """
    if not parts:
        return None

    # Flatten any pre-joined fragments into atomic tokens.
    raw: list[str] = []
    for part in parts:
        raw.extend(re.split(r"\s+", part.strip()))
    raw = [t for t in raw if t]

    out: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        upper = token.upper()

        if upper in {"N/A", "N/A."}:
            index += 1
            continue

        # SEE SINGLE LINE (and SEE … LINE variants).
        if upper == "SEE" and index + 2 < len(raw):
            a, b = raw[index + 1].upper(), raw[index + 2].upper()
            if a == "SINGLE" and b == "LINE":
                out.append("SEE SINGLE LINE")
                index += 3
                continue
            if a in {"DETAIL", "SHEET", "PLAN"}:
                out.append(_clean(f"SEE {raw[index + 1]} {raw[index + 2]}"))
                index += 3
                continue

        # GREY/GRAY = EXISTING (tokens may be interleaved with bubble refs).
        if upper in {"GREY", "GRAY"}:
            exist_at: int | None = None
            between: list[str] = []
            for j in range(index + 1, min(index + 8, len(raw))):
                ju = raw[j].upper()
                if ju == "EXISTING" or ju in {"(E)", "(E)."}:
                    exist_at = j
                    break
                if raw[j] == "=":
                    continue
                between.append(raw[j])
            if exist_at is not None:
                out.append("GREY = EXISTING")
                nested = _tokenize_tech_notes(between)
                if nested:
                    out.extend(nested)
                index = exist_at + 1
                continue
            out.append("GREY")
            index += 1
            continue

        # Rack-elevation / detail bubble refs.
        if re.fullmatch(r"T-\d+", token, re.I):
            digits = re.search(r"\d+", token)
            out.append(f"T-{digits.group(0)}" if digits else token.upper())
            index += 1
            continue

        # Bare 1–2 digit elevation/detail marks — not useful note text.
        if re.fullmatch(r"\d{1,2}", token):
            index += 1
            continue

        # Brand / OFCI / other note words.
        out.append(_clean(token))
        index += 1

    # Unique preserve order (especially repeated T-800 bubbles).
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        key = item.upper()
        if not item or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique or None


def _row_x_bounds(centers: list[float], index: int) -> tuple[float, float]:
    """Return (x_right, x_left) clip bounds for a descending-X row list."""
    center = centers[index]
    if index == 0:
        x_right = center + abs(centers[0] - centers[1]) / 2.0 if len(centers) > 1 else center + 18
    else:
        x_right = (centers[index - 1] + center) / 2.0
    if index == len(centers) - 1:
        x_left = center - abs(centers[-2] - centers[-1]) / 2.0 if len(centers) > 1 else center - 18
    else:
        x_left = (center + centers[index + 1]) / 2.0
    # Ensure left < right in page space.
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    pad = 1.5
    return x_right + pad, x_left - pad


def _crop_symbol_cell(
    page: fitz.Page,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    skip: bool = False,
) -> bytes | None:
    if skip:
        return None
    left, right = (x0, x1) if x0 < x1 else (x1, x0)
    top, bottom = (y0, y1) if y0 < y1 else (y1, y0)
    # Minimum cell size — line glyphs are wide/short.
    if right - left < 6 or bottom - top < 6:
        return None
    clip = fitz.Rect(left, top, right, bottom) & page.rect
    if clip.is_empty or clip.width < 4 or clip.height < 4:
        return None
    try:
        # Higher zoom so thin CAD strokes survive blank-detection.
        pix = page.get_pixmap(
            matrix=fitz.Matrix(3.0, 3.0), clip=clip, alpha=False
        )
    except Exception:  # noqa: BLE001
        return None
    if pix.width < 4 or pix.height < 4:
        return None
    if _pixmap_mostly_blank(pix):
        return None
    return pix.tobytes("png")


def _pixmap_mostly_blank(pix: fitz.Pixmap, min_dark_pixels: int = 4) -> bool:
    """True when the crop has almost no dark ink (empty symbol cell)."""
    try:
        samples = pix.samples
    except Exception:  # noqa: BLE001
        return False
    if not samples:
        return True
    n = pix.n
    dark = 0
    # Scan all pixels; crops are small.
    for i in range(0, len(samples) - n + 1, n):
        if n >= 3:
            if samples[i] < 250 or samples[i + 1] < 250 or samples[i + 2] < 250:
                dark += 1
        elif samples[i] < 250:
            dark += 1
        if dark >= min_dark_pixels:
            return False
    return True


def _render_legend_symbol(
    symbol: str | None,
    description: str,
    *,
    part_number: str | None = None,
    size: int = 128,
) -> bytes | None:
    """Draw full Architect-CD-style legend glyphs (box tags, lines, devices)."""
    upper = (description or "").upper()
    part = (part_number or "").upper()
    tag = (symbol or "").strip().upper()

    if "UNDERGROUND" in upper and "CONDUIT" in upper:
        return _draw_dashed_line_icon(size)
    if "CONDUIT STUB" in upper:
        return _draw_conduit_stub_icon(size)
    if "CONDUIT" in upper and "STUB" not in upper:
        return _draw_solid_line_icon(size)
    if "SURFACE RACEWAY" in upper:
        label = "2300"
        for token in ("WM5500", "WM5400", "WM2300", "5500", "5400", "2300"):
            if token in part or token in upper:
                label = token.replace("WM", "")
                break
        return _draw_raceway_icon(size, label)
    if "J-HOOK" in upper:
        return _draw_jhook_icon(size)
    if "DATA POLE" in upper:
        return _draw_data_pole_icon(size)
    if "PERMANENT LINK" in upper:
        return _draw_triangle_icon(size, with_hash=True)
    if "DATA LOCATION" in upper and "CEILING" in upper:
        return _draw_iap_icon(size)
    if "LADDER RACK" in upper:
        return _draw_ladder_rack_icon(size)
    if "WIRELESS ACCESS POINT" in upper or (tag == "AP" and "MOUNT" not in upper):
        return _draw_iap_icon(size)
    if "MULTI-SENSOR" in upper:
        return _draw_camera_multi_icon(size)
    if "NETWORK CAMERA" in upper and "DROP" not in upper and "MOUNT" not in upper:
        return _draw_camera_profile_icon(size)
    if "DATA DROP" in upper and "CAMERA" in upper:
        return _draw_camera_profile_icon(size)
    if "DATA DROP" in upper and "CLOCK" in upper:
        return _draw_clock_speaker_icon(size)
    if "DATA DROP" in upper and "INTERCOM" in upper:
        return _draw_exterior_speaker_icon(size)
    if tag == "WP":
        return _draw_exterior_speaker_icon(size)
    if upper.strip() == "MOUNT" or upper.startswith("MOUNT "):
        return _draw_mount_bracket_icon(size)

    if tag:
        return _render_symbol_badge(tag, size=size)
    return None


def _render_symbol_badge(tag: str, size: int = 128) -> bytes:
    """Square/rect tag matching TECHNOLOGY SYMBOL LEGEND boxes."""
    tag = (tag or "?").strip()[:8]
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    if len(tag) <= 2:
        box = [16, 16, size - 17, size - 17]
    else:
        box = [6, 30, size - 7, size - 31]
    draw.rectangle(box, outline="black", width=4)
    font = _badge_font(tag, size)
    bbox = draw.textbbox((0, 0), tag, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
        tag,
        fill="black",
        font=font,
    )
    return _png_bytes(img)


def _badge_font(tag: str, size: int):
    px = 52 if len(tag) <= 1 else 44 if len(tag) <= 2 else 28 if len(tag) <= 4 else 20
    px = min(px, max(14, size // 2))
    for name in ("arial.ttf", "Arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _png_bytes(img: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_solid_line_icon(size: int) -> bytes:
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    y = size // 2
    draw.line([(6, y), (size - 6, y)], fill="black", width=5)
    return _png_bytes(img)


def _draw_dashed_line_icon(size: int) -> bytes:
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    y = size // 2
    x = 6
    while x < size - 6:
        x2 = min(x + 12, size - 6)
        draw.line([(x, y), (x2, y)], fill="black", width=5)
        x += 18
    return _png_bytes(img)


def _draw_conduit_stub_icon(size: int) -> bytes:
    """Sheet glyph: capital E with the middle bar extended as the conduit."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    # Vertical spine of the E (left).
    x0, y0, y1 = 22, 28, size - 28
    mid_y = size // 2
    arm = 28
    w = 5
    draw.line([(x0, y0), (x0, y1)], fill="black", width=w)
    # Top / bottom short arms of the E.
    draw.line([(x0, y0), (x0 + arm, y0)], fill="black", width=w)
    draw.line([(x0, y1), (x0 + arm, y1)], fill="black", width=w)
    # Middle bar continues across as the conduit run.
    draw.line([(x0, mid_y), (size - 10, mid_y)], fill="black", width=w)
    return _png_bytes(img)


def _draw_raceway_icon(size: int, label: str | None = None) -> bytes:
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    y = size // 2
    draw.line([(4, y), (size - 4, y)], fill="black", width=6)
    text = label or "2300"
    font = _badge_font(text, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 5
    box = [
        (size - tw) / 2 - pad,
        y - th / 2 - pad,
        (size + tw) / 2 + pad,
        y + th / 2 + pad,
    ]
    draw.rectangle(box, fill="white", outline="black", width=2)
    draw.text(
        ((size - tw) / 2 - bbox[0], y - th / 2 - bbox[1]),
        text,
        fill="black",
        font=font,
    )
    return _png_bytes(img)


def _draw_jhook_icon(size: int) -> bytes:
    """Sheet glyph: horizontal line through three capital J letters."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    mid = size // 2
    draw.line([(8, mid), (size - 8, mid)], fill="black", width=3)
    font = _badge_font("J", size)
    for cx in (size // 2 - 32, size // 2, size // 2 + 32):
        bbox = draw.textbbox((0, 0), "J", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Center each J on the line (line crosses the stem).
        draw.text(
            (cx - tw / 2 - bbox[0], mid - th / 2 - bbox[1]),
            "J",
            fill="black",
            font=font,
        )
    return _png_bytes(img)


def _draw_data_pole_icon(size: int) -> bytes:
    """Square with solid vertical bar (sheet DATA POLE — not an X)."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    m = 18
    draw.rectangle([m, m, size - m, size - m], outline="black", width=4)
    bar_w = 18
    draw.rectangle(
        [size // 2 - bar_w // 2, m + 8, size // 2 + bar_w // 2, size - m - 8],
        fill="black",
    )
    return _png_bytes(img)


def _draw_triangle_icon(size: int, *, with_hash: bool) -> bytes:
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    mid = size // 2
    top = 12 if with_hash else 20
    bottom = size - (28 if with_hash else 18)
    draw.polygon(
        [(mid, top), (14, bottom), (size - 14, bottom)],
        outline="black",
        fill="black",
    )
    if with_hash:
        font = _badge_font("#", size)
        bbox = draw.textbbox((0, 0), "#", font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            ((size - tw) / 2 - bbox[0], size - 24 - bbox[1]),
            "#",
            fill="black",
            font=font,
        )
    return _png_bytes(img)


def _draw_ladder_rack_icon(size: int) -> bytes:
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    top, bot = size // 2 - 20, size // 2 + 20
    draw.rectangle([10, top, size - 10, bot], outline="black", width=3)
    for x in (26, size // 2, size - 26):
        draw.line([(x, top), (x, bot)], fill="black", width=3)
    # Side rails continuity
    draw.line([(10, top - 8), (10, bot + 8)], fill="black", width=3)
    draw.line([(size - 10, top - 8), (size - 10, bot + 8)], fill="black", width=3)
    return _png_bytes(img)


def _draw_iap_icon(size: int) -> bytes:
    """Ceiling IAP / interior WAP: circle, mast, triangle, IAP label."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    mid = size // 2
    # Outer circle
    draw.ellipse([14, 10, size - 14, size - 22], outline="black", width=3)
    # Antenna tip
    draw.ellipse([mid - 8, 18, mid + 8, 34], outline="black", width=2, fill="black")
    draw.line([(mid, 34), (mid, size - 48)], fill="black", width=3)
    draw.polygon(
        [(mid, size - 48), (mid - 20, size - 28), (mid + 20, size - 28)],
        outline="black",
        fill="black",
    )
    font = _badge_font("IAP", size)
    bbox = draw.textbbox((0, 0), "IAP", font=font)
    draw.text((mid + 10, mid - 8 - bbox[1]), "IAP", fill="black", font=font)
    return _png_bytes(img)


def _draw_camera_profile_icon(size: int) -> bytes:
    """Side-view network camera on a short mount arm."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    # Wall plate
    draw.rectangle([14, 36, 28, size - 36], outline="black", width=3, fill="black")
    # Arm
    draw.line([(28, size // 2), (48, size // 2)], fill="black", width=4)
    # Camera body
    draw.rounded_rectangle([48, 40, 98, size - 40], radius=10, outline="black", width=3)
    draw.ellipse([62, 52, 90, size - 52], outline="black", width=2)
    draw.ellipse([70, 60, 82, size - 60], fill="black")
    # Lens hood
    draw.polygon([(98, 50), (118, 44), (118, size - 44), (98, size - 50)], outline="black", width=2)
    return _png_bytes(img)


def _draw_camera_multi_icon(size: int) -> bytes:
    """Circular multi-sensor (4-sensor) camera."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([12, 12, size - 12, size - 12], outline="black", width=4)
    mid = size // 2
    # Four sensor lobes
    r = 18
    for dx, dy in ((0, -22), (22, 0), (0, 22), (-22, 0)):
        draw.ellipse(
            [mid + dx - r, mid + dy - r, mid + dx + r, mid + dy + r],
            outline="black",
            width=2,
        )
    draw.ellipse([mid - 8, mid - 8, mid + 8, mid + 8], fill="black")
    return _png_bytes(img)


def _draw_camera_dual_icon(size: int) -> bytes:
    """Dual trapezoid / twin-lens exterior camera."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    # Two wedge bodies
    draw.polygon([(20, 30), (60, 38), (60, size - 38), (20, size - 30)], outline="black", width=3)
    draw.polygon(
        [(68, 38), (108, 30), (108, size - 30), (68, size - 38)],
        outline="black",
        width=3,
    )
    draw.ellipse([30, 52, 50, size - 52], outline="black", width=2)
    draw.ellipse([78, 52, 98, size - 52], outline="black", width=2)
    return _png_bytes(img)


def _draw_clock_speaker_icon(size: int) -> bytes:
    """IP clock / speaker combo (round face + 12:00)."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([16, 16, size - 16, size - 16], outline="black", width=4)
    mid = size // 2
    draw.line([(mid, mid), (mid, 36)], fill="black", width=3)
    draw.line([(mid, mid), (mid + 22, mid + 8)], fill="black", width=3)
    draw.ellipse([mid - 3, mid - 3, mid + 3, mid + 3], fill="black")
    font = _badge_font("12", size)
    bbox = draw.textbbox((0, 0), "12", font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((size - tw) / 2 - bbox[0], 22 - bbox[1]), "12", fill="black", font=font)
    return _png_bytes(img)


def _draw_exterior_speaker_icon(size: int) -> bytes:
    """Weatherproof exterior intercom / speaker horn."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([18, 44, 48, size - 44], outline="black", width=3, fill="black")
    draw.polygon(
        [(48, 40), (110, 28), (110, size - 28), (48, size - 40)],
        outline="black",
        width=3,
    )
    draw.ellipse([70, 52, 96, size - 52], outline="black", width=2)
    font = _badge_font("WP", size)
    bbox = draw.textbbox((0, 0), "WP", font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((size - tw) / 2 - bbox[0], size - 22 - bbox[1]), "WP", fill="black", font=font)
    return _png_bytes(img)


def _draw_mount_bracket_icon(size: int) -> bytes:
    """Fallback mount glyph when not covered by a group icon."""
    img = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    mid = size // 2
    draw.rectangle([mid - 22, 18, mid + 22, 34], outline="black", width=3, fill="black")
    draw.line([(mid, 34), (mid, size - 30)], fill="black", width=4)
    draw.ellipse([mid - 12, size - 42, mid + 12, size - 18], outline="black", width=3)
    return _png_bytes(img)


def _parse_legend_page(
    lines: list[str],
    page_number: int,
    *,
    skip_technology: bool = False,
    page: fitz.Page | None = None,
    spans: list[_TextSpan] | None = None,
) -> list[SymbolEntry]:
    heading_indexes = [
        i for i, line in enumerate(lines) if LEGEND_HEADING_RE.match(line)
    ]
    if not heading_indexes:
        # Fallback: page matched loosely; try first SYMBOL LEGEND substring line.
        heading_indexes = [
            i
            for i, line in enumerate(lines)
            if "SYMBOL LEGEND" in line.upper() and len(line) < 80
        ]
    if not heading_indexes:
        return []

    entries: list[SymbolEntry] = []
    for position, heading_index in enumerate(heading_indexes):
        title = lines[heading_index].rstrip(":").strip() or "Symbol Legend"
        end = (
            heading_indexes[position + 1]
            if position + 1 < len(heading_indexes)
            else min(len(lines), heading_index + 200)
        )
        body = lines[heading_index + 1 : end]
        # Stop at next major sheet section inside the window.
        trimmed: list[str] = []
        for line in body:
            if STOP_SECTION_RE.match(line) and trimmed:
                break
            trimmed.append(line)

        is_technology = "TECHNOLOGY" in title.upper()
        if is_technology and skip_technology:
            continue

        if any(NUMBERED_ITEM_RE.match(line) for line in trimmed[:20]):
            numbered: list[SymbolEntry] = []
            if page is not None:
                page_spans = spans if spans is not None else _page_text_spans(page)
                numbered = _parse_numbered_legend_spatial(
                    page,
                    page_spans,
                    page_number,
                    title,
                )
            if not numbered:
                numbered = _parse_numbered_legend(trimmed, page_number, title)
                if page is not None:
                    _attach_numbered_legend_glyphs(
                        page,
                        numbered,
                        spans if spans is not None else _page_text_spans(page),
                        heading=None,
                    )
            entries.extend(numbered)
        else:
            entries.extend(
                _parse_technology_legend(trimmed, page_number, title)
            )
    return entries


_NUMBER_LABEL_RE = re.compile(r"^(\d+)\.$")
_SCOPE_VERB_RE = re.compile(
    r"^(REMOVE|INSTALL|PERFORM|CLEAN|TEST|NAIL|RAISE|MECHANICALLY|"
    r"APPLY|ENSURE|ALL\s+DRAINS|ON\s+BUILDINGS)\b",
    re.IGNORECASE,
)
_LEGEND_DESC_HINT_RE = re.compile(
    r"\b(DTL|DETAIL|FLASHING|DRAIN|STACK|CURB|HATCH|SKYLIGHT|"
    r"FAN|VENT|PIPE|ROOF|WALL|EDGE|SUPPORT|HOUSING|PENETRATION|"
    r"COPING|SILL|INTAKE|GRAVEL|COATED)\b",
    re.IGNORECASE,
)


def _parse_numbered_legend(
    lines: list[str], page_number: int, title: str
) -> list[SymbolEntry]:
    """Garland-style: ``1.`` / body or ``1. DESCRIPTION``."""
    entries: list[SymbolEntry] = []
    current_num: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal current_num, parts
        text = " ".join(parts).strip()
        if text:
            entries.append(
                SymbolEntry(
                    symbol=current_num,
                    description=text,
                    source_page=page_number,
                    legend_title=title,
                )
            )
        current_num = None
        parts = []

    for line in lines:
        if STOP_SECTION_RE.match(line):
            break
        numbered = NUMBERED_ITEM_RE.match(line)
        if numbered:
            body = (numbered.group(2) or "").strip()
            # Stop when a second numbered list (scope of work) begins.
            if (
                entries
                and numbered.group(1) == "1"
                and current_num is not None
                and int(current_num) > 1
            ):
                flush()
                break
            if body and _SCOPE_VERB_RE.match(body) and entries:
                flush()
                break
            flush()
            current_num = numbered.group(1)
            parts = [body] if body else []
            continue
        if current_num is not None:
            if _is_junk(line):
                continue
            if _SCOPE_VERB_RE.match(line) and entries:
                flush()
                break
            parts.append(line)
    flush()
    return entries


def _symbol_legend_heading_span(spans: list[_TextSpan]) -> _TextSpan | None:
    """Locate non-technology SYMBOL LEGEND heading text."""
    for span in spans:
        upper = (span.text or "").upper()
        if "TECHNOLOGY" in upper:
            continue
        if re.search(r"\bSYMBOL\s+LEGEND\b", upper) or upper.strip() in {
            "SYMBOL",
            "LEGEND",
        }:
            # Prefer a span that is the full title or the word SYMBOL near LEGEND.
            if "SYMBOL" in upper:
                return span
    # Two-word title split across spans: SYMBOL + LEGEND on same baseline.
    for index, span in enumerate(spans):
        if (span.text or "").upper().strip() != "SYMBOL":
            continue
        for other in spans[index + 1 : index + 6]:
            if abs(other.y0 - span.y0) > 8:
                continue
            if (other.text or "").upper().strip() == "LEGEND":
                return span
    return None


def _number_label_candidates(
    page: fitz.Page | None,
    spans: list[_TextSpan],
) -> list[tuple[int, _TextSpan]]:
    """Collect ``N.`` labels; prefer word boxes so ``10. PIPE…`` still works."""
    candidates: list[tuple[int, _TextSpan]] = []
    if page is not None:
        try:
            for word in page.get_text("words") or []:
                text = str(word[4]).strip()
                match = _NUMBER_LABEL_RE.match(text)
                if not match:
                    continue
                candidates.append(
                    (
                        int(match.group(1)),
                        _TextSpan(
                            x0=float(word[0]),
                            y0=float(word[1]),
                            x1=float(word[2]),
                            y1=float(word[3]),
                            text=text,
                        ),
                    )
                )
        except Exception:  # noqa: BLE001
            candidates = []
    if candidates:
        return candidates

    for span in spans:
        text = (span.text or "").strip()
        match = _NUMBER_LABEL_RE.match(text)
        if match:
            candidates.append((int(match.group(1)), span))
            continue
        numbered = NUMBERED_ITEM_RE.match(text)
        if not numbered:
            continue
        num = int(numbered.group(1))
        width = max(6.0, (span.x1 - span.x0) * 0.12)
        candidates.append(
            (
                num,
                _TextSpan(
                    x0=span.x0,
                    y0=span.y0,
                    x1=span.x0 + width,
                    y1=span.y1,
                    text=f"{num}.",
                ),
            )
        )
    return candidates


def _legend_number_spans(
    spans: list[_TextSpan],
    *,
    heading: _TextSpan | None = None,
    page: fitz.Page | None = None,
) -> dict[int, _TextSpan]:
    """Pick the ``N.`` column that belongs to SYMBOL LEGEND (not scope lists)."""
    candidates = _number_label_candidates(page, spans)
    if not candidates:
        return {}

    buckets: dict[int, list[tuple[int, _TextSpan]]] = {}
    for num, span in candidates:
        key = int(round(span.x0 / 8.0) * 8)
        buckets.setdefault(key, []).append((num, span))

    def _score(items: list[tuple[int, _TextSpan]]) -> tuple[float, int]:
        xs = [s.x0 for _, s in items]
        ys = [s.y0 for _, s in items]
        mean_x = sum(xs) / len(xs)
        if heading is not None:
            dx = abs(mean_x - heading.x0)
            below = sum(1 for y in ys if y >= heading.y0 - 4)
            return (-(dx) - (0 if below else 500), len(items))
        return (float(len(items)), len(items))

    best = max(buckets.values(), key=_score)

    by_num: dict[int, _TextSpan] = {}
    for num, span in sorted(best, key=lambda item: item[1].y0):
        by_num.setdefault(num, span)
    return by_num


def _description_for_legend_number(
    spans: list[_TextSpan],
    num_span: _TextSpan,
    *,
    x_right_limit: float,
) -> str | None:
    """Collect description text to the right of a legend number on the same row."""
    y_mid = (num_span.y0 + num_span.y1) / 2.0
    row_tol = max(6.0, (num_span.y1 - num_span.y0) * 1.2)
    parts: list[tuple[float, str]] = []
    for span in spans:
        text = (span.text or "").strip()
        if not text or _NUMBER_LABEL_RE.match(text):
            continue
        # Whole-line span starting with ``N. DESC`` — strip the number prefix.
        numbered = NUMBERED_ITEM_RE.match(text)
        if numbered and int(numbered.group(1)) >= 0:
            body = (numbered.group(2) or "").strip()
            if body and abs(((span.y0 + span.y1) / 2.0) - y_mid) <= row_tol:
                if span.x0 <= num_span.x1 + 40:
                    parts.append((span.x0, body))
                    continue
        if span.x0 < num_span.x1 - 1:
            continue
        if span.x0 > x_right_limit:
            continue
        span_mid = (span.y0 + span.y1) / 2.0
        if abs(span_mid - y_mid) > row_tol:
            continue
        if _SCOPE_VERB_RE.match(text):
            continue
        parts.append((span.x0, text))
    if not parts:
        return None
    parts.sort(key=lambda item: item[0])
    desc = _clean(" ".join(p[1] for p in parts))
    return desc or None


def _description_from_words(
    page: fitz.Page,
    num_span: _TextSpan,
    *,
    x_right_limit: float,
) -> str | None:
    y_mid = (num_span.y0 + num_span.y1) / 2.0
    row_tol = max(5.0, (num_span.y1 - num_span.y0) * 1.1)
    parts: list[tuple[float, str]] = []
    try:
        words = page.get_text("words") or []
    except Exception:  # noqa: BLE001
        return None
    for word in words:
        text = str(word[4]).strip()
        if not text or _NUMBER_LABEL_RE.match(text):
            continue
        x0, y0, x1, y1 = map(float, word[:4])
        if x0 < num_span.x1 - 1 or x0 > x_right_limit:
            continue
        if abs(((y0 + y1) / 2.0) - y_mid) > row_tol:
            continue
        if _SCOPE_VERB_RE.match(text):
            continue
        if re.search(r"\b(GARLAND|COMPANY|AGENT|SALAZAR|REVISION|PHONE|FAX)\b", text, re.I):
            continue
        parts.append((x0, text))
    if not parts:
        return None
    parts.sort(key=lambda item: item[0])
    return _clean(" ".join(p[1] for p in parts)) or None


def _parse_numbered_legend_spatial(
    page: fitz.Page,
    spans: list[_TextSpan],
    page_number: int,
    title: str,
) -> list[SymbolEntry]:
    """Parse numbered SYMBOL LEGEND using the column under the heading."""
    heading = _symbol_legend_heading_span(spans)
    by_num = _legend_number_spans(spans, heading=heading, page=page)
    if len(by_num) < 2:
        return []

    xs = [s.x0 for s in by_num.values()]
    mean_x = sum(xs) / len(xs)
    # Stop before right-hand title-block / agent strip on Garland sheets.
    x_right_limit = mean_x + 175

    entries: list[SymbolEntry] = []
    for num in sorted(by_num):
        span = by_num[num]
        desc = _description_for_legend_number(
            spans, span, x_right_limit=x_right_limit
        )
        if not desc:
            desc = _description_from_words(page, span, x_right_limit=x_right_limit)
        if not desc:
            continue
        if _SCOPE_VERB_RE.match(desc) and not _LEGEND_DESC_HINT_RE.search(desc):
            continue
        # Drop title-block bleed (company / agent strip to the right of the legend).
        if re.search(
            r"\b(GARLAND|COMPANY|AGENT|SALAZAR|REVISION|PHONE|FAX)\b",
            desc,
            re.I,
        ):
            desc = re.split(
                r"\b(?:GARLAND|COMPANY|AGENT|SALAZAR|REVISION|PHONE|FAX)\b",
                desc,
                maxsplit=1,
                flags=re.I,
            )[0].strip(" ,;-")
            if not desc or not _LEGEND_DESC_HINT_RE.search(desc):
                continue
        entries.append(
            SymbolEntry(
                symbol=str(num),
                description=desc,
                source_page=page_number,
                legend_title=title if "SYMBOL" in title.upper() else "SYMBOL LEGEND",
                row_x=span.x0,
            )
        )

    if not entries:
        return []

    _attach_numbered_legend_glyphs(
        page, entries, spans, heading=heading, by_num=by_num
    )
    return entries


def _attach_numbered_legend_glyphs(
    page: fitz.Page,
    entries: list[SymbolEntry],
    spans: list[_TextSpan],
    *,
    heading: _TextSpan | None = None,
    by_num: dict[int, _TextSpan] | None = None,
) -> None:
    """Crop CAD glyphs sitting immediately left of each ``N.`` label."""
    if by_num is None:
        by_num = _legend_number_spans(spans, heading=heading, page=page)
    if not by_num:
        return

    # Visual top→bottom order for row-height midlines.
    ordered = sorted(
        by_num.items(),
        key=lambda item: _text_to_visual_rect(
            page, fitz.Rect(item[1].x0, item[1].y0, item[1].x1, item[1].y1)
        ).y0,
    )
    visual_nums: list[tuple[int, fitz.Rect]] = []
    for num, span in ordered:
        rect = _text_to_visual_rect(
            page, fitz.Rect(span.x0, span.y0, span.x1, span.y1)
        )
        visual_nums.append(
            (
                num,
                fitz.Rect(
                    min(rect.x0, rect.x1),
                    min(rect.y0, rect.y1),
                    max(rect.x0, rect.x1),
                    max(rect.y0, rect.y1),
                ),
            )
        )

    row_bounds: dict[int, tuple[float, float]] = {}
    for index, (num, rect) in enumerate(visual_nums):
        if index == 0:
            y_lo = rect.y0 - 2
        else:
            prev = visual_nums[index - 1][1]
            y_lo = (prev.y1 + rect.y0) / 2.0
        if index + 1 >= len(visual_nums):
            y_hi = rect.y1 + 2
        else:
            nxt = visual_nums[index + 1][1]
            y_hi = (rect.y1 + nxt.y0) / 2.0
        y_lo = min(y_lo, rect.y0 - 1)
        y_hi = max(y_hi, rect.y1 + 1)
        row_bounds[num] = (y_lo, y_hi)

    for entry in entries:
        if not entry.symbol or not str(entry.symbol).isdigit():
            continue
        num = int(entry.symbol)
        match = next((item for item in visual_nums if item[0] == num), None)
        if match is None:
            continue
        _, num_rect = match
        y_lo, y_hi = row_bounds.get(num, (num_rect.y0 - 1, num_rect.y1 + 1))
        # Slight vertical pad so top/bottom box strokes are not clipped, but
        # stay inside the inter-row midline so the next glyph does not bleed in.
        y_lo -= 0.6
        y_hi += 0.6
        row_h = max(8.0, y_hi - y_lo)
        # Wide enough for hatch/fan/skylight boxes; still stop before the digits.
        glyph_w = max(28.0, min(52.0, row_h * 2.8))
        clip = fitz.Rect(
            num_rect.x0 - glyph_w,
            y_lo,
            # Keep a hair of gap before the "N." so digit strokes are excluded,
            # but do not cut the glyph's right wall (was -0.8 → missing sides).
            num_rect.x0 - 0.15,
            y_hi,
        )
        clip = clip & page.rect
        if clip.is_empty or clip.width < 4 or clip.height < 4:
            continue
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(4.0, 4.0), clip=clip, alpha=False
            )
        except Exception:  # noqa: BLE001
            continue
        if pix.width < 4 or pix.height < 4 or _pixmap_mostly_blank(pix):
            continue
        # Do not strip box sides as table grid; only drop distant border rules.
        png = _trim_pixmap_png(
            pix,
            margin=12,
            strip_grid=False,
            erase_separated_rules=True,
        )
        if png and not _png_is_thin_vertical_rule(png):
            entry.symbol_image_png = png


def _png_is_thin_vertical_rule(png: bytes) -> bool:
    """True when the crop is basically a single vertical table border."""
    try:
        img = PILImage.open(io.BytesIO(png)).convert("L")
    except Exception:  # noqa: BLE001
        return False
    mask = img.point(lambda p: 255 if p < 245 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return True
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if height < 8:
        return False
    # A lone rule is very narrow relative to its height.
    return width <= max(3, int(height * 0.18))


def _parse_technology_legend(
    lines: list[str], page_number: int, title: str
) -> list[SymbolEntry]:
    """Architect CD TECHNOLOGY SYMBOL LEGEND: description + optional tag/notes."""
    entries: list[SymbolEntry] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if STOP_SECTION_RE.match(line):
            break
        if _is_junk(line) or not _looks_like_description(line):
            index += 1
            continue

        description = line
        symbol: str | None = None
        notes: list[str] = []
        index += 1

        # Absorb dashed continuations and nearby tag / MFG cells.
        while index < len(lines):
            nxt = lines[index]
            if STOP_SECTION_RE.match(nxt):
                break
            if nxt.startswith("-") or nxt.startswith("–"):
                description = f"{description} {nxt.lstrip('-– ').strip()}".strip()
                index += 1
                continue
            if symbol is None and _looks_like_tag(nxt):
                symbol = nxt
                index += 1
                continue
            if _looks_like_note(nxt):
                # Skip pure EXISTING / N/A noise unless nothing else yet.
                if nxt.upper() in {"N/A", "N/A.", "EXISTING", "COMMERCIAL", "GENERIC"}:
                    index += 1
                    continue
                if nxt.upper().startswith("GREY"):
                    index += 1
                    continue
                notes.append(nxt)
                index += 1
                # Allow one more note cell (e.g. PANDUIT then SEE 27 10 00)
                if index < len(lines) and _looks_like_note(lines[index]):
                    extra = lines[index]
                    if extra.upper() not in {
                        "N/A",
                        "EXISTING",
                        "COMMERCIAL",
                        "GENERIC",
                    } and not extra.upper().startswith("GREY"):
                        notes.append(extra)
                    index += 1
                break
            # Next description / unknown — stop pairing.
            break

        entries.append(
            SymbolEntry(
                description=description,
                symbol=symbol,
                notes=notes or None,
                source_page=page_number,
                legend_title=title,
            )
        )
    return entries


def _is_junk(line: str) -> bool:
    if not line:
        return True
    if JUNK_LINE_RE.match(line):
        return True
    if HEADER_NOTE_RE.search(line):
        return True
    if line.upper() in {"EXISTING", "N/A", "COMMERCIAL", "GENERIC", "LINE"}:
        return True
    return False


def _looks_like_description(line: str) -> bool:
    if _is_junk(line) or _looks_like_tag(line):
        return False
    if line.upper().startswith("SEE ") and len(line.split()) <= 5:
        return False
    if EQUIPMENT_HINT_RE.search(line):
        return True
    # Multi-word equipment label.
    words = line.split()
    if len(words) >= 2 and len(line) >= 10 and not NOTE_RE.match(line):
        return True
    return False


def _looks_like_tag(line: str) -> bool:
    return bool(SHORT_TAG_RE.match(line.strip()))


def _looks_like_note(line: str) -> bool:
    if NOTE_RE.match(line):
        return True
    if re.match(r"^SEE\s+\d{2}\s+\d{2}", line, re.I):
        return True
    if re.match(r"^WM\d+", line, re.I):
        return True
    if line.upper() in {"WIREMOLD", "PANDUIT", "OFCI", "POLY"}:
        return True
    return False


def _dedupe_entries(entries: list[SymbolEntry]) -> list[SymbolEntry]:
    """Drop exact duplicates; for repeated numbered-legend rows keep the glyph."""
    best: dict[tuple[str, str, str], SymbolEntry] = {}
    order: list[tuple[str, str, str]] = []
    for entry in entries:
        desc = re.sub(r"\s+", " ", entry.description).strip().upper()[:120]
        legend = (entry.legend_title or "").upper()
        # Same list index across pages = one legend row (prefer the better glyph).
        if (
            entry.symbol
            and str(entry.symbol).isdigit()
            and "SYMBOL LEGEND" in legend
            and "TECHNOLOGY" not in legend
        ):
            key = ("NUM", (entry.symbol or "").upper(), "")
        elif (
            legend.startswith("LEGEND - FLOOR")
            or legend.startswith("LEGEND - CEILING")
            or legend.startswith("SYMBOLS")
        ):
            # Plan/cover legends repeat on every sheet — one row per meaning.
            # Collapse FLOOR / CEILING (+ DEMOLITION) families so the same
            # description is not listed twice; different meanings stay separate.
            if legend.startswith("LEGEND - FLOOR"):
                family = "LEGEND - FLOOR PLAN"
            elif legend.startswith("LEGEND - CEILING"):
                family = "LEGEND - CEILING"
            else:
                family = legend
            key = ("PLAN", family, desc)
        else:
            key = (
                "ROW",
                f"{(entry.symbol or '').upper()}|{desc}|{(entry.mfg_model or '').upper()[:60]}|"
                f"{(entry.part_number or '').upper()[:60]}|"
                f"{(' · '.join(entry.notes or [])).upper()[:80]}|"
                f"{entry.source_page}|{int(round(entry.row_x)) if entry.row_x is not None else -1}",
                "",
            )
        prior = best.get(key)
        if prior is None:
            best[key] = entry
            order.append(key)
            continue
        chosen = prior
        if entry.symbol_image_png and not prior.symbol_image_png:
            chosen = entry
        elif (
            entry.symbol_image_png
            and prior.symbol_image_png
            and len(entry.symbol_image_png) > len(prior.symbol_image_png)
        ):
            chosen = entry
        elif len(entry.description or "") > len(prior.description or "") and (
            bool(entry.symbol_image_png) == bool(prior.symbol_image_png)
        ):
            # Keep longer clean description when glyph quality is tied.
            if entry.symbol_image_png:
                chosen = entry
                chosen.symbol_image_png = (
                    entry.symbol_image_png or prior.symbol_image_png
                )
            else:
                chosen = prior if prior.symbol_image_png else entry
        # Merge: keep best image onto the richer description when needed.
        if chosen is entry and prior.symbol_image_png and not entry.symbol_image_png:
            entry.symbol_image_png = prior.symbol_image_png
        if chosen is prior and entry.symbol_image_png and not prior.symbol_image_png:
            prior.symbol_image_png = entry.symbol_image_png
        best[key] = chosen

    unique = [best[key] for key in order]
    return _sort_numbered_legend_entries(unique)


def _sort_numbered_legend_entries(entries: list[SymbolEntry]) -> list[SymbolEntry]:
    """Keep relative order, but sort plain numbered SYMBOL LEGEND rows by index."""
    numbered: list[SymbolEntry] = []
    others: list[SymbolEntry] = []
    for entry in entries:
        legend = (entry.legend_title or "").upper()
        if (
            entry.symbol
            and str(entry.symbol).isdigit()
            and "SYMBOL LEGEND" in legend
            and "TECHNOLOGY" not in legend
        ):
            numbered.append(entry)
        else:
            others.append(entry)
    numbered.sort(key=lambda e: int(e.symbol or 0))
    # Numbered legend block first (stable for sheets that only have this), then others.
    if numbered and not others:
        return numbered
    if not numbered:
        return others
    # Preserve non-numbered entries' original relative placement: emit numbered
    # block where the first numbered entry originally appeared.
    first_num = next(
        i
        for i, e in enumerate(entries)
        if e.symbol
        and str(e.symbol).isdigit()
        and "SYMBOL LEGEND" in (e.legend_title or "").upper()
        and "TECHNOLOGY" not in (e.legend_title or "").upper()
    )
    return entries[:first_num] + numbered + [
        e
        for e in entries[first_num:]
        if not (
            e.symbol
            and str(e.symbol).isdigit()
            and "SYMBOL LEGEND" in (e.legend_title or "").upper()
            and "TECHNOLOGY" not in (e.legend_title or "").upper()
        )
    ]
