"""Extract General Notes, Sheet Notes, and Demo Tag Notes from bid PDFs.

Scans every page (no CV page cap). Primary input is PyMuPDF embedded text.
Sheet-note callout digits are cropped from the boxed number drawn on the sheet
so the same symbol can later be matched on the plan.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from pipeline import config
from pipeline.extraction.sheet_identity import (
    PdfSheetScan,
    SheetScanEntry,
    extract_floor_plan_name,
    scan_pdf_diagram_pages,
)

logger = logging.getLogger(__name__)

GENERAL_HEADING_RE = re.compile(
    r"^(?:LEGEND\s*[-–—]\s*)?(?:FLOOR\s+PLAN\s+)?"
    r"(?:DSA\s+|FIRE\s+ALARM\s+)?"
    r"GENERAL(?:\s+PROJECT)?\s+NOTES\s*:?\s*$",
    re.IGNORECASE,
)
SHEET_NOTES_HEADING_RE = re.compile(r"^SHEET\s+NOTES\s*:?\s*$", re.IGNORECASE)
DEMO_TAG_HEADING_RE = re.compile(
    r"^(?:SHEET\s+(T-\d{3,4})\s+)?DEMO\s+TAG\s+NOTES\s*:?\s*$",
    re.IGNORECASE,
)
STOP_HEADING_RE = re.compile(
    r"^(?:"
    r"GENERAL(?:\s+PROJECT)?\s+NOTES|"
    r"(?:LEGEND\s*[-–—]\s*)?(?:FLOOR\s+PLAN\s+)?(?:DSA\s+|FIRE\s+ALARM\s+)?"
    r"GENERAL(?:\s+PROJECT)?\s+NOTES|"
    r"SHEET\s+NOTES|"
    r"(?:SHEET\s+T-\d+\s+)?DEMO\s+TAG\s+NOTES|"
    r"KEY\s+PLAN|SYMBOL\s+LEGEND|TECHNOLOGY\s+SYMBOL|"
    r"PICTURE\s+DETAIL|PHOTO(?:GRAPHIC)?\s+DETAIL|"
    r"DATE(\s+PLOTTED)?|JOB\s+NUMBER|SHEET\s+TITLE|SHEET\s+NO\.?|"
    r"DRAWN\s+BY|CHK\s*BY|SCALE|REVISION|"
    r"TAG\s+ID|LOCATION|DESCRIPTION"
    r")\s*:?\s*$",
    re.IGNORECASE,
)
GENERAL_ITEM_RE = re.compile(r"^(\d+)\.\s*(.*)$")
SHEET_ITEM_RE = re.compile(r"^(\d{1,2})(?:\s+(.*))?$")
LOCATION_RE = re.compile(r"^[A-Z]{1,4}-?\d{1,3}[A-Z]?$")
CABLE_ID_RE = re.compile(r"^\d{5,}$")
DESC_HINT_RE = re.compile(
    r"\b(EA\.?|PAIR|CAT\s*\d|RG\s*\d+|TERMINAL|COAX|FIBER)\b",
    re.IGNORECASE,
)
SHEET_NO_RE = re.compile(r"\b(T-\d{3,4})\b", re.IGNORECASE)
JUNK_LINE_RE = re.compile(
    r"^(www\.|office:|phone:|fax:|the architect does not|"
    r"\d{3,5}\s+\w+\s+(ave|st|rd|blvd)|"
    r"date:|job number:|sheet title|drawn by:?)$",
    re.IGNORECASE,
)
TITLE_BLOCK_JUNK_RE = re.compile(
    r"("
    r"\d{3,5}\s+\S+.*\b(AVE\.?|AVENUE|ST\.|STREET|RD\.|ROAD|BLVD)"
    r"|OFFICE:"
    r"|WWW\."
    r"|KMMSERVICES"
    r"|THE ARCHITECT DOES NOT"
    r"|CARMICHAEL"
    r"|JOB NUMBER"
    r"|SHEET TITLE"
    r"|DRAWN BY"
    r"|DATE PLOTTED"
    r")",
    re.IGNORECASE,
)


@dataclass
class _TextSpan:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class NoteItem:
    """One general note, sheet note, or demo-tag row."""

    note_type: str
    text: str
    symbol: str | None = None
    tag_id: str | None = None
    location: str | None = None
    source_page: int = 1
    sheet_number: str | None = None
    display_name: str = ""
    slug: str | None = None
    sheet_kind: str | None = None
    related_diagram_page: int | None = None
    related_diagram_sheet: str | None = None
    heading: str = ""
    bbox: tuple[float, float, float, float] | None = None
    symbol_image_png: bytes | None = None

    def to_json(self) -> dict:
        return {
            "note_type": self.note_type,
            "text": self.text,
            "symbol": self.symbol,
            "tag_id": self.tag_id,
            "location": self.location,
            "source_page": self.source_page,
            "sheet_number": self.sheet_number,
            "display_name": self.display_name,
            "slug": self.slug,
            "sheet_kind": self.sheet_kind,
            "related_diagram_page": self.related_diagram_page,
            "related_diagram_sheet": self.related_diagram_sheet,
            "heading": self.heading,
            "bbox": list(self.bbox) if self.bbox else None,
            "has_symbol_image": bool(self.symbol_image_png),
        }


@dataclass
class SheetNotesInfo:
    """All drawing notes extracted from a bid set."""

    items: list[NoteItem] = field(default_factory=list)
    total_pages: int = 0
    note_pages: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def count_of(self, note_type: str) -> int:
        return sum(1 for item in self.items if item.note_type == note_type)

    def grouped_by_diagram(self) -> list[tuple[str, list[NoteItem]]]:
        """Group notes by floor-plan slug (falls back to sheet number / page)."""
        grouped: dict[str, list[NoteItem]] = {}
        order: list[str] = []
        for item in self.items:
            key = (
                item.slug
                or (item.related_diagram_sheet or item.sheet_number or "")
                or f"page-{item.source_page:02d}"
            )
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(item)

        def sort_key(key: str) -> tuple:
            rows = grouped[key]
            page = rows[0].related_diagram_page or rows[0].source_page
            return (page, key)

        return [(key, grouped[key]) for key in sorted(order, key=sort_key)]

    def to_json_dict(self) -> dict:
        diagrams: list[dict] = []
        for slug, rows in self.grouped_by_diagram():
            sample = rows[0]
            diagrams.append(
                {
                    "slug": slug,
                    "display_name": sample.display_name,
                    "sheet_number": sample.sheet_number,
                    "source_page": sample.source_page,
                    "related_diagram_page": sample.related_diagram_page,
                    "folder": f"05_metadata/page_{sample.related_diagram_page or sample.source_page:03d}.json",
                    "general": [i.to_json() for i in rows if i.note_type == "general"],
                    "sheet": [i.to_json() for i in rows if i.note_type == "sheet"],
                    "sheet_tag": [i.to_json() for i in rows if i.note_type == "sheet_tag"],
                }
            )
        return {
            "total_pages": self.total_pages,
            "note_pages": self.note_pages,
            "item_count": self.item_count,
            "general_count": self.count_of("general"),
            "sheet_count": self.count_of("sheet"),
            "sheet_tag_count": self.count_of("sheet_tag"),
            "notes": self.notes,
            "diagrams": diagrams,
            "items": [item.to_json() for item in self.items],
        }


def write_diagram_notes_json(info: SheetNotesInfo, job_root: Path) -> int:
    """Write ``05_metadata/page_NNN_notes.json`` for each diagram page.

    Matches by slug in page metadata first, then by page number.
    """
    meta_dir = job_root / "05_metadata"
    if not meta_dir.is_dir():
        return 0

    by_slug: dict[str, Path] = {}
    by_page: dict[int, Path] = {}
    for meta_path in sorted(meta_dir.glob("page_*.json")):
        if meta_path.name.endswith("_notes.json"):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        slug = meta.get("slug")
        if isinstance(slug, str) and slug:
            by_slug[slug] = meta_path
        page = meta.get("page")
        if isinstance(page, int):
            by_page[page] = meta_path

    dest_rows: dict[Path, list[NoteItem]] = {}
    dest_order: list[Path] = []
    for slug, rows in info.grouped_by_diagram():
        sample = rows[0]
        meta_path = by_slug.get(slug)
        if meta_path is None:
            page = sample.related_diagram_page or sample.source_page
            meta_path = by_page.get(page)
        if meta_path is None:
            continue
        if meta_path not in dest_rows:
            dest_rows[meta_path] = []
            dest_order.append(meta_path)
        dest_rows[meta_path].extend(rows)

    written = 0
    for meta_path in dest_order:
        rows = dest_rows[meta_path]
        sample = rows[0]
        page_num = sample.related_diagram_page or sample.source_page
        notes_name = f"page_{page_num:03d}_notes.json"
        payload = {
            "page_number": page_num,
            "display_name": sample.display_name,
            "slug": slug,
            "sheet_number": sample.sheet_number,
            "sheet_kind": sample.sheet_kind,
            "related_diagram_page": sample.related_diagram_page,
            "related_diagram_sheet": sample.related_diagram_sheet,
            "folder": f"05_metadata/{meta_path.name}",
            "general": [i.to_json() for i in rows if i.note_type == "general"],
            "sheet": [i.to_json() for i in rows if i.note_type == "sheet"],
            "sheet_tag": [i.to_json() for i in rows if i.note_type == "sheet_tag"],
        }
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload["slug"] = meta.get("slug") or payload["slug"]
            payload["display_name"] = meta.get("display_name") or payload["display_name"]
        except (OSError, json.JSONDecodeError):
            pass
        (meta_dir / notes_name).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        written += 1
    return written


def extract_sheet_notes_from_pdf(pdf_path: Path) -> SheetNotesInfo:
    """Open PDF, scan all pages for notes blocks, return structured items."""
    info = SheetNotesInfo()
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        info.notes.append(f"Could not open PDF: {exc}")
        return info

    try:
        scan = scan_pdf_diagram_pages(pdf_path)
        sheet_to_diagram = _sheet_to_diagram_page(scan)
        info.total_pages = document.page_count
        used_slugs: set[str] = {e.name.slug for e in scan.entries}
        for page_index in range(document.page_count):
            page = document[page_index]
            page_number = page_index + 1
            text = page.get_text("text") or ""
            if not _page_has_notes(text):
                continue
            scanned = scan.by_page().get(page_number)
            identity = _page_identity(scanned, text, page_number, used_slugs)
            spans = _page_text_spans(page)
            lines = [_clean(line) for line in text.splitlines()]
            lines = [line for line in lines if line]
            page_items = _parse_general_notes(lines, page_number)
            page_items.extend(_parse_sheet_notes(lines, page_number))
            _attach_sheet_note_glyphs(page, spans, page_items)
            page_items.extend(_parse_demo_tags(spans, page, page_number, text))
            if not page_items:
                continue
            _apply_identity(page_items, identity, sheet_to_diagram, text)
            info.items.extend(page_items)
            info.note_pages.append(page_number)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sheet notes extraction failed for %s", pdf_path)
        info.notes.append(f"Extraction error: {exc}")
    finally:
        document.close()

    info.note_pages = sorted(set(info.note_pages))
    logger.info(
        "Sheet notes %s: pages=%d notes=%d general=%d sheet=%d tags=%d",
        pdf_path.name,
        info.total_pages,
        info.item_count,
        info.count_of("general"),
        info.count_of("sheet"),
        info.count_of("sheet_tag"),
    )
    return info


def _page_has_notes(text: str) -> bool:
    upper = text.upper()
    return (
        "GENERAL NOTES" in upper
        or "SHEET NOTES" in upper
        or "DEMO TAG NOTES" in upper
    )


def _sheet_to_diagram_page(scan: PdfSheetScan) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for entry in scan.diagram_entries:
        match = SHEET_NO_RE.search(entry.name.display_name)
        if match:
            mapping[match.group(1).upper()] = entry.page_number
    return mapping


def _page_identity(
    scanned: SheetScanEntry | None,
    text: str,
    page_number: int,
    used_slugs: set[str],
) -> dict:
    if scanned is not None:
        name = scanned.name
        kind = scanned.sheet_kind.value
        is_diagram = scanned.is_diagram
    else:
        name = extract_floor_plan_name(text, page_number, used_slugs=used_slugs)
        kind = None
        is_diagram = False
    sheet_number = None
    match = SHEET_NO_RE.search(name.display_name) or SHEET_NO_RE.search(text)
    if match:
        sheet_number = match.group(1).upper()
    return {
        "display_name": name.display_name,
        "slug": name.slug,
        "sheet_kind": kind,
        "sheet_number": sheet_number,
        "is_diagram": is_diagram,
        "page_number": page_number,
    }


def _apply_identity(
    items: list[NoteItem],
    identity: dict,
    sheet_to_diagram: dict[str, int],
    page_text: str,
) -> None:
    heading_sheet = None
    for raw in page_text.splitlines():
        match = DEMO_TAG_HEADING_RE.match(_clean(raw))
        if match and match.group(1):
            heading_sheet = match.group(1).upper()
            break
    related_sheet = heading_sheet or identity["sheet_number"]
    related_page = identity["page_number"] if identity["is_diagram"] else None
    if related_sheet and related_sheet in sheet_to_diagram:
        related_page = sheet_to_diagram[related_sheet]
    elif identity["is_diagram"]:
        related_page = identity["page_number"]
        related_sheet = identity["sheet_number"]
    for item in items:
        item.sheet_number = identity["sheet_number"]
        item.display_name = identity["display_name"]
        item.slug = identity["slug"]
        item.sheet_kind = identity["sheet_kind"]
        item.related_diagram_page = related_page
        item.related_diagram_sheet = related_sheet
        if item.note_type == "sheet_tag" and heading_sheet:
            item.related_diagram_sheet = heading_sheet
            item.related_diagram_page = sheet_to_diagram.get(heading_sheet, related_page)


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _page_text_spans(page: fitz.Page) -> list[_TextSpan]:
    spans: list[_TextSpan] = []
    data = page.get_text("dict") or {}
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            spans.append(
                _TextSpan(x0=x0, y0=y0, x1=x1, y1=y1, text=_clean(text))
            )
    return spans


def _parse_general_notes(lines: list[str], page_number: int) -> list[NoteItem]:
    return _parse_numbered_block(
        lines,
        page_number,
        heading_re=GENERAL_HEADING_RE,
        note_type="general",
        item_fn=_general_item_start,
        extra_stop=_is_sheet_or_tag_boundary,
    )


def _parse_sheet_notes(lines: list[str], page_number: int) -> list[NoteItem]:
    return _parse_numbered_block(
        lines,
        page_number,
        heading_re=SHEET_NOTES_HEADING_RE,
        note_type="sheet",
        item_fn=_sheet_item_start,
        extra_stop=_is_tag_boundary,
    )


def _general_item_start(line: str) -> tuple[str | None, str]:
    match = GENERAL_ITEM_RE.match(line)
    if not match:
        return None, ""
    return match.group(1), (match.group(2) or "").strip()


def _sheet_item_start(line: str) -> tuple[str | None, str]:
    if GENERAL_ITEM_RE.match(line):
        return None, ""
    match = SHEET_ITEM_RE.match(line)
    if not match:
        return None, ""
    rest = (match.group(2) or "").strip()
    if rest and not re.match(r"^[A-Z(\"]", rest):
        return None, ""
    return match.group(1), rest


def _is_sheet_or_tag_boundary(line: str) -> bool:
    return bool(
        SHEET_NOTES_HEADING_RE.match(line)
        or _is_tag_boundary(line)
    )


def _is_title_block_junk(line: str) -> bool:
    if JUNK_LINE_RE.match(line):
        return True
    if TITLE_BLOCK_JUNK_RE.search(line):
        return True
    # Spaced-out company wordmarks: "T E C H N O L O G Y"
    if re.match(r"^(?:[A-Z]\s+){5,}[A-Z]\s*$", line):
        return True
    return False


def _is_tag_boundary(line: str) -> bool:
    return bool(
        DEMO_TAG_HEADING_RE.match(line)
        or LOCATION_RE.match(line)
        or CABLE_ID_RE.match(line)
        or re.match(r"^(LOCATION|TAG\s+ID|DESCRIPTION)\s*:?\s*$", line, re.I)
    )


def _parse_numbered_block(
    lines: list[str],
    page_number: int,
    *,
    heading_re: re.Pattern[str],
    note_type: str,
    item_fn,
    extra_stop,
) -> list[NoteItem]:
    items: list[NoteItem] = []
    index = 0
    while index < len(lines):
        if not heading_re.match(lines[index]):
            index += 1
            continue
        heading = lines[index].rstrip(":")
        index += 1
        current_num: str | None = None
        parts: list[str] = []

        def flush() -> None:
            nonlocal current_num, parts
            text = " ".join(part for part in parts if part).strip()
            if current_num is None or not text:
                current_num = None
                parts = []
                return
            items.append(
                NoteItem(
                    note_type=note_type,
                    text=text,
                    symbol=current_num,
                    source_page=page_number,
                    heading=heading,
                )
            )
            current_num = None
            parts = []

        while index < len(lines):
            line = lines[index]
            if heading_re.match(line):
                break
            if STOP_HEADING_RE.match(line) and not heading_re.match(line):
                break
            if extra_stop(line):
                break
            if _is_title_block_junk(line):
                break
            number, rest = item_fn(line)
            if number is not None:
                flush()
                current_num = number
                if rest:
                    parts.append(rest)
            elif current_num is not None:
                parts.append(line)
            index += 1
        flush()
    return items


def _small_box_rects(page: fitz.Page) -> list[fitz.Rect]:
    boxes: list[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return boxes
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        width = abs(rect.x1 - rect.x0)
        height = abs(rect.y1 - rect.y0)
        if min(width, height) < 5 or max(width, height) > 48:
            continue
        if abs(width - height) > 14:
            continue
        boxes.append(fitz.Rect(rect))
    return boxes


def _heading_span(spans: list[_TextSpan], pattern: re.Pattern[str]) -> _TextSpan | None:
    for span in spans:
        if pattern.match(span.text):
            return span
    return None


def _in_notes_number_column(heading: _TextSpan, span: _TextSpan) -> bool:
    """True when ``span`` is a callout digit in the notes column, not on the plan."""
    head_w = heading.x1 - heading.x0
    head_h = heading.y1 - heading.y0
    if head_h > head_w * 1.5:
        # Rotated heading: list runs toward decreasing x; digits sit left of the words.
        return (
            span.x1 <= heading.x1 + 24
            and span.x0 >= heading.x0 - 920
            and span.y1 <= heading.y0 + 24
        )
    return (
        abs(span.x0 - heading.x0) < 90
        and span.y0 >= heading.y0 - 8
        and span.y0 <= heading.y1 + 900
    )


def _containing_box(span: _TextSpan, boxes: list[fitz.Rect]) -> fitz.Rect | None:
    text_rect = fitz.Rect(span.x0, span.y0, span.x1, span.y1)
    best: fitz.Rect | None = None
    best_area = 10**9
    for box in boxes:
        if box.contains(text_rect) or box.intersects(text_rect):
            area = abs(box.width * box.height)
            if area < best_area:
                best = box
                best_area = area
    return best


def _to_visual(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect:
    mapped = fitz.Rect(rect)
    if page.rotation:
        mapped = mapped * page.rotation_matrix
    return fitz.Rect(
        min(mapped.x0, mapped.x1),
        min(mapped.y0, mapped.y1),
        max(mapped.x0, mapped.x1),
        max(mapped.y0, mapped.y1),
    )


def _crop_rect_png(page: fitz.Page, rect: fitz.Rect, *, pad: float = 1.2) -> bytes | None:
    visual = _to_visual(page, rect)
    clip = fitz.Rect(
        visual.x0 - pad,
        visual.y0 - pad,
        visual.x1 + pad,
        visual.y1 + pad,
    ) & page.rect
    if clip.is_empty or clip.width < 2 or clip.height < 2:
        return None
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(4.0, 4.0), clip=clip, alpha=False)
    except Exception:  # noqa: BLE001
        return None
    if pix.width < 4 or pix.height < 4:
        return None
    return pix.tobytes("png")


def _render_boxed_badge(tag: str, size: int = 96) -> bytes:
    tag = (tag or "?").strip()[:4]
    image = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    inset = 8
    draw.rectangle([inset, inset, size - inset - 1, size - inset - 1], outline="black", width=4)
    font = None
    px = 44 if len(tag) <= 1 else 36 if len(tag) <= 2 else 24
    for name in ("arial.ttf", "Arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            font = ImageFont.truetype(name, px)
            break
        except Exception:  # noqa: BLE001
            continue
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), tag, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
        tag,
        fill="black",
        font=font,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _attach_sheet_note_glyphs(
    page: fitz.Page,
    spans: list[_TextSpan],
    items: list[NoteItem],
) -> None:
    if not config.SHEET_NOTES_GLYPH_CROP:
        return
    sheet_items = [item for item in items if item.note_type == "sheet" and item.symbol]
    if not sheet_items:
        return
    heading = _heading_span(spans, SHEET_NOTES_HEADING_RE)
    boxes = _small_box_rects(page)
    used: set[int] = set()
    digit_spans = [
        span
        for span in spans
        if span.text.isdigit()
        and 1 <= len(span.text) <= 2
        and (heading is None or _in_notes_number_column(heading, span))
    ]
    for item in sheet_items:
        match: _TextSpan | None = None
        for index, span in enumerate(digit_spans):
            if index in used or span.text != item.symbol:
                continue
            match = span
            used.add(index)
            break
        box = _containing_box(match, boxes) if match is not None else None
        png = None
        if box is not None:
            png = _crop_rect_png(page, box)
            item.bbox = (box.x0, box.y0, box.x1, box.y1)
        elif match is not None:
            pad = max(4.0, (match.x1 - match.x0) * 0.45, (match.y1 - match.y0) * 0.45)
            fallback = fitz.Rect(
                match.x0 - pad,
                match.y0 - pad,
                match.x1 + pad,
                match.y1 + pad,
            )
            png = _crop_rect_png(page, fallback)
            item.bbox = (fallback.x0, fallback.y0, fallback.x1, fallback.y1)
        item.symbol_image_png = png or _render_boxed_badge(item.symbol or "?")


def _parse_demo_tags(
    spans: list[_TextSpan],
    page: fitz.Page,
    page_number: int,
    page_text: str,
) -> list[NoteItem]:
    heading = None
    heading_sheet = None
    for span in spans:
        match = DEMO_TAG_HEADING_RE.match(span.text)
        if match:
            heading = span
            if match.group(1):
                heading_sheet = match.group(1).upper()
            break
    if heading is None:
        if "DEMO TAG NOTES" not in page_text.upper():
            return []
        return []

    loc_header = _header_y(spans, r"^LOCATION\s*:?\s*$")
    tag_header = _header_y(spans, r"^TAG\s+ID\s*:?\s*$")
    desc_header = _header_y(spans, r"^DESCRIPTION\s*:?\s*$")

    head_w = heading.x1 - heading.x0
    head_h = heading.y1 - heading.y0
    vertical = head_h > head_w * 1.5

    region: list[_TextSpan] = []
    for span in spans:
        if span is heading:
            continue
        if STOP_HEADING_RE.match(span.text) and not DEMO_TAG_HEADING_RE.match(span.text):
            continue
        if vertical:
            if span.x0 > heading.x0 + 30:
                continue
            if span.x0 < heading.x0 - 500:
                continue
            if loc_header is not None and span.y1 < loc_header - 40:
                continue
            if desc_header is not None and span.y0 > desc_header + 280:
                continue
        else:
            if span.y1 < heading.y0 - 10:
                continue
            if span.y0 > heading.y0 + 420:
                continue
            if abs(span.x0 - heading.x0) > 420:
                continue
        region.append(span)

    rows = _cluster_tag_rows(region, vertical=vertical)
    items: list[NoteItem] = []
    boxes = _small_box_rects(page)
    for row in rows:
        location = None
        tag_id = None
        desc_parts: list[str] = []
        glyph_span: _TextSpan | None = None
        for span in row:
            text = span.text
            if re.match(r"^(LOCATION|TAG\s+ID|DESCRIPTION)\s*:?\s*$", text, re.I):
                continue
            if DEMO_TAG_HEADING_RE.match(text):
                continue
            col = _tag_column(span, loc_header, tag_header, desc_header, vertical)
            if LOCATION_RE.match(text) and location is None:
                location = text
                glyph_span = glyph_span or span
            elif CABLE_ID_RE.match(text) and tag_id is None:
                tag_id = text
                glyph_span = glyph_span or span
            elif DESC_HINT_RE.search(text) or col == "desc":
                desc_parts.append(text)
            elif col == "loc" and LOCATION_RE.match(text):
                location = text
            elif col == "tag" and (CABLE_ID_RE.match(text) or LOCATION_RE.match(text)):
                tag_id = text
        if not location and not tag_id:
            continue
        description = " ".join(desc_parts).strip()
        symbol = location or tag_id
        item = NoteItem(
            note_type="sheet_tag",
            text=description,
            symbol=symbol,
            tag_id=tag_id or location,
            location=location,
            source_page=page_number,
            heading="DEMO TAG NOTES"
            if not heading_sheet
            else f"SHEET {heading_sheet} DEMO TAG NOTES",
        )
        if glyph_span is not None and config.SHEET_NOTES_GLYPH_CROP:
            box = _containing_box(glyph_span, boxes)
            if box is not None:
                item.symbol_image_png = _crop_rect_png(page, box)
                item.bbox = (box.x0, box.y0, box.x1, box.y1)
        items.append(item)
    return items


def _header_y(spans: list[_TextSpan], pattern: str) -> float | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for span in spans:
        if regex.match(span.text):
            return span.y0
    return None


def _tag_column(
    span: _TextSpan,
    loc_y: float | None,
    tag_y: float | None,
    desc_y: float | None,
    vertical: bool,
) -> str | None:
    if not vertical:
        return None
    ys = [(loc_y, "loc"), (tag_y, "tag"), (desc_y, "desc")]
    known = [(y, name) for y, name in ys if y is not None]
    if not known:
        return None
    y_mid = (span.y0 + span.y1) / 2.0
    best_name = None
    best_dist = 10**9
    for y_val, name in known:
        dist = abs(y_mid - y_val)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name if best_dist < 80 else None


def _cluster_tag_rows(spans: list[_TextSpan], *, vertical: bool) -> list[list[_TextSpan]]:
    if not spans:
        return []
    key_fn = (lambda s: s.x0) if vertical else (lambda s: s.y0)
    ordered = sorted(spans, key=key_fn, reverse=vertical)
    rows: list[list[_TextSpan]] = []
    current: list[_TextSpan] = []
    current_key: float | None = None
    for span in ordered:
        value = key_fn(span)
        if current_key is None or abs(value - current_key) <= 8:
            current.append(span)
            if current_key is None:
                current_key = value
            else:
                current_key = (current_key * (len(current) - 1) + value) / len(current)
        else:
            rows.append(current)
            current = [span]
            current_key = value
    if current:
        rows.append(current)
    return rows
