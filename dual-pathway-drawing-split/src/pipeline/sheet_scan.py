"""Read a drawing sheet's own text layer: sheet code, plan titles, room tags.

Nothing here is tied to a particular drawing set. Sets differ in every detail —
sheet codes run T-020, A1.1 or nothing at all; plans are titled "B-WING (EAST)",
"B WING", "ANNEX" or "ROOF PLAN" — so each sheet is read on its own terms: the
biggest type on a sheet is its titles, and room numbers carry the area they
belong to. Only wording that never names a plan is listed below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Titles that head something other than a plan of a building area.
NOT_A_PLAN_RE = re.compile(
    r"\b(?:"
    r"KEY\s*PLAN|SHEET\s+NOTE|GENERAL\s+NOTE|DEMO\s+TAG\s+NOTE|KEY\s*NOTE|NOTES?:"
    r"|LEGEND|ABBREVIATION|SYMBOL|SCALE|MATCH\s*LINE"
    r"|CONSTRUCTION\s+DOCUMENT|NOT\s+FOR\s+CONSTRUCTION|FOR\s+REFERENCE\s+ONLY"
    r"|COVER\s+SHEET|SHEET\s+INDEX|DRAWING\s+INDEX|TITLE\s+SHEET|SHEET\s+LIST"
    r"|DETAIL|SECTION|ELEVATION|SCHEDULE|DIAGRAM|SCOPE\s+OF\s+WORK|IMPORTANT"
    r"|AGENCY\s+APPROVAL|CONSULTANT|PROFESSIONAL\s+STAMP|ARCHITECT"
    r"|SCHOOL\s+DISTRICT|UPGRADE\s+AT|STUBS|BLEACHERS"
    r")",
    re.IGNORECASE,
)
SHEET_CODE_RE = re.compile(r"^[A-Z]{1,3}[-\s]?\d{2,4}[A-Z]?$", re.IGNORECASE)
ADDRESS_RE = re.compile(
    r"\b(?:SUITE|TEL|AVE|AVENUE|STREET|DRIVE|ROAD|BLVD|WAY|CA\s+9\d{4})\b"
    r"|\b\d{3,}\s+[A-Z]",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
# Standalone words that label a compass, a site feature or a drawing state
# rather than a plan. Matched whole, so "E-WING (NORTH)" is unaffected.
NOT_A_PLAN_EXACT = {
    "NORTH",
    "TRUE NORTH",
    "GRASS",
    "PLANTER",
    "PLAY AREA",
    "PARKING",
    "EXISTING",
    "NEW",
    "DEMO",
    "TYP",
}
MIN_TITLE_SIZE = 12.0
TITLE_SIZE_TOLERANCE = 0.85
MAX_TITLE_CHARS = 40

# Room tags such as B-13, C-19, AD-17A, MU-1, E-05B. The prefix says which
# building area a room belongs to, which is the only reliable way to place a
# boundary between two wings that share walls.
ROOM_TAG_RE = re.compile(r"^(?P<prefix>[A-Z]{1,3})[-\s]?(?P<number>\d{1,3})[A-Z]?$")
MAX_ROOM_TAG_SIZE = 20.0
MIN_TAGS_FOR_PLAN = 4


@dataclass
class WingHint:
    name: str
    # Normalized 0-1 on the PDF page
    box: list[float]


@dataclass
class PageScan:
    page: int
    is_diagram: bool
    sheet_no: str | None
    title: str
    reason: str
    wing_hints: list[WingHint] = field(default_factory=list)
    # Raw label boxes (not expanded) — exact anchors for naming CV panels.
    wing_labels: list[WingHint] = field(default_factory=list)
    # Room tags, named by their prefix (B, C, AD, MU, …).
    room_tags: list[WingHint] = field(default_factory=list)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u2212", "-").replace("–", "-").replace("—", "-")).strip()


TextLine = tuple[float, str, tuple[float, float, float, float]]


def _lines(text_dict: dict) -> list[TextLine]:
    """Every text line on the page as (type size, text, bbox)."""
    out: list[TextLine] = []
    for block in text_dict.get("blocks") or []:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = _clean(" ".join(str(s.get("text", "")).strip() for s in spans if s.get("text")))
            if not text:
                continue
            bbox = line.get("bbox") or block.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            size = max((float(s.get("size") or 0) for s in spans), default=0.0)
            out.append((size, text, tuple(map(float, bbox[:4]))))
    return out


def _pdf_to_norm(
    bbox: tuple[float, float, float, float],
    pdf_w: float,
    pdf_h: float,
    rotation: int,
) -> list[float]:
    """Map an unrotated PDF text box into the rendered page orientation."""
    x0, y0, x1, y1 = bbox
    nx0, nx1 = x0 / max(pdf_w, 1.0), x1 / max(pdf_w, 1.0)
    ny0, ny1 = y0 / max(pdf_h, 1.0), y1 / max(pdf_h, 1.0)
    points = [(nx0, ny0), (nx1, ny0), (nx0, ny1), (nx1, ny1)]

    def rotate(x: float, y: float) -> tuple[float, float]:
        angle = rotation % 360
        if angle == 90:
            return 1.0 - y, x
        if angle == 180:
            return 1.0 - x, 1.0 - y
        if angle == 270:
            return y, 1.0 - x
        return x, y

    rotated = [rotate(x, y) for x, y in points]
    xs, ys = [p[0] for p in rotated], [p[1] for p in rotated]
    return [
        max(0.0, min(1.0, min(xs))),
        max(0.0, min(1.0, min(ys))),
        max(0.0, min(1.0, max(xs))),
        max(0.0, min(1.0, max(ys))),
    ]


def _title_name(text: str) -> str:
    """Folder-friendly name, spelled the way the sheet spells it."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.upper()).strip("-")
    return cleaned or "PLAN"


def _sheet_code(lines: list[TextLine]) -> str | None:
    """The sheet's own number, set in the largest type in its title block."""
    codes = [
        (size, text.upper())
        for size, text, _bbox in lines
        if SHEET_CODE_RE.fullmatch(text) and any(c.isdigit() for c in text)
    ]
    if not codes:
        return None
    codes.sort(key=lambda item: -item[0])
    return _clean(codes[0][1])


def _wing_hints_from_dict(
    lines: list[TextLine],
    pdf_w: float,
    pdf_h: float,
    rotation: int,
) -> list[WingHint]:
    """Find the plan titles on a sheet.

    Every set names its plans differently — "B-WING (EAST)", "B WING", "ANNEX",
    "MULTI-USE BLDG.", "ROOF PLAN" — so titles are found by type size instead of
    by wording: a sheet sets its plan titles in the largest type on the drawing,
    well above room tags and note text.
    """
    candidates: list[TextLine] = []
    for size, text, bbox in lines:
        if len(text) < 3 or len(text) > MAX_TITLE_CHARS:
            continue
        if text.upper().strip(".:") in NOT_A_PLAN_EXACT:
            continue
        if NOT_A_PLAN_RE.search(text) or ADDRESS_RE.search(text):
            continue
        if SHEET_CODE_RE.fullmatch(text) or DATE_RE.fullmatch(text):
            continue
        if not re.search(r"[A-Za-z]", text):
            continue
        if size < MIN_TITLE_SIZE:
            continue
        candidates.append((size, text, bbox))

    if not candidates:
        return []
    largest = max(size for size, _text, _bbox in candidates)

    hints: list[WingHint] = []
    for size, text, bbox in candidates:
        if size < largest * TITLE_SIZE_TOLERANCE:
            continue
        name = _title_name(text)
        hints.append(
            WingHint(name=name, box=_pdf_to_norm(bbox, pdf_w, pdf_h, rotation))
        )
    return hints


def _room_tags_from_dict(
    lines: list[TextLine],
    pdf_w: float,
    pdf_h: float,
    rotation: int,
) -> list[WingHint]:
    """Collect room tags, named by the area prefix they carry."""
    tags: list[WingHint] = []
    for size, text, bbox in lines:
        match = ROOM_TAG_RE.fullmatch(text.upper())
        if not match or size > MAX_ROOM_TAG_SIZE:
            continue
        tags.append(
            WingHint(
                name=match.group("prefix"),
                box=_pdf_to_norm(bbox, pdf_w, pdf_h, rotation),
            )
        )
    return tags


def _in_title_block(box: list[float]) -> bool:
    """True when a label sits in the sheet's title-block column, not on a plan."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return cx >= 0.78 or (cx >= 0.70 and cy >= 0.78)


def expand_wing_hints(hints: list[WingHint]) -> list[WingHint]:
    """Grow label boxes into non-overlapping wing regions via midpoints."""
    if len(hints) < 2:
        return hints
    centers = []
    for hint in hints:
        x1, y1, x2, y2 = hint.box
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, hint.name, hint.box))
    out: list[WingHint] = []
    pad = 0.012
    for cx, cy, name, box in centers:
        left, right, top, bottom = 0.0, 1.0, 0.0, 1.0
        # Start from a generous box around the label
        half_x = max(0.18, (box[2] - box[0]) * 4)
        half_y = max(0.18, (box[3] - box[1]) * 4)
        left, right = cx - half_x, cx + half_x
        top, bottom = cy - half_y, cy + half_y
        for ox, oy, oname, _ob in centers:
            if abs(ox - cx) < 1e-6 and abs(oy - cy) < 1e-6:
                continue
            if abs(ox - cx) >= abs(oy - cy):
                mid = (cx + ox) / 2.0
                if ox < cx:
                    left = max(left, mid)
                else:
                    right = min(right, mid)
            else:
                mid = (cy + oy) / 2.0
                if oy < cy:
                    top = max(top, mid)
                else:
                    bottom = min(bottom, mid)
        region = [
            max(0.0, left - pad),
            max(0.0, top - pad),
            min(1.0, right + pad),
            min(1.0, bottom + pad),
        ]
        if region[2] - region[0] < 0.05 or region[3] - region[1] < 0.05:
            continue
        out.append(WingHint(name=name, box=region))
    return out


def scan_pdf_pages(pdf_path: Path) -> list[PageScan]:
    """Read every page's sheet code, plan titles and room tags."""
    import pymupdf

    results: list[PageScan] = []
    doc = pymupdf.open(pdf_path)
    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            # Text bboxes use unrotated PDF coordinates; rendered images honor
            # page.rotation. Keep both so label boxes map correctly to pixels.
            pdf_w, pdf_h = float(page.cropbox.width), float(page.cropbox.height)
            rotation = int(page.rotation)
            lines = _lines(page.get_text("dict") or {})
            sheet_no = _sheet_code(lines)
            # Titles and tags are read on every page; the vision model decides
            # which pages hold a drawing.
            labels = _wing_hints_from_dict(lines, pdf_w, pdf_h, rotation)
            labels = [
                label
                for label in labels
                if not _in_title_block(label.box)
            ]
            hints = expand_wing_hints(labels) if labels else []
            room_tags = _room_tags_from_dict(lines, pdf_w, pdf_h, rotation)
            named = ", ".join(hint.name for hint in labels)
            title = " ".join(part for part in (sheet_no, named) if part)
            has_plan_text = bool(labels)
            reason = (
                f"plan titles in text: {named}"
                if labels
                else f"{len(labels)} title(s), {len(room_tags)} room tag(s) in text"
            )
            results.append(
                PageScan(
                    page=index + 1,
                    is_diagram=has_plan_text,
                    sheet_no=sheet_no,
                    title=(title or f"page_{index + 1:03d}")[:120],
                    reason=reason[:200],
                    wing_hints=hints,
                    wing_labels=labels,
                    room_tags=room_tags,
                )
            )
    finally:
        doc.close()
    return _drop_boilerplate(results)


def _drop_boilerplate(pages: list[PageScan]) -> list[PageScan]:
    """Drop large-type lines that repeat across the set — those are the title block.

    Project name, district, consultant and sheet-family wording are set in big
    type on every page. Plan titles are not: they change from sheet to sheet.
    """
    from collections import Counter
    from dataclasses import replace

    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({label.name for label in page.wing_labels})
    floor = max(3, (len(pages) + 1) // 2)
    repeated = {name for name, n in counts.items() if n >= floor}
    if not repeated:
        return pages
    cleaned: list[PageScan] = []
    for page in pages:
        labels = [label for label in page.wing_labels if label.name not in repeated]
        if labels == page.wing_labels:
            cleaned.append(page)
            continue
        named = ", ".join(label.name for label in labels)
        title = " ".join(part for part in (page.sheet_no, named) if part)
        cleaned.append(
            replace(
                page,
                is_diagram=bool(labels),
                title=(title or f"page_{page.page:03d}")[:120],
                reason=(
                    f"plan titles in text: {named}"
                    if labels
                    else page.reason
                )[:200],
                wing_labels=labels,
                wing_hints=expand_wing_hints(labels) if labels else [],
            )
        )
    return cleaned
