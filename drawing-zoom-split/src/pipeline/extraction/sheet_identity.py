"""Classify technology floor-plan sheets and derive wing/folder names.

Uses PDF embedded text when available (even if CV OCR is mocked), so wing
names like ``C-WING (EAST)`` come from the real drawing title block / sheet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pipeline.extraction.base import OcrLine

logger = logging.getLogger(__name__)


class SheetKind(str, Enum):
    TECHNOLOGY_FLOOR_PLAN = "technology_floor_plan"
    LEGEND = "legend"
    OTHER = "other"


@dataclass(frozen=True)
class FloorPlanName:
    """Human display name + filesystem slug for one floor-plan sheet."""

    display_name: str
    slug: str


# C-WING (EAST), A-WING, etc. (common on tech sets)
_DASH_WING_RE = re.compile(
    r"\b([A-Z])-WING(?:[^\S\n]*\(([^)]+)\))?",
    re.IGNORECASE,
)
# WING C / C WING — same-line only (avoid "PLAN\nWING" across newlines)
_WING_RE = re.compile(
    r"\b(?:"
    r"WING[^\S\n]+([A-Z0-9\-]{1,8})"
    r"|([A-Z0-9\-]{1,8})[^\S\n]+WING"
    r")\b",
    re.IGNORECASE,
)
_BUILDING_RE = re.compile(
    r"\b(?:MULTI-USE\s+)?BUILDING\s+([A-Z0-9]{1,4})(?!\s+SERVICES)\b"
    r"|\bMULTI-USE\s+BUILDING\b",
    re.IGNORECASE,
)
# Prefer hyphenated sheet numbers (T-703); bare T1100 is usually noise
_DRAWING_NO_RE = re.compile(r"\b(T-\d{3,4})\b", re.IGNORECASE)
# Title-block "SHEET NO." often sits just above the real T-xxx
_SHEET_NO_NEAR_RE = re.compile(
    r"SHEET\s*NO\.?\s*[:\-]?\s*(T-\d{3,4})\b",
    re.IGNORECASE,
)

_LEGEND_RE = re.compile(
    r"\b(?:"
    r"SYMBOL[^\S\n]+LEGEND|TECHNOLOGY[^\S\n]+SYMBOL|LEGEND\s*-|ABBREVIATION|"
    r"GENERAL[^\S\n]+NOTES"
    r")\b",
    re.IGNORECASE,
)

# Optional bridge on Architect CD sets:
# TECHNOLOGY / INFRASTRUCTURE / FLOOR PLANS DEMO  (also typo INFRASTUCTURE)
_INFRA_BRIDGE = r"(?:INFRA(?:STUCTURE|STRUCTURE)?\.?\s+)?"

# All diagram sheet titles (\s matches line breaks: TECHNOLOGY\nSITE PLAN NEW)
_DIAGRAM_SHEET_TITLE_RE = re.compile(
    r"\b(?:"
    r"TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"FLOOR\s+PLANS?(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|TECH\s+FLOOR\s+PLANS?(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"SITE\s+PLANS?(?:\s*[-–—:]?\s*(?:NEW|DEMO|EXISTING|EXTERIOR)[\w\s/]*)?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"RCP(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"REFLECTED\s+CEILING\s+PLANS?(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"SINGLE\s+LINES?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"DETAILS?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"RACK\s+ELEVATIONS?"
    r"|(?:DATA|VOICE|CCTV|INTERCOM|NETWORK)\s+SINGLE\s+LINE\s+DIAGRAM"
    r"|SINGLE\s+LINE\s+DIAGRAM"
    r"|HVAC\s+(?:FLOOR\s+)?(?:PLAN|LAYOUT)"
    # Garland roof sets
    r"|ROOF\s+PLAN\b"
    # Architectural / ACT-shade sets (1962) — not "LEGEND - FLOOR PLAN"
    r"|REFLECTED\s+CEILING\s+PLAN"
    r"|CEILING\s+PLAN\s*[-–—:]"
    r"|FLOOR\s+PLAN\s*[-–—:]"
    r"|FLOOR\s+PLANS?\s+RS\b"
    r"|(?:WINDOW\s+)?ROLLER\s+SHADE\s+FLOOR\s+PLAN"
    r"|DETAILS\s*[-–—:]\s*ROLLER"
    r"|[-–—]\s*DETAILS?\b"
    # Campus / site modernization (1963)
    r"|PARTIAL\s+SITE\s+PLAN"
    r"|OVERALL\s+CAMPUS\s+SITE"
    r"|CODE\s+ANALYSIS\s+PLAN"
    r"|ENLARGED\s+(?:DROP\s+OFF\s+)?SITE\s+PLAN"
    r")",
    re.IGNORECASE,
)

# Non-diagram alone — skipped unless a real diagram title is also present
_NON_DIAGRAM_SHEET_RE = re.compile(
    r"(?:"
    r"\bPICTURE\s+DETAIL\b"
    r"|\bPHOTO(?:GRAPHIC)?\s+DETAIL\b"
    r"|\bDEMO\s+TAG\s+NOTES\b"
    r"|\bSHEET\s+T-\d+\s+DEMO\s+TAG\s+NOTES\b"
    r")",
    re.IGNORECASE,
)

_TECH_FLOOR_RE = _DIAGRAM_SHEET_TITLE_RE

# Short label for folder naming — prefer title-block forms over body diagram captions
_DIAGRAM_KIND_LABEL_RE = re.compile(
    r"\b(?P<label>"
    r"TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"FLOOR\s+PLANS?(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"SITE\s+PLANS?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"RCP(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"REFLECTED\s+CEILING\s+PLANS?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"SINGLE\s+LINES?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"DETAILS?"
    r"|TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"RACK\s+ELEVATIONS?"
    r"|SINGLE\s+LINE\s+DIAGRAM"
    r"|HVAC\s+(?:FLOOR\s+)?(?:PLAN|LAYOUT)"
    r"|ROOF\s+PLAN"
    r"|REFLECTED\s+CEILING\s+PLAN"
    r"|FLOOR\s+PLAN"
    r"|PARTIAL\s+SITE\s+PLAN"
    r"|OVERALL\s+CAMPUS\s+SITE"
    r"|CODE\s+ANALYSIS\s+PLAN"
    r")\b",
    re.IGNORECASE,
)

# Title-block sheet kinds only (Architect CD + 1948). Prefer over body captions.
_TITLE_BLOCK_KIND_RE = re.compile(
    r"\b(?P<label>"
    r"TECHNOLOGY\s+"
    + _INFRA_BRIDGE
    + r"(?:"
    r"FLOOR\s+PLANS?(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|SITE\s+PLANS?"
    r"|RCP(?:\s*(?:NEW|DEMO|EXISTING))?"
    r"|REFLECTED\s+CEILING\s+PLANS?"
    r"|SINGLE\s+LINES?"
    r"|DETAILS?"
    r"|RACK\s+ELEVATIONS?"
    r")"
    r"|ROOF\s+PLAN"
    r"|REFLECTED\s+CEILING\s+PLAN"
    r"|FLOOR\s+PLAN\s*[-–—:][^\n]{0,60}"
    r"|PARTIAL\s+SITE\s+PLAN"
    r"|OVERALL\s+CAMPUS\s+SITE"
    r")\b",
    re.IGNORECASE,
)

_STRIP_TECH_PREFIX_RE = re.compile(
    r"^TECHNOLOGY\s+(?:INFRA(?:STUCTURE|STRUCTURE)?\.?\s+)?",
    re.IGNORECASE,
)

_SLUG_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str) -> str:
    cleaned = " ".join(name.strip().split())
    cleaned = cleaned.replace(" ", "_")
    cleaned = _SLUG_BAD.sub("_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._-") or "SHEET"
    return cleaned[:80]


def _joined_text(ocr_lines: list[OcrLine] | str) -> str:
    if isinstance(ocr_lines, str):
        return ocr_lines
    return "\n".join(line.text for line in ocr_lines)


def pdf_page_text_lines(pdf_path: Path, page_number: int) -> list[OcrLine]:
    """Extract embedded PDF text for a 1-based page as OcrLine list (no boxes)."""
    try:
        import fitz
    except ImportError:
        logger.debug("PyMuPDF not available for sheet identity text")
        return []
    try:
        doc = fitz.open(pdf_path)
        try:
            if page_number < 1 or page_number > doc.page_count:
                return []
            text = doc.load_page(page_number - 1).get_text("text") or ""
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("pdf_page_text_lines failed page %s: %s", page_number, exc)
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [OcrLine(text=ln, confidence=1.0) for ln in lines]


def _has_diagram_sheet_title(text: str) -> bool:
    """True when the sheet title is any technology/HVAC diagram (not notes-only)."""
    return bool(_DIAGRAM_SHEET_TITLE_RE.search(text))


def classify_sheet(
    ocr_lines: list[OcrLine] | str,
    *,
    keywords: list[str] | None = None,
) -> SheetKind:
    """Classify a sheet from PDF/OCR text.

    Keeps **every diagram sheet**: floor plans, site plans, RCP/ceiling plans,
    single-line diagrams, and detail sheets. Skips notes-only, picture-detail
    alone, legends, and index sheets so ROI/zooms run only on drawings.
    """
    del keywords  # reserved; diagram titles are authoritative
    text = _joined_text(ocr_lines)
    upper = text.upper()
    has_diagram_title = _has_diagram_sheet_title(text)

    # Index / cover — same-line only (\s would match across "TITLE" … "SHEET NO")
    if re.search(
        r"SHEET[^\S\n]+INDEX|COVER[^\S\n]+SHEET|TITLE[^\S\n]+SHEET|"
        r"TECHNOLOGY[^\S\n]+SHEET[^\S\n]+INDEX|DRAWING[^\S\n]+INDEX",
        upper,
    ):
        return SheetKind.LEGEND

    # Picture-detail / demo-tag table sheets (no plan diagram title)
    if _NON_DIAGRAM_SHEET_RE.search(text) and not has_diagram_title:
        return SheetKind.OTHER

    # Notes / legend / abbreviations without a diagram sheet title
    if not has_diagram_title:
        if _LEGEND_RE.search(text):
            return SheetKind.LEGEND
        if "SYMBOL LEGEND" in upper or "TECHNOLOGY SYMBOL" in upper:
            return SheetKind.LEGEND
        return SheetKind.OTHER

    # Any recognized diagram title → process (ROI + zooms)
    return SheetKind.TECHNOLOGY_FLOOR_PLAN


def _wing_labels(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _DASH_WING_RE.finditer(text):
        letter = m.group(1).upper()
        detail = (m.group(2) or "").strip()
        label = f"{letter}-Wing"
        if detail:
            label = f"{label} ({detail.title()})"
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            found.append(label)
    if found:
        return found
    for m in _WING_RE.finditer(text):
        token = (m.group(1) or m.group(2) or "").strip().upper().rstrip(".-")
        if not token or token == "AWING":
            continue
        label = f"Wing {token}"
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            found.append(label)
    return found


def _drawing_number(text: str) -> str | None:
    """Pick the title-block sheet number (not note callouts like 4/T-801)."""
    near = _SHEET_NO_NEAR_RE.search(text)
    if near:
        return near.group(1).upper()

    matches = list(_DRAWING_NO_RE.finditer(text))
    if not matches:
        return None

    # Architect CD / most tech sets put the real sheet number last in the
    # title block; earlier T-xxx values are usually detail references.
    title = _TITLE_BLOCK_KIND_RE.search(text) or _DIAGRAM_KIND_LABEL_RE.search(
        text
    ) or _DIAGRAM_SHEET_TITLE_RE.search(text)
    if title is not None:
        # Prefer the last T-xxx that appears at or after the title-block kind
        after = [m for m in matches if m.start() >= title.start()]
        if after:
            return after[-1].group(1).upper()
        # Title appears after body callouts — still prefer overall last number
        return matches[-1].group(1).upper()

    return matches[-1].group(1).upper()


def extract_floor_plan_name(
    ocr_lines: list[OcrLine] | str,
    page_number: int,
    *,
    used_slugs: set[str] | None = None,
) -> FloorPlanName:
    """Derive display name + unique slug (drawing no + wings when present)."""
    text = _joined_text(ocr_lines)
    parts: list[str] = []

    drawing = _drawing_number(text)
    if drawing:
        parts.append(drawing)

    kind_m = _TITLE_BLOCK_KIND_RE.search(text) or _DIAGRAM_KIND_LABEL_RE.search(text)
    kind_label = ""
    if kind_m:
        label = " ".join(kind_m.group("label").split())
        kind_label = _STRIP_TECH_PREFIX_RE.sub("", label).strip()
        kind_label = kind_label.title() if kind_label else ""
        if kind_label and kind_label.casefold() not in " ".join(parts).casefold():
            parts.append(kind_label)

    # Wings only for floor / RCP plans (not site / single-line / details)
    if re.search(r"FLOOR\s+PLANS?|\bRCP\b|REFLECTED\s+CEILING", kind_label, re.I):
        for w in _wing_labels(text)[:3]:
            parts.append(w)

    if not parts:
        if _DIAGRAM_SHEET_TITLE_RE.search(text):
            m = _DIAGRAM_SHEET_TITLE_RE.search(text)
            parts.append(m.group(0).title() if m else "Technology Diagram")
        elif "MULTI-USE BUILDING" in text.upper():
            parts.append("Multi-Use Building")
        else:
            parts.append(f"SHEET_{page_number:02d}")

    display = " ".join(parts)
    base_slug = slugify(display)
    slug = base_slug
    used = used_slugs if used_slugs is not None else set()
    if slug in used:
        slug = f"{base_slug}_p{page_number:02d}"
    n = 2
    while slug in used:
        slug = f"{base_slug}_p{page_number:02d}_{n}"
        n += 1
    used.add(slug)
    return FloorPlanName(display_name=display, slug=slug)


def resolve_unique_slug(slug: str, used_slugs: set[str], page_number: int) -> str:
    """Ensure slug uniqueness (mutates ``used_slugs``)."""
    candidate = slug
    if candidate not in used_slugs:
        used_slugs.add(candidate)
        return candidate
    candidate = f"{slug}_p{page_number:02d}"
    n = 2
    while candidate in used_slugs:
        candidate = f"{slug}_p{page_number:02d}_{n}"
        n += 1
    used_slugs.add(candidate)
    return candidate


@dataclass(frozen=True)
class SheetScanEntry:
    """One page from a whole-PDF diagram scan."""

    page_number: int
    sheet_kind: SheetKind
    name: FloorPlanName
    is_diagram: bool


@dataclass(frozen=True)
class PdfSheetScan:
    """Result of analyzing every page before ROI / zoom work."""

    total_pages: int
    entries: tuple[SheetScanEntry, ...]

    @property
    def diagram_entries(self) -> list[SheetScanEntry]:
        return [e for e in self.entries if e.is_diagram]

    @property
    def skipped_entries(self) -> list[SheetScanEntry]:
        return [e for e in self.entries if not e.is_diagram]

    def by_page(self) -> dict[int, SheetScanEntry]:
        return {e.page_number: e for e in self.entries}

    def to_manifest(self) -> dict:
        return {
            "total_pages": self.total_pages,
            "diagram_count": len(self.diagram_entries),
            "skipped_count": len(self.skipped_entries),
            "diagram_pages": [
                {
                    "page_number": e.page_number,
                    "display_name": e.name.display_name,
                    "slug": e.name.slug,
                    "sheet_kind": e.sheet_kind.value,
                }
                for e in self.diagram_entries
            ],
            "skipped_pages": [
                {
                    "page_number": e.page_number,
                    "reason": e.sheet_kind.value,
                    "display_name": e.name.display_name,
                }
                for e in self.skipped_entries
            ],
        }


def scan_pdf_diagram_pages(pdf_path: Path) -> PdfSheetScan:
    """Phase 1: classify every page in the PDF from embedded text.

    Identifies diagram sheets (floor / site / RCP / single-line / details) vs
    cover, index, legend, and notes-only sheets **before** ROI or zoom work.
    """
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF unavailable; cannot scan diagram pages")
        return PdfSheetScan(total_pages=0, entries=())

    used_slugs: set[str] = set()
    entries: list[SheetScanEntry] = []
    try:
        doc = fitz.open(pdf_path)
        try:
            total = int(doc.page_count)
            for page_number in range(1, total + 1):
                text = doc.load_page(page_number - 1).get_text("text") or ""
                lines = [
                    OcrLine(text=ln.strip(), confidence=1.0)
                    for ln in text.splitlines()
                    if ln.strip()
                ]
                kind = classify_sheet(lines)
                name = extract_floor_plan_name(
                    lines, page_number, used_slugs=used_slugs
                )
                is_diagram = kind == SheetKind.TECHNOLOGY_FLOOR_PLAN
                entries.append(
                    SheetScanEntry(
                        page_number=page_number,
                        sheet_kind=kind,
                        name=name,
                        is_diagram=is_diagram,
                    )
                )
                logger.info(
                    "Sheet scan p%02d → %s (%s)%s",
                    page_number,
                    kind.value,
                    name.display_name,
                    " [diagram]" if is_diagram else "",
                )
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("scan_pdf_diagram_pages failed for %s: %s", pdf_path, exc)
        return PdfSheetScan(total_pages=0, entries=())

    scan = PdfSheetScan(total_pages=len(entries), entries=tuple(entries))
    logger.info(
        "PDF sheet scan %s: %d pages, %d diagrams, %d skipped",
        pdf_path.name,
        scan.total_pages,
        len(scan.diagram_entries),
        len(scan.skipped_entries),
    )
    return scan
