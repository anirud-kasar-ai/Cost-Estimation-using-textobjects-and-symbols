"""Best-effort client (Bill To) extraction from a drawing set's cover sheet.

Construction drawing title blocks usually name the owner — for school work a
district such as "Mt. Diablo Unified School District" — followed by its
address. The text layout varies per architect, so this is heuristic:

1. Prefer an owner line ending in "DISTRICT", stitching together names that
   printers split across lines ("MOUNT DIABLO UNIFIED" / "SCHOOL DISTRICT").
2. Take the street + city/state/zip lines printed next to that owner block.
3. Otherwise fall back to the first plausible address on the sheet with a
   title-like line above it, skipping known vendor/architect blocks.

Returns None when nothing convincing is found; the UI lets users fill the
client in manually either way.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# "2351 OLIVERA RD, CONCORD, CA 94520" — full address on one line.
_ADDR_ONE = re.compile(r"^\d{1,6}\s+.+,\s*[A-Za-z]{2}\.?,?\s+\d{5}(?:-\d{4})?\s*$")
# "1936 Carlotta Dr." — street on its own line.
_STREET = re.compile(r"^\d{1,6}\s+[A-Za-z0-9 .'\-]+$")
# "Concord, CA 94519" — city/state/zip on its own line.
_CITY = re.compile(r"^[A-Za-z .'\-]+,\s*[A-Za-z]{2}\.?\s*\d{5}(?:-\d{4})?\s*$")
# Owner block candidate: line ending in DISTRICT.
_DISTRICT_END = re.compile(r"\bDISTRICT\s*$", re.IGNORECASE)
# Title-like line usable as (part of) a client name: words only, no digits.
_NAME_LIKE = re.compile(r"^[A-Za-z][A-Za-z .,&'\-]{2,44}$")
# Lines that belong to vendor/architect/sheet furniture, never the client.
_NOISE = re.compile(
    r"PHONE|FAX|TEL\b|WWW\.|\.COM|ARCHITECT|GARLAND|CONSULTANT|REVISION|SHEET|"
    r"DRAWING|SCALE|DATE|DWG|CHK|STAMP|APPROVAL|OWNER & TITLE|INSPECTOR|SPECIFICATION",
    re.IGNORECASE,
)


def _page_lines(pdf_path: Path) -> list[str]:
    import fitz

    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            return []
        text = doc[0].get_text() or ""
    # Drop empties and single characters (vertically-printed stamps).
    return [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 1]


def _address_near(lines: list[str], start: int, stop: int) -> str | None:
    """Scan lines[start:stop] for a one-line or street+city address."""
    for k in range(max(0, start), min(stop, len(lines))):
        line = lines[k]
        if _ADDR_ONE.match(line):
            return line
        if _STREET.match(line) and k + 1 < len(lines) and _CITY.match(lines[k + 1]):
            street, city_line = line.rstrip(","), lines[k + 1]
            city = city_line.split(",")[0].strip()
            if street.upper().endswith(f" {city.upper()}"):
                street = street[: -len(city) - 1].rstrip(" ,.")
            return f"{street}, {city_line}"
    return None


def _tidy_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,.-")
    return name.title() if name.isupper() else name


def extract_client_info(pdf_path: Path) -> dict[str, str] | None:
    try:
        lines = _page_lines(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("client_info: cannot read %s: %s", pdf_path, exc)
        return None
    if not lines:
        return None

    # --- Preferred: owner block ending in "DISTRICT" -----------------------
    for i, line in enumerate(lines):
        if not _DISTRICT_END.search(line) or len(line) > 48 or _NOISE.search(line):
            continue
        name = line
        j = i - 1
        # Stitch names split across lines until it reads like a full owner.
        while len(name.split()) < 4 and j >= 0 and j >= i - 2:
            prev = lines[j]
            if _NAME_LIKE.match(prev) and not _NOISE.search(prev):
                name = f"{prev} {name}"
                j -= 1
            else:
                break
        if len(name.split()) < 3:
            continue  # a stray "DISTRICT" with nothing usable above it
        address = _address_near(lines, i + 1, i + 7) or _address_near(lines, i - 4, i)
        return {"name": _tidy_name(name), "address": address or "", "contact": ""}

    # --- Fallback: first clean address with a title-like line above --------
    for k, line in enumerate(lines):
        if not _ADDR_ONE.match(line):
            continue
        context = lines[max(0, k - 3) : k + 2]
        if any(_NOISE.search(c) for c in context):
            continue
        name = next(
            (
                lines[j]
                for j in range(k - 1, max(-1, k - 4), -1)
                if _NAME_LIKE.match(lines[j]) and not _NOISE.search(lines[j])
            ),
            None,
        )
        if name:
            return {"name": _tidy_name(name), "address": line, "contact": ""}

    return None
