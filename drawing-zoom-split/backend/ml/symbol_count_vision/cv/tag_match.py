"""Match plan OCR spans to legend character tags and part-number aliases.

Legend rows are not only graphical icons. Many are letter tags (G, J, AP,
NVR) or part labels drawn on the plan as text (WM2300 → vertical \"2300\").
"""

from __future__ import annotations

import re

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo

SHORT_TAG_MAX_SIZE = 30.0
SHORT_TAG_MIN_SIZE = 6.0
TITLE_FONTS = ("romantic", "times")

# Part/model strings that are notes, not drawable labels on the plan.
_JUNK_PART_RE = re.compile(
    r"^(?:SEE\b|EXISTING|N/?A|OFCI|OFOI|CFCI|GENERIC|COMMERCIAL|"
    r"ARUBA|AXIS|PANDUIT|WIREMOLD|BOGEN|LEGRAND|CHATSWORTH)\b",
    re.IGNORECASE,
)
_PART_CORE_RE = re.compile(r"^[A-Z]{0,4}\d{3,6}[A-Z0-9\-]*$", re.IGNORECASE)


def _display_key(symbol: str | None, description: str) -> str:
    return str(symbol or description or "").strip()


def count_key(entry: SymbolEntry) -> str:
    """Stable counting id: tag, or description(+part) when parts distinguish rows."""
    tag = (entry.symbol or "").strip()
    if tag:
        return tag
    desc = (entry.description or "").strip()
    part = (entry.part_number or "").strip()
    if desc and part and not _JUNK_PART_RE.match(part):
        return f"{desc} ({part})"
    return desc or part


def _part_aliases(part: str) -> list[str]:
    """Aliases a part number may appear as on the plan (e.g. WM2300 → 2300)."""
    raw = (part or "").strip().upper()
    if not raw or _JUNK_PART_RE.match(raw):
        return []
    # Drop decorative suffixes like **V / (E)
    core = re.sub(r"\*+$", "", raw)
    core = re.sub(r"\(.*?\)", "", core).strip("-_/ ")
    if not core or not _PART_CORE_RE.match(core):
        return []
    aliases = {core}
    # Wiremold-style: WM2300 also prints as 2300 next to raceway
    m = re.match(r"^([A-Z]{1,4})(\d{3,6})$", core)
    if m:
        aliases.add(m.group(2))
    return sorted(aliases, key=len, reverse=True)


def tagged_legend_keys(legend: SymbolTableInfo) -> list[tuple[str, str]]:
    """(TAG, count_key) for entries that have a countable letter/number tag."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in legend.entries:
        tag = (entry.symbol or "").strip().upper()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append((tag, count_key(entry)))
    out.sort(key=lambda item: (-len(item[0]), item[0]))
    return out


def text_legend_keys(legend: SymbolTableInfo) -> list[tuple[str, str]]:
    """All text aliases that should be counted: tags + part-number forms.

    Returns (ALIAS_UPPER, count_key) sorted longest-first so WM2300 wins
    over a bare 2300 when both exist.
    """
    out: list[tuple[str, str]] = []
    seen_alias: set[str] = set()

    for entry in legend.entries:
        key = count_key(entry)
        if not key:
            continue
        candidates: list[str] = []
        tag = (entry.symbol or "").strip().upper()
        if tag:
            candidates.append(tag)
        for alias in _part_aliases(entry.part_number or ""):
            candidates.append(alias)

        for alias in candidates:
            if alias in seen_alias:
                continue
            # Never treat the drop-quantity marker as a text alias here —
            # triangles + digit OCR own '#'.
            if alias == "#":
                continue
            seen_alias.add(alias)
            out.append((alias, key))

    out.sort(key=lambda item: (-len(item[0]), item[0]))
    return out


def match_span_to_tag(
    text: str,
    tags: list[tuple[str, str]],
    *,
    size: float | None = None,
    font: str | None = None,
) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    font_l = (font or "").lower()
    if any(name in font_l for name in TITLE_FONTS):
        return None
    if size is not None and (size < SHORT_TAG_MIN_SIZE or size > SHORT_TAG_MAX_SIZE):
        return None
    upper = raw.upper()
    # Plan labels sometimes append a short dash: "2300-"
    upper = upper.strip("-–—_/ ")

    for tag, display in tags:
        if tag == "#":
            if upper == "#" or re.fullmatch(r"#\d+", upper):
                return display
            continue
        matched = upper == tag or bool(
            re.fullmatch(re.escape(tag) + r"(?:-\d+|\d+)$", upper)
        )
        if matched:
            return display
    return None


def match_span_to_text_key(
    text: str,
    keys: list[tuple[str, str]],
    *,
    size: float | None = None,
) -> str | None:
    """Match OCR text to tag or part-number aliases."""
    return match_span_to_tag(text, keys, size=size)
