"""Map raw title-block OCR text to structured project metadata.

Real title blocks are inconsistent: one template labels a field "Client",
another "Owner"; the engineer may appear as "Consultant" or "Building
Services"; a single firm can be listed as "Architect / Engineer". Instead of
per-field regexes, labels are fuzzy-matched (rapidfuzz) against a canonical
alias table, which handles both OCR noise (e.g. "CL1ENT") and template
variance. Extend ``FIELD_ALIASES`` to support new templates — no code change
needed.

Two title-block text layouts are supported:
- inline:   ``CLIENT: Meridian Property Group``
- stacked:  a label line followed by a value line (most CAD templates)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from rapidfuzz import fuzz

from ml.base import OcrLine

# Canonical schema fields -> known label spellings across drawing templates.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("project title", "drawing title", "project name", "title", "job title"),
    "client": ("client", "client name", "owner", "developer", "employer"),
    "architect": ("architect", "architect name", "architectural firm"),
    "engineer": (
        "engineer",
        "consultant",
        "mechanical engineer",
        "services engineer",
        "building services",
        "m&e consultant",
        "mep engineer",
    ),
    "project_address": (
        "project address",
        "address",
        "site address",
        "location",
        "site location",
    ),
    "due_date": (
        "due date",
        "issue date",
        "date",
        "issued",
        "completion date",
        "tender date",
    ),
}

# Two matching tiers: token-set matching for word-order/extra-word variance
# ("ARCHITECT / ENGINEER"), plain character ratio for OCR noise ("CL1ENT").
TOKEN_SET_CUTOFF = 85
CHAR_RATIO_CUTOFF = 80
# Real title-block labels are short; longer lines are values (company names,
# addresses), which must never be mistaken for labels.
MAX_LABEL_WORDS = 3
MAX_LABEL_CHARS = 40

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%m-%d-%y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%B %d %Y",
)


@dataclass
class ExtractedMetadata:
    """Structured title-block metadata (None = not found in the OCR text)."""

    title: str | None = None
    client: str | None = None
    architect: str | None = None
    engineer: str | None = None
    project_address: str | None = None
    due_date: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _alias_matches(label: str, alias: str) -> bool:
    return (
        fuzz.token_set_ratio(label, alias) >= TOKEN_SET_CUTOFF
        or fuzz.ratio(label, alias) >= CHAR_RATIO_CUTOFF
    )


def _match_fields(label: str) -> list[str]:
    """Fields this label refers to (usually one; several for combined labels
    like "ARCHITECT / ENGINEER"). Empty if nothing clears the cutoffs."""
    if len(label) > MAX_LABEL_CHARS or len(label.split()) > MAX_LABEL_WORDS:
        return []
    normalized = label.lower().strip(" :.-_")
    return [
        field
        for field, aliases in FIELD_ALIASES.items()
        if any(_alias_matches(normalized, alias) for alias in aliases)
    ]


def normalize_date(raw: str) -> str:
    """Normalize a date string to ISO ``YYYY-MM-DD``; return raw if unparseable."""
    candidate = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return candidate


def map_metadata(lines: list[OcrLine] | list[str]) -> ExtractedMetadata:
    """Extract structured metadata from OCR'd title-block lines."""
    texts = [
        (line.text if isinstance(line, OcrLine) else line).strip()
        for line in lines
    ]
    texts = [text for text in texts if text]

    result = ExtractedMetadata()

    def assign(fields: list[str], value: str) -> None:
        value = value.strip()
        if not value:
            return
        for field in fields:
            if getattr(result, field) is None:
                if field == "due_date":
                    value_out = normalize_date(value)
                else:
                    value_out = value
                setattr(result, field, value_out)

    index = 0
    while index < len(texts):
        text = texts[index]

        # Inline layout: "CLIENT: Meridian Property Group"
        if ":" in text:
            label, _, value = text.partition(":")
            fields = _match_fields(label)
            if fields and value.strip():
                assign(fields, value)
                index += 1
                continue

        # Stacked layout: label line followed by a value line
        fields = _match_fields(text)
        if fields and index + 1 < len(texts):
            next_text = texts[index + 1]
            if not _match_fields(next_text):  # next line is a value, not a label
                assign(fields, next_text)
                index += 2
                continue

        index += 1

    return result
