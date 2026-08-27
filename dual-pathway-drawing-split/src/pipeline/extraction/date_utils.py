"""Date normalization helpers for requirement extraction."""

from __future__ import annotations

from datetime import datetime

DATE_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def normalize_date(raw: str) -> str:
    """Normalize a date string to ISO ``YYYY-MM-DD``; return raw if unparseable."""
    candidate = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return candidate
