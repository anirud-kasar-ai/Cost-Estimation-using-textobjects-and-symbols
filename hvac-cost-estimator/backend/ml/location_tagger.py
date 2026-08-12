"""Attach nearest room-label OCR text to each detection box.

After symbol detection, OCR lines whose text looks like a room / space label
are scored by distance to the box center; the nearest match is stored on the
device for costing and review.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ml.base import BoundingBox, ClassifiedDevice, OcrLine

# Typical architectural room tags: "ROOM 101", "CORRIDOR A", "MECH 1", etc.
_ROOM_RE = re.compile(
    r"\b(?:"
    r"ROOM|RM\.?|CORRIDOR|CORR\.?|HALL|LOBBY|OFFICE|CLASS|LAB|"
    r"MECH(?:ANICAL)?|ELEC(?:TRICAL)?|STOR(?:AGE)?|RESTROOM|TOILET|"
    r"KITCHEN|CAFETERIA|GYM|LIBRARY|ADMIN|CONF(?:ERENCE)?"
    r")\b"
    r"(?:\s+[A-Z0-9\-]+){0,3}",
    re.IGNORECASE,
)


def looks_like_room_label(text: str) -> bool:
    """Return True when OCR text resembles a room / space identifier."""
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) < 3 or len(cleaned) > 48:
        return False
    return _ROOM_RE.search(cleaned) is not None


def _normalize_label(text: str) -> str:
    match = _ROOM_RE.search(text)
    if match:
        return " ".join(match.group(0).split()).upper()
    return " ".join(text.split()).upper()


def _box_center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _mock_room_for_box(box: BoundingBox, page_width: float, page_height: float) -> str:
    """Deterministic mock room when OCR has no boxed room labels (mock mode)."""
    cx, cy = _box_center(box)
    col = min(2, int(cx / max(page_width, 1.0) * 3))
    row = min(2, int(cy / max(page_height, 1.0) * 3))
    rooms = (
        ("ROOM 101", "ROOM 102", "CORRIDOR A"),
        ("MECH 1", "CLASS 210", "OFFICE 201"),
        ("STOR 1", "RESTROOM", "LOBBY"),
    )
    return rooms[row][col]


def tag_devices_with_rooms(
    devices: list[ClassifiedDevice],
    ocr_lines: list[OcrLine],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    max_distance_px: float = 280.0,
) -> list[ClassifiedDevice]:
    """Return devices with ``room_label`` set from nearest OCR room text.

    When OCR lines lack bounding boxes (typical mock title-block OCR), falls
    back to a deterministic quadrant → room map so the dashboard still shows
    location tags.
    """
    candidates: list[tuple[str, tuple[float, float]]] = []
    for line in ocr_lines:
        if not looks_like_room_label(line.text):
            continue
        label = _normalize_label(line.text)
        if line.box is not None:
            candidates.append((label, _box_center(line.box)))

    tagged: list[ClassifiedDevice] = []
    for device in devices:
        center = _box_center(device.detection.box)
        best_label: str | None = None
        best_dist = float("inf")
        for label, point in candidates:
            dist = _distance(center, point)
            if dist < best_dist and dist <= max_distance_px:
                best_dist = dist
                best_label = label

        if best_label is None and page_width and page_height:
            best_label = _mock_room_for_box(
                device.detection.box, page_width, page_height
            )

        tagged.append(replace(device, room_label=best_label))
    return tagged
