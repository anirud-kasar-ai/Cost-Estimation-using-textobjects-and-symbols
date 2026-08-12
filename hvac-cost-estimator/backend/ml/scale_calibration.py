"""Graphic scale-bar OCR → feet-per-pixel calibration for linear takeoffs.

Looks for common architectural scale strings on the sheet (e.g.
``1/4" = 1'-0"``, ``SCALE: 1/8\"=1'-0\"``) and converts raceway / conduit
detections from pixel length to linear feet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ml.base import BoundingBox, ClassifiedDevice, OcrLine

# 1/4" = 1'-0"  |  1/8"=1'  |  SCALE 3/32" = 1'-0"
_SCALE_RE = re.compile(
    r"(?:SCALE[:\s]*)?"
    r"(?P<num>\d+)\s*/\s*(?P<den>\d+)\s*[\"″]?\s*"
    r"(?:=\s*|TO\s+)"
    r"(?P<feet>\d+)\s*['′]?\s*-\s*(?P<inches>\d+)\s*[\"″]?",
    re.IGNORECASE,
)

# Fallback: "SCALE 1:100" (unitless ratio — treat as 1 drawing unit = N real units
# only when we also know DPI; for LF we require imperial graphic scale above).
_RATIO_RE = re.compile(r"SCALE\s*[:=]?\s*1\s*:\s*(\d+)", re.IGNORECASE)

LINE_TYPE_DEVICES = frozenset({"raceway", "conduit", "cable_tray"})


@dataclass(frozen=True)
class ScaleCalibration:
    """Feet of real length represented by one image pixel."""

    feet_per_pixel: float
    label: str
    source: str  # "ocr" | "mock" | "default"


def parse_graphic_scale(text: str) -> tuple[float, str] | None:
    """Parse an imperial graphic-scale string into inches-on-drawing per foot.

    Returns ``(drawing_inches_per_real_foot, normalized_label)`` or None.
    """
    match = _SCALE_RE.search(text.replace("″", '"').replace("′", "'"))
    if not match:
        return None
    num = int(match.group("num"))
    den = int(match.group("den"))
    feet = int(match.group("feet"))
    inches = int(match.group("inches"))
    if den <= 0:
        return None
    drawing_inches = num / den
    real_feet = feet + inches / 12.0
    if real_feet <= 0:
        return None
    # drawing_inches on paper represents real_feet of building length
    drawing_inches_per_real_foot = drawing_inches / real_feet
    label = f'{num}/{den}" = {feet}\'-{inches}"'
    return drawing_inches_per_real_foot, label


def feet_per_pixel_from_scale(
    drawing_inches_per_real_foot: float,
    dpi: float,
) -> float:
    """Convert graphic scale + raster DPI into feet per image pixel."""
    if dpi <= 0 or drawing_inches_per_real_foot <= 0:
        raise ValueError("dpi and scale must be positive")
    # pixels for one real foot = drawing_inches_per_real_foot * dpi
    pixels_per_foot = drawing_inches_per_real_foot * dpi
    return 1.0 / pixels_per_foot


def calibrate_from_ocr(
    ocr_lines: list[OcrLine],
    *,
    dpi: float,
    use_mock_fallback: bool = True,
) -> ScaleCalibration:
    """Find a graphic scale string in OCR output and return ft/px."""
    for line in ocr_lines:
        parsed = parse_graphic_scale(line.text)
        if parsed is None:
            continue
        drawing_in_per_ft, label = parsed
        return ScaleCalibration(
            feet_per_pixel=feet_per_pixel_from_scale(drawing_in_per_ft, dpi),
            label=label,
            source="ocr",
        )

    if use_mock_fallback:
        # Common school bid default: 1/8" = 1'-0" at the configured PDF DPI
        drawing_in_per_ft, label = parse_graphic_scale('1/8" = 1\'-0"')  # type: ignore[misc]
        assert drawing_in_per_ft is not None
        return ScaleCalibration(
            feet_per_pixel=feet_per_pixel_from_scale(drawing_in_per_ft, dpi),
            label=label,
            source="mock",
        )

    # Last resort: 1/4" = 1'-0"
    drawing_in_per_ft, label = parse_graphic_scale('1/4" = 1\'-0"')  # type: ignore[misc]
    assert drawing_in_per_ft is not None
    return ScaleCalibration(
        feet_per_pixel=feet_per_pixel_from_scale(drawing_in_per_ft, dpi),
        label=label,
        source="default",
    )


def _box_length_px(box: BoundingBox) -> float:
    return max(abs(box.x2 - box.x1), abs(box.y2 - box.y1))


def apply_linear_feet(
    devices: list[ClassifiedDevice],
    calibration: ScaleCalibration,
) -> list[ClassifiedDevice]:
    """Convert raceway-like detections from pixel span → linear feet quantity."""
    updated: list[ClassifiedDevice] = []
    for device in devices:
        if device.device_type not in LINE_TYPE_DEVICES:
            updated.append(device)
            continue
        length_px = _box_length_px(device.detection.box)
        lf = max(0.5, round(length_px * calibration.feet_per_pixel, 1))
        updated.append(
            replace(
                device,
                quantity=lf,
                unit="LF",
                category=device.category or "Raceway",
            )
        )
    return updated


def is_line_type_box(box: BoundingBox, *, min_aspect: float = 3.5) -> bool:
    """Heuristic: elongated boxes are treated as raceway / conduit spans."""
    w = abs(box.x2 - box.x1)
    h = abs(box.y2 - box.y1)
    short = min(w, h) or 1.0
    return max(w, h) / short >= min_aspect
