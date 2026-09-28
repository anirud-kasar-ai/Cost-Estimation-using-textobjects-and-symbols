"""Regression: two-digit callouts (14/15) + Shadeland technical-symbol PDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.callout_ocr import _ocr_digits_in_crop, _pick_best_digit_string
from pipeline.legend_ocr import parse_legend_file, parse_legend_lines

ROOT = Path(__file__).resolve().parents[1]
SHADELAND_PDF = (
    ROOT.parent
    / "drawing-zoom-split"
    / "storage"
    / "technical_symbols"
    / "1952BIDDrawings-Shadeland-Sunrise010625 technical symbol.pdf"
)


def _render_circled_number(text: str, size: int = 96) -> np.ndarray:
    """Synthetic hollow circle with centered digits (grayscale)."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    margin = 6
    draw.ellipse([margin, margin, size - margin - 1, size - margin - 1], outline=0, width=4)
    try:
        font = ImageFont.truetype("arialbd.ttf", 36 if len(text) > 1 else 42)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 36 if len(text) > 1 else 42)
        except OSError:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (size - tw) // 2 - bbox[0]
    ty = (size - th) // 2 - bbox[1] - 2
    draw.text((tx, ty), text, fill=0, font=font)
    arr = np.array(img)
    # Interior crop (leave the ring out)
    pad = 14
    return arr[pad : size - pad, pad : size - pad]


def test_pick_prefers_longer_digit_string():
    text, conf = _pick_best_digit_string([("4", 90.0), ("14", 70.0), ("1", 95.0)])
    assert text == "14"
    assert conf == 70.0


def test_ocr_two_digit_circle_14():
    crop = _render_circled_number("14")
    text, conf = _ocr_digits_in_crop(crop)
    assert text == "14", f"expected 14, got {text!r} conf={conf}"
    assert conf >= 30


def test_ocr_two_digit_circle_15():
    crop = _render_circled_number("15")
    text, conf = _ocr_digits_in_crop(crop)
    assert text == "15", f"expected 15, got {text!r} conf={conf}"
    assert conf >= 30


def test_pdf_table_rows_merge_number_and_description():
    lines = [
        ("14.", 95.0),
        ("EXHAUST FAN, DTL P/6", 95.0),
        ("15.", 95.0),
        ("VENT INTAKE, DTL Q/6", 95.0),
        ("2.", 95.0),
        ("SKYLIGHT, DTL D/4", 95.0),
    ]
    entries = parse_legend_lines(lines)
    by = {e.number: e for e in entries}
    assert 14 in by and "EXHAUST" in by[14].description
    assert 15 in by and "VENT" in by[15].description
    assert 2 in by and "SKYLIGHT" in by[2].description


def test_shadeland_pdf_has_14_and_15():
    if not SHADELAND_PDF.is_file():
        return  # skip if sample not checked out
    entries = parse_legend_file(SHADELAND_PDF)
    by = {e.number: e for e in entries}
    assert 14 in by, f"missing 14; got {sorted(by)}"
    assert 15 in by, f"missing 15; got {sorted(by)}"
    assert "EXHAUST" in by[14].description
    assert "VENT" in by[15].description
