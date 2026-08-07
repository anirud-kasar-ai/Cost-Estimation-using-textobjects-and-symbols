"""Generate a synthetic HVAC layout PDF for local end-to-end testing.

The page mimics a real drawing sheet: a large drawing area on the left with
rooms and simple geometric device symbols, and a title block strip on the
right with project metadata fields. Usage:

    python scripts/generate_sample_pdf.py [output.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = landscape(A3)  # 1190 x 842 pt

TITLE_BLOCK_FIELDS: list[tuple[str, str]] = [
    ("PROJECT TITLE", "Riverside Office Tower - Level 3 HVAC Layout"),
    ("CLIENT", "Meridian Property Group"),
    ("ARCHITECT", "Atelier North Architects"),
    ("CONSULTANT", "Vector Building Services Ltd."),
    ("PROJECT ADDRESS", "128 Riverside Drive, Springfield"),
    ("ISSUE DATE", "2026-08-15"),
    ("DRAWING NO", "M-301"),
    ("SCALE", "1:100"),
]

PROVIDER_BLOCK: list[str] = [
    "THE GARLAND COMPANY INC",
    "3800 EAST 91st STREET - CLEVELAND, OHIO 44105",
    "PHONE (800) 321-9336 / FAX (216) 641-0633",
    "AGENT: DOUG CLARK",
]

SCOPE_OF_WORK: list[str] = [
    "SCOPE OF WORK:",
    "1. REMOVE AND DISPOSE OF ALL DAMAGED FLASHINGS PER MANUFACTURER REQUIREMENTS.",
    "2. INSTALL NEW SUPPLY AND RETURN DIFFUSERS IN OPEN OFFICE AREAS.",
    "3. INSTALL CO2 AND TEMPERATURE SENSORS AT LOCATIONS SHOWN ON PLAN.",
    "4. TEST AND BALANCE ALL NEW HVAC DEVICES PRIOR TO HANDOVER.",
    "5. CLEAN WORK AREA AT END OF EACH WORK DAY.",
]


def _draw_room(c: canvas.Canvas, x: float, y: float, w: float, h: float, name: str) -> None:
    c.rect(x, y, w, h)
    c.setFont("Helvetica", 9)
    c.drawString(x + 6, y + h - 14, name)


def _supply_diffuser(c: canvas.Canvas, x: float, y: float, s: float = 18) -> None:
    """Square with an X — common supply air diffuser symbol."""
    c.rect(x, y, s, s)
    c.line(x, y, x + s, y + s)
    c.line(x, y + s, x + s, y)


def _return_grille(c: canvas.Canvas, x: float, y: float, s: float = 18) -> None:
    """Square with a single diagonal."""
    c.rect(x, y, s, s)
    c.line(x, y, x + s, y + s)


def _exhaust_grille(c: canvas.Canvas, x: float, y: float, s: float = 18) -> None:
    """Square with horizontal hatching."""
    c.rect(x, y, s, s)
    for i in range(1, 4):
        c.line(x, y + i * s / 4, x + s, y + i * s / 4)


def _sensor(c: canvas.Canvas, x: float, y: float, letter: str, r: float = 9) -> None:
    """Circle with a letter — sensor/thermostat symbols."""
    c.circle(x + r, y + r, r)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + r, y + r - 3, letter)


def _draw_drawing_area(c: canvas.Canvas) -> None:
    area_w = PAGE_W * 0.72
    c.rect(20, 20, area_w - 40, PAGE_H - 40)

    _draw_room(c, 60, 480, 300, 280, "OPEN OFFICE 3.01")
    _draw_room(c, 400, 480, 200, 280, "MEETING 3.02")
    _draw_room(c, 60, 120, 260, 300, "OPEN OFFICE 3.03")
    _draw_room(c, 360, 120, 240, 300, "BREAK ROOM 3.04")
    _draw_room(c, 640, 120, 160, 640, "CORRIDOR 3.05")

    # Supply diffusers (6)
    for x, y in [(110, 620), (240, 620), (110, 520), (240, 520), (450, 620), (450, 540)]:
        _supply_diffuser(c, x, y)
    # Return grilles (4)
    for x, y in [(170, 570), (500, 580), (150, 250), (450, 250)]:
        _return_grille(c, x, y)
    # Exhaust grilles (2)
    for x, y in [(420, 160), (700, 160)]:
        _exhaust_grille(c, x, y)
    # Temperature sensors (3)
    for x, y in [(90, 300), (390, 300), (670, 400)]:
        _sensor(c, x, y, "T")
    # CO2 sensors (2)
    for x, y in [(200, 300), (520, 300)]:
        _sensor(c, x, y, "C")
    # Thermostats (2)
    for x, y in [(320, 700), (560, 700)]:
        _sensor(c, x, y, "TH")

    c.setFont("Helvetica", 8)
    c.drawString(30, 28, "LEGEND:  [X] SUPPLY DIFFUSER   [\\] RETURN GRILLE   [=] EXHAUST GRILLE   (T) TEMP SENSOR   (C) CO2 SENSOR   (TH) THERMOSTAT")

    # Scope of work block (exercises requirement extraction in E2E tests)
    c.setFont("Helvetica-Bold", 9)
    y = 100
    for line in SCOPE_OF_WORK:
        c.setFont("Helvetica-Bold" if line.startswith("SCOPE") else "Helvetica", 8)
        c.drawString(60, y, line[:110])
        y -= 12


def _draw_title_block(c: canvas.Canvas) -> None:
    block_x = PAGE_W * 0.74
    block_w = PAGE_W * 0.24
    block_y = 20
    block_h = PAGE_H - 40
    c.rect(block_x, block_y, block_w, block_h)

    # Provider / agent block at the top of the title strip
    c.setFont("Helvetica-Bold", 8)
    provider_y = block_y + block_h - 14
    for line in PROVIDER_BLOCK:
        c.drawString(block_x + 8, provider_y, line[:42])
        provider_y -= 11

    row_h = 48.0
    y = provider_y - 20
    for label, value in TITLE_BLOCK_FIELDS:
        c.line(block_x, y, block_x + block_w, y)
        c.setFont("Helvetica", 7)
        c.drawString(block_x + 8, y + row_h - 14, label)
        c.setFont("Helvetica-Bold", 8)
        value_lines = [value] if len(value) <= 40 else [value[:40], value[40:]]
        for i, line in enumerate(value_lines):
            c.drawString(block_x + 8, y + row_h - 28 - i * 10, line)
        y -= row_h


def generate_sample_pdf(output_path: Path, pages: int = 1) -> Path:
    """Write the synthetic layout PDF to ``output_path`` and return it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=(PAGE_W, PAGE_H))
    for _ in range(pages):
        _draw_drawing_area(c)
        _draw_title_block(c)
        c.showPage()
    c.save()
    return output_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("storage/sample_layout.pdf")
    result = generate_sample_pdf(target)
    print(f"Sample HVAC layout written to {result.resolve()}")
