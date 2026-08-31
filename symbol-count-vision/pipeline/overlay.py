from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def _color_for_key(key: str) -> tuple[int, int, int]:
    # Deterministic color derived from label hash.
    h = hashlib.md5((key or "").encode("utf-8")).digest()
    r = 120 + (h[0] % 120)
    g = 80 + (h[1] % 140)
    b = 60 + (h[2] % 160)
    return (r, g, b)


def short_overlay_label(symbol: str, qty: int = 1) -> str | None:
    """Compact chip for the overlay. None = draw the box with no text bar.

    Singleton ``#`` marks are the majority of a wing; labeling every one
    covers the plan. QTY>1 still gets ``#4``. Long legend keys become CAM /
    POLE / 5400 so bars stay tiny.
    """
    s = (symbol or "").strip()
    if not s:
        return None
    qty = max(1, int(qty or 1))
    if s == "#":
        return f"#{qty}" if qty > 1 else None
    u = s.upper()
    if u in {"AP", "G", "J", "WP", "WAP"}:
        return u
    if "CAMERA" in u:
        return "CAM"
    if "POLE" in u:
        return "POLE"
    if "CONDUIT STUB" in u or u == "STUB":
        return "STUB"
    m = re.search(r"(?:WM)?(2300|2400|5400|5500)\b", u)
    if m:
        return m.group(1)
    if len(s) <= 6:
        return s
    return s[:5] + "…"


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    xy: list[float],
    *,
    outline: tuple[int, int, int],
    width: int = 3,
    dash: int = 8,
) -> None:
    x1, y1, x2, y2 = xy
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in segments:
        length = max(abs(ex - sx), abs(ey - sy))
        if length <= 0:
            continue
        dx = (ex - sx) / length
        dy = (ey - sy) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            draw.line(
                [(sx + dx * pos, sy + dy * pos), (sx + dx * end, sy + dy * end)],
                fill=outline,
                width=width,
            )
            pos += dash * 2


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _overlay_font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def draw_detections_on_image(
    base: Image.Image,
    detections: Iterable[dict[str, object]],
    *,
    dashed: bool = False,
) -> Image.Image:
    draw = ImageDraw.Draw(base)
    img_w, img_h = base.size
    canvas = max(img_w, img_h)
    # Full-wing overlays are 3–5k px; 1px strokes disappear when the UI
    # shows the whole sheet. Scale so boxes stay visible zoomed out.
    if canvas >= 3600:
        box_w, font_size, text_h = 4, 16, 18.0
    elif canvas >= 2000:
        box_w, font_size, text_h = 3, 13, 15.0
    else:
        box_w, font_size, text_h = 2, 11, 12.0
    font = _overlay_font(font_size)

    parsed: list[tuple[str, int, float, float, float, float, tuple[int, int, int]]] = []
    for det in detections:
        sym = str(det.get("symbol") or "")
        box = det.get("box") or {}
        try:
            x1 = float(box.get("x1"))
            y1 = float(box.get("y1"))
            x2 = float(box.get("x2"))
            y2 = float(box.get("y2"))
        except (TypeError, ValueError):
            continue
        qty = 1
        try:
            qty = max(1, int(det.get("qty") or 1))
        except (TypeError, ValueError):
            qty = 1
        parsed.append((sym, qty, x1, y1, x2, y2, _color_for_key(sym)))

    # Pass 1: outline every mark. Detection already finished; boxes are
    # display-only and must not cover a neighbour's glyph.
    box_rects: list[tuple[float, float, float, float]] = []
    for _sym, _qty, x1, y1, x2, y2, color in parsed:
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        line_w = max(2, box_w - 1) if max(bw, bh) < 36 else box_w
        rect = [x1, y1, x2, y2]
        if dashed:
            _draw_dashed_rect(draw, rect, outline=color, width=line_w)
        else:
            draw.rectangle(rect, outline=color, width=line_w)
        box_rects.append((x1, y1, x2, y2))

    # Pass 2: compact labels, never on top of another symbol box.
    placed: list[tuple[float, float, float, float]] = []
    for i, (sym, qty, x1, y1, x2, y2, color) in enumerate(parsed):
        label = short_overlay_label(sym, qty)
        if not label:
            continue
        if min(x2 - x1, y2 - y1) < 12:
            # Sliver / empty-space hits: box only, no chip on a neighbour glyph.
            continue
        text_w = min(56.0, float(draw.textlength(label, font=font)) + 6.0)
        candidates = [
            (x1, max(0.0, y1 - text_h), x1 + text_w, max(text_h, y1)),
            (x1, min(img_h - text_h, y2), x1 + text_w, min(img_h, y2 + text_h)),
            (min(img_w - text_w, x2), max(0.0, y1 - text_h), min(img_w, x2 + text_w), y1),
        ]
        obstacles = [r for j, r in enumerate(box_rects) if j != i] + placed
        chosen: tuple[float, float, float, float] | None = None
        for cand in candidates:
            if any(_rects_overlap(cand, p) for p in obstacles):
                continue
            chosen = cand
            break
        if chosen is None:
            continue
        placed.append(chosen)
        draw.rectangle(list(chosen), fill=color)
        draw.text([chosen[0] + 2, chosen[1] + 1], label, fill=(0, 0, 0), font=font)

    return base


def draw_detections_overlay(
    image: Image.Image,
    detections: Iterable[dict[str, object]],
    output_path: Path,
    *,
    dashed: bool = False,
) -> None:
    """
    detections items expected:
      {"symbol": str, "score": float, "box": {"x1":..., "y1":..., "x2":..., "y2":...}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = draw_detections_on_image(image.convert("RGB"), detections, dashed=dashed)
    base.save(output_path, format="PNG")


__all__ = [
    "draw_detections_overlay",
    "draw_detections_on_image",
    "short_overlay_label",
]
