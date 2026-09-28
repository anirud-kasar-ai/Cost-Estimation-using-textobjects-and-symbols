from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from pipeline.callout_ocr import CalloutHit
from pipeline.yolo_detect import YoloHit


def _color_for_key(key: str) -> tuple[int, int, int]:
    """Deterministic, saturated, visually distinct color per class name.

    Same golden-angle hue spread the i-TAB UI uses, so a class keeps one
    recognizable color everywhere it is drawn.
    """
    s = (key or "").strip().upper()
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    hue = (h * 137.508) % 360.0
    sat = (68 + (h % 3) * 8) / 100.0
    light = (36 + ((h >> 3) % 3) * 7) / 100.0
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, sat)
    return (int(r * 255), int(g * 255), int(b * 255))


def _font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label_chip(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    color: tuple[int, int, int],
    font,
) -> None:
    tx, ty = x, max(0, y - 16)
    try:
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    except Exception:  # noqa: BLE001
        draw.text((tx, ty), text, fill=color, font=font)


def draw_detection_overlay(
    image: Image.Image,
    *,
    callouts: Iterable[CalloutHit] = (),
    yolo_hits: Iterable[YoloHit] = (),
    out_path: Path | None = None,
) -> Image.Image:
    """Draw YOLO boxes (solid) and OCR callout circles (dashed-style thin boxes)."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    font_sm = _font(11)

    rescued_color = (255, 140, 0)  # orange: added by the low-conf verification pass
    for hit in yolo_hits:
        rescued = bool(getattr(hit, "rescued", False))
        color = rescued_color if rescued else _color_for_key(hit.class_name)
        xy = [hit.x1, hit.y1, hit.x2, hit.y2]
        draw.rectangle(xy, outline=color, width=3 if rescued else 2)
        label = hit.class_name
        if hit.legend_number is not None:
            label = f"#{hit.legend_number} {label}"
        label = f"{label} {hit.confidence:.2f}"
        if rescued:
            label = f"+ {label}"
        _label_chip(draw, hit.x1, hit.y1, label[:42], color, font_sm)

    for hit in callouts:
        color = _color_for_key(str(hit.number))
        # Slightly thicker ring cue for callouts
        xy = [hit.x1, hit.y1, hit.x2, hit.y2]
        draw.ellipse(xy, outline=color, width=2)
        _label_chip(draw, hit.x1, hit.y1, f"○{hit.number}", color, font)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
    return canvas


# Back-compat alias
def draw_callout_overlay(
    image: Image.Image,
    hits: list[CalloutHit],
    *,
    out_path: Path | None = None,
) -> Image.Image:
    return draw_detection_overlay(image, callouts=hits, yolo_hits=(), out_path=out_path)
