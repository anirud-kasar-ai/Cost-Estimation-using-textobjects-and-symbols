from __future__ import annotations

import hashlib
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


def draw_detections_on_image(
    base: Image.Image,
    detections: Iterable[dict[str, object]],
    *,
    dashed: bool = False,
) -> Image.Image:
    draw = ImageDraw.Draw(base)

    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

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

        color = _color_for_key(sym)
        rect = [x1, y1, x2, y2]
        if dashed:
            _draw_dashed_rect(draw, rect, outline=color, width=3)
        else:
            draw.rectangle(rect, outline=color, width=3)

        label = sym
        try:
            score = float(det.get("score", 0.0))
            short = sym if len(sym) <= 32 else sym[:29] + "..."
            src = str(det.get("source") or "").strip()
            src_short = src.split(":")[0] if src else ""
            if src_short:
                label = f"{short} [{src_short}] ({score:.2f})"
            else:
                label = f"{short} ({score:.2f})"
        except (TypeError, ValueError):
            pass

        tx1, ty1 = x1, max(0.0, y1 - 18.0)
        max_label_w = min(220.0, max(40.0, x2 - x1))
        text_w = min(max_label_w, float(draw.textlength(label, font=font)))
        tx2, ty2 = tx1 + text_w, ty1 + 18.0
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text([tx1 + 2, ty1 + 2], label, fill=(0, 0, 0), font=font)

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


__all__ = ["draw_detections_overlay", "draw_detections_on_image"]

