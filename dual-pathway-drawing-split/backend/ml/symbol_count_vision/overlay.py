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


def draw_detections_overlay(
    image: Image.Image,
    detections: Iterable[dict[str, object]],
    output_path: Path,
) -> None:
    """
    detections items expected:
      {"symbol": str, "score": float, "box": {"x1":..., "y1":..., "x2":..., "y2":...}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGB")
    draw = ImageDraw.Draw(base)

    try:
        # Prefer a system font if available.
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = sym
        try:
            score = float(det.get("score", 0.0))
            label = f"{sym} ({score:.2f})"
        except (TypeError, ValueError):
            pass

        # Label background.
        tx1, ty1 = x1, max(0.0, y1 - 18.0)
        tx2, ty2 = x1 + draw.textlength(label, font=font), ty1 + 18.0
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text([tx1 + 2, ty1 + 2], label, fill=(0, 0, 0), font=font)

    base.save(output_path, format="PNG")


__all__ = ["draw_detections_overlay"]

