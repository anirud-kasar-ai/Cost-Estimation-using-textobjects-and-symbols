"""Disambiguate legend look-alikes: J-HOOK vs J-box, CONDUIT STUB vs letter E."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.cv.tag_match import count_key
from pipeline.extraction.symbol_table_extractor import SymbolTableInfo

_JHOOK_RE = re.compile(r"J-?HOOK", re.I)
_STUB_RE = re.compile(r"CONDUIT\s+STUB", re.I)


def j_hook_legend_key(legend: SymbolTableInfo) -> str | None:
    for entry in legend.entries:
        key = count_key(entry)
        if not key:
            continue
        blob = f"{key} {entry.description or ''}"
        if _JHOOK_RE.search(blob):
            return key
    return None


def conduit_stub_legend_key(legend: SymbolTableInfo) -> str | None:
    for entry in legend.entries:
        key = count_key(entry)
        if not key:
            continue
        blob = f"{key} {entry.description or ''}"
        if _STUB_RE.search(blob):
            return key
    return None


def _center(det: SymbolDetection) -> tuple[float, float]:
    return ((det.box.x1 + det.box.x2) / 2.0, (det.box.y1 + det.box.y2) / 2.0)


def _x_gap(a: SymbolDetection, b: SymbolDetection) -> float:
    if a.box.x2 < b.box.x1:
        return b.box.x1 - a.box.x2
    if b.box.x2 < a.box.x1:
        return a.box.x1 - b.box.x2
    return 0.0


def reclassify_j_hook_clusters(
    detections: list[SymbolDetection],
    legend: SymbolTableInfo,
) -> list[SymbolDetection]:
    """2+ collinear J tags on one line → one J-HOOK; leftover isolated J stays J-box."""
    hook_key = j_hook_legend_key(legend)
    if not hook_key:
        return detections
    j_hits = [d for d in detections if (d.symbol or "").strip().upper() == "J"]
    others = [d for d in detections if (d.symbol or "").strip().upper() != "J"]
    if len(j_hits) < 2:
        return detections

    ordered = sorted(j_hits, key=lambda d: (_center(d)[1], _center(d)[0]))
    used: set[int] = set()
    out = list(others)
    for i, anchor in enumerate(ordered):
        if i in used:
            continue
        _acy = _center(anchor)[1]
        ah = max(4.0, anchor.box.y2 - anchor.box.y1)
        cluster = [anchor]
        used.add(i)
        grew = True
        while grew:
            grew = False
            for j, cand in enumerate(ordered):
                if j in used:
                    continue
                cy = _center(cand)[1]
                if abs(cy - _acy) > max(14.0, ah * 1.25):
                    continue
                max_gap = max(36.0, max(d.box.x2 - d.box.x1 for d in cluster) * 3.0)
                if any(_x_gap(m, cand) <= max_gap for m in cluster):
                    cluster.append(cand)
                    used.add(j)
                    grew = True
        if len(cluster) >= 2:
            x1 = min(d.box.x1 for d in cluster)
            y1 = min(d.box.y1 for d in cluster)
            x2 = max(d.box.x2 for d in cluster)
            y2 = max(d.box.y2 for d in cluster)
            score = max(float(d.score) for d in cluster)
            out.append(
                SymbolDetection(
                    symbol=hook_key,
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=score,
                    source="jhook",
                    qty=1,
                )
            )
        else:
            out.append(anchor)
    return out


def detect_conduit_stubs(
    image: Image.Image,
    *,
    exclude_top_pct: float = 0.12,
) -> list[dict[str, Any]]:
    """E-like trident whose middle bar is much longer than the others (CONDUIT STUB)."""
    import cv2

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    y_min = int(h * max(0.0, exclude_top_pct))
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    if y_min > 0:
        binary[:y_min, :] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stubs: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40 or area > 6000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 10 or bh < 10 or y + bh / 2.0 < y_min:
            continue
        # Stub is wider than tall (long middle bar) or roughly square-E.
        if bw / float(bh) < 0.7:
            continue
        if bh / float(bw) > 1.8:
            continue
        fill = area / float(bw * bh)
        if fill < 0.12 or fill > 0.55:
            continue
        patch = binary[y : y + bh, x : x + bw]
        if patch.size == 0:
            continue
        # Three horizontal bands: top, mid, bottom. Mid must be longer (more ink
        # on the right half) than top/bottom.
        band = max(1, bh // 5)
        top = patch[:band, :]
        mid = patch[2 * band : 3 * band, :] if 3 * band <= bh else patch[bh // 3 : 2 * bh // 3, :]
        bot = patch[bh - band :, :]
        if top.size == 0 or mid.size == 0 or bot.size == 0:
            continue
        right = patch[:, bw // 2 :]
        if right.size == 0:
            continue
        # Right-half ink concentrated in the middle band.
        mid_right = mid[:, bw // 2 :] if mid.shape[1] > 1 else mid
        top_right = top[:, bw // 2 :] if top.shape[1] > 1 else top
        bot_right = bot[:, bw // 2 :] if bot.shape[1] > 1 else bot
        mid_r = float(np.mean(mid_right > 0))
        top_r = float(np.mean(top_right > 0))
        bot_r = float(np.mean(bot_right > 0))
        if mid_r < 0.12:
            continue
        if mid_r < 1.45 * max(top_r, 0.04) or mid_r < 1.45 * max(bot_r, 0.04):
            continue
        stubs.append(
            {
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + bw),
                "y2": float(y + bh),
                "cx": x + bw / 2.0,
                "cy": y + bh / 2.0,
            }
        )
    return stubs
