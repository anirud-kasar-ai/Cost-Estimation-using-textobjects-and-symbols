"""Ink profile helpers used to snap wing boundaries onto real gaps in the linework.

Wing plans on these sheets are not separated by clean whitespace (border lines and
leader lines run across the whole sheet), so instead of looking for empty gutters
we look for the *sparsest* line within a small band around a proposed boundary.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

MASK_MAX_SIDE = 1600
# A rule drawn to separate two drawings runs edge to edge, so it inks nearly
# every pixel of its row, far more than the plan linework around it. A long wall
# can cover a row too, but the rows beside it are just as busy.
RULE_COVERAGE = 0.95
RULE_STANDOUT = 2.5
# What counts as blank paper, and how wide a blank band has to be before it
# reads as the parting between two plans rather than a gap inside one.
GAP_COVERAGE = 0.03
MIN_GAP_RUN = 0.005
# How long a line has to run before it reads as drafting structure, and how
# blank the paper beside it must be to confirm it.
LINE_MIN_FRAC = 0.15
LINE_ISOLATION = 0.15


def build_mask(image: Image.Image) -> tuple[np.ndarray, float]:
    """Downscaled binary ink mask plus the scale factor from full resolution."""
    width, height = image.size
    scale = min(1.0, MASK_MAX_SIDE / max(width, height))
    small = (
        image
        if scale >= 1.0
        else image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
    )
    gray = np.asarray(small.convert("L"))
    mask = (gray < 248).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask, scale


def strip_rules(mask: np.ndarray) -> np.ndarray:
    """The mask without the long lines a sheet draws around and between drawings.

    Panel rules, borders and title underlines run for a good part of the sheet
    with blank paper on either side, and they wire every drawing on the sheet
    into one shape. Walls run long too, but they always have rooms up against
    them, so isolation is what separates structure from plan.
    """
    cleaned = mask.copy()
    height, width = mask.shape
    binary = (mask > 0).astype(np.float32)
    for axis in (0, 1):
        along = width if axis == 0 else height
        across = height if axis == 0 else width
        span = max(9, int(along * LINE_MIN_FRAC))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (span, 1) if axis == 0 else (1, span)
        )
        lines = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        if not lines.any():
            continue
        reach = max(3, int(across * 0.01))
        window = (1, 2 * reach + 1) if axis == 0 else (2 * reach + 1, 1)
        density = cv2.blur(binary, window)
        cleaned[(lines > 0) & (density <= LINE_ISOLATION)] = 0
    return cleaned


def rule_lines(mask: np.ndarray, axis: int) -> list[int]:
    """Where the sheet rules off one drawing from the next.

    axis 0 finds horizontal rules (row indices), axis 1 vertical ones.
    """
    counts = mask.sum(axis=1 if axis == 0 else 0).astype(np.float32)
    span = mask.shape[1] if axis == 0 else mask.shape[0]
    coverage = counts / max(1, span)
    length = len(coverage)
    band = max(4, int(length * 0.02))
    hits: list[int] = []
    for index in np.flatnonzero(coverage >= RULE_COVERAGE):
        index = int(index)
        before = coverage[max(0, index - band) : max(0, index - 1)]
        after = coverage[min(length, index + 2) : min(length, index + band + 1)]
        around = np.concatenate([before, after])
        if around.size and coverage[index] >= float(np.median(around)) * RULE_STANDOUT:
            hits.append(index)
    # A rule is a few pixels thick; report each once, at its centre.
    rules: list[int] = []
    for index in hits:
        if rules and index - rules[-1] <= 3:
            rules[-1] = (rules[-1] + index) // 2
        else:
            rules.append(index)
    return rules


def coverage_profile(mask: np.ndarray, axis: int, lo: int, hi: int) -> np.ndarray:
    """How much ink each line holds, measured across [lo, hi) of the other axis.

    axis 0 profiles columns (for a cut between plans side by side), axis 1
    profiles rows (for a cut between plans one above the other).
    """
    if axis == 0:
        lo, hi = max(0, min(lo, mask.shape[0])), max(0, min(hi, mask.shape[0]))
        counts = mask[lo:hi, :].sum(axis=0)
    else:
        lo, hi = max(0, min(lo, mask.shape[1])), max(0, min(hi, mask.shape[1]))
        counts = mask[:, lo:hi].sum(axis=1)
    return counts.astype(np.float32) / max(1, hi - lo)


def blank_bands(coverage: np.ndarray, start: int, stop: int) -> list[tuple[int, int]]:
    """Bands of blank paper in [start, stop), widest first.

    Two plans are parted by blank paper, while the gaps inside a plan — a
    doorway, a corridor — are narrow, so width tells the two apart.
    """
    start, stop = max(0, start), min(len(coverage), stop)
    if stop - start < 3:
        return []
    empty = np.concatenate(([False], coverage[start:stop] <= GAP_COVERAGE, [False]))
    edges = np.flatnonzero(np.diff(empty.astype(np.int8)))
    if edges.size < 2:
        return []
    floor = max(1, int((stop - start) * MIN_GAP_RUN))
    bands = [
        (start + int(a), start + int(b))
        for a, b in zip(edges[0::2], edges[1::2])
        if b - a >= floor
    ]
    return sorted(bands, key=lambda band: band[0] - band[1])


def sparsest_line(coverage: np.ndarray, start: int, stop: int) -> int | None:
    """The emptiest line in [start, stop), for sheets with no blank band at all."""
    start, stop = max(0, start), min(len(coverage), stop)
    if stop - start < 3:
        return None
    window = coverage[start:stop]
    kernel = min(9, max(3, (len(window) // 4) | 1))
    smoothed = cv2.blur(window.reshape(-1, 1), (1, kernel)).ravel()
    return start + int(np.argmin(smoothed))
