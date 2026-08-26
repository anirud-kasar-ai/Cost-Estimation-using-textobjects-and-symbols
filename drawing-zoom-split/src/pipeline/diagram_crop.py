"""OpenCV drawing crop: Llama box → ink-tight plan, notes/title-block stripped."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, width: int, height: int) -> "PixelBox":
        x1 = max(0, min(width - 1, self.x1))
        y1 = max(0, min(height - 1, self.y1))
        x2 = max(x1 + 1, min(width, self.x2))
        y2 = max(y1 + 1, min(height, self.y2))
        return PixelBox(x1, y1, x2, y2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def normalized_to_pixels(
    box: list[float],
    width: int,
    height: int,
    *,
    pad: float = 0.015,
) -> PixelBox:
    x1, y1, x2, y2 = box
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(1.0, x2 + pad)
    y2 = min(1.0, y2 + pad)
    return PixelBox(
        x1=int(round(x1 * width)),
        y1=int(round(y1 * height)),
        x2=int(round(x2 * width)),
        y2=int(round(y2 * height)),
    ).clamp(width, height)


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"))


def _ink_mask(gray: np.ndarray, threshold: int = 248) -> np.ndarray:
    ink = (gray < threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)


def _content_bbox(mask: np.ndarray, pad_pct: float = 0.02) -> PixelBox | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    pad = int(max(4, min(w, h) * pad_pct))
    return PixelBox(
        x1=max(0, int(xs.min()) - pad),
        y1=max(0, int(ys.min()) - pad),
        x2=min(w, int(xs.max()) + 1 + pad),
        y2=min(h, int(ys.max()) + 1 + pad),
    )


TEXT_PANEL_DENSITY = 0.18
TEXT_PANEL_RATIO = 2.0


def _text_panel_mask(gray: np.ndarray) -> np.ndarray:
    """True where the sheet holds a block of notes/legend/key-plan text.

    Notes are packed glyphs with almost no long lines; plans are the opposite, so
    comparing local text density against line density tells them apart.
    """
    h, w = gray.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = (
        gray
        if scale >= 1.0
        else cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    )
    sh, sw = small.shape[:2]
    ink = (small < 248).astype(np.uint8)
    long_side = max(15, int(sw * 0.02))
    lines = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (long_side, 1))
    )
    lines |= cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, long_side))
    )
    text = ink.copy()
    text[lines > 0] = 0
    # Sized off the long edge so a wide, shallow strip still gets a window big
    # enough to tell a block of text from linework.
    win = max(21, int(max(sw, sh) * 0.035) | 1)
    text_density = cv2.blur(text.astype(np.float32), (win, win))
    line_density = cv2.blur(lines.astype(np.float32), (win, win))
    panel = (
        (text_density >= TEXT_PANEL_DENSITY)
        & (text_density > line_density * TEXT_PANEL_RATIO)
    ).astype(np.uint8)
    fill = max(3, (win // 2) | 1)
    panel = cv2.morphologyEx(
        panel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (fill, fill))
    )
    # Notes tables are boxed, and their frames are long lines that would survive
    # as content, so blank out each block's whole enclosing rectangle.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(panel, connectivity=8)
    grow = max(2, int(min(sw, sh) * 0.015))
    min_block = (sw * sh) * 0.004
    blocks = np.zeros_like(panel)
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if area < min_block:
            continue
        blocks[
            max(0, y - grow) : min(sh, y + bh + grow),
            max(0, x - grow) : min(sw, x + bw + grow),
        ] = 1
    panel = blocks
    if panel.shape[:2] != (h, w):
        panel = cv2.resize(panel, (w, h), interpolation=cv2.INTER_NEAREST)
    return panel > 0


def plan_mask(image: Image.Image) -> np.ndarray:
    """Ink mask with notes/legend text panels removed, leaving plan linework."""
    gray = _gray(image)
    mask = _ink_mask(gray)
    mask[_text_panel_mask(gray)] = 0
    return mask


def _column_ink_ratio(mask: np.ndarray) -> np.ndarray:
    h = max(1, mask.shape[0])
    return (mask > 0).sum(axis=0).astype(np.float64) / h


def _trim_side_strips(mask: np.ndarray, fraction: float = 0.12, factor: float = 0.42) -> np.ndarray:
    """Zero out left/right strips that are much sparser than the plan core."""
    ratios = _column_ink_ratio(mask)
    w = ratios.size
    if w < 40:
        return mask
    mid = ratios[int(w * 0.25) : int(w * 0.75)]
    if mid.size == 0:
        return mask
    median = float(np.median(mid))
    if median < 0.02:
        return mask
    band = max(8, int(w * fraction))
    out = mask.copy()
    left_mean = float(np.mean(ratios[:band]))
    right_mean = float(np.mean(ratios[-band:]))
    if left_mean < median * factor:
        out[:, :band] = 0
    if right_mean < median * factor:
        out[:, w - band :] = 0
    return out


def ink_ratio(image: Image.Image) -> float:
    """Fraction of dark pixels — used to reject blank crops."""
    gray = _gray(image)
    if gray.size == 0:
        return 0.0
    return float((gray < 248).sum()) / float(gray.size)


FRAME_LINE_RATIO = 0.88
FRAME_TOLERANCE = 0.03


def _frame_bounds(binary: np.ndarray, seed: PixelBox) -> PixelBox:
    """Limits set by the sheet's ruled frame lines.

    A title block is fenced off by a rule running nearly the full height of the
    sheet, and its own horizontal rules put ink in every column — so ink density
    alone never stops there. Only near-full-span rules count, and only outside
    the seed, so the frame caps expansion without ever cutting into the plan.
    """
    h, w = binary.shape[:2]
    col_ratio = binary.sum(axis=0) / float(max(1, h))
    row_ratio = binary.sum(axis=1) / float(max(1, w))
    margin_x = max(2, int(w * 0.004))
    margin_y = max(2, int(h * 0.004))
    slack_x = int(w * FRAME_TOLERANCE)
    slack_y = int(h * FRAME_TOLERANCE)

    rules_x = [i for i, r in enumerate(col_ratio) if r >= FRAME_LINE_RATIO]
    rules_y = [i for i, r in enumerate(row_ratio) if r >= FRAME_LINE_RATIO]
    right = [i for i in rules_x if i >= seed.x2 - slack_x]
    left = [i for i in rules_x if i <= seed.x1 + slack_x]
    bottom = [i for i in rules_y if i >= seed.y2 - slack_y]
    top = [i for i in rules_y if i <= seed.y1 + slack_y]

    return PixelBox(
        x1=(max(left) + margin_x) if left else 0,
        y1=(max(top) + margin_y) if top else 0,
        x2=(min(right) - margin_x) if right else w,
        y2=(min(bottom) - margin_y) if bottom else h,
    ).clamp(w, h)


def _frame_on_page(rgb: Image.Image, seed: PixelBox) -> PixelBox:
    """Frame bounds measured on a downscaled page, mapped back to full res."""
    width, height = rgb.size
    scale = min(1.0, 2000.0 / max(width, height))
    small = rgb if scale >= 1.0 else rgb.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )
    sw, sh = small.size
    mask = (np.asarray(small.convert("L")) < 248).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    seed_small = PixelBox(
        x1=int(seed.x1 * sw / width),
        y1=int(seed.y1 * sh / height),
        x2=int(seed.x2 * sw / width),
        y2=int(seed.y2 * sh / height),
    ).clamp(sw, sh)
    frame = _frame_bounds(mask.astype(np.int32), seed_small)
    return PixelBox(
        x1=int(frame.x1 * width / sw),
        y1=int(frame.y1 * height / sh),
        x2=int(round(frame.x2 * width / sw)),
        y2=int(round(frame.y2 * height / sh)),
    ).clamp(width, height)


def dense_blobs(mask: np.ndarray, *, min_area_frac: float = 0.01) -> list[PixelBox]:
    """Coarse blocks of linework: a plan's rooms merge into one blob, while
    separate plans, key plans and photos stay separate blobs."""
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.float32)
    scale = min(1.0, 1200.0 / max(h, w))
    if scale < 1.0:
        binary = cv2.resize(
            binary,
            (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    sh, sw = binary.shape[:2]
    win = max(9, int(max(sw, sh) * 0.02) | 1)
    dense = (cv2.blur(binary, (win, win)) >= 0.04).astype(np.uint8)
    close = max(3, int(min(sw, sh) * 0.03) | 1)
    dense = cv2.morphologyEx(
        dense, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close))
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dense, connectivity=8)
    total = float(sw * sh)
    inverse = 1.0 / scale if scale > 0 else 1.0
    blobs: list[PixelBox] = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if area / total < min_area_frac:
            continue
        blobs.append(
            PixelBox(
                x1=int(x * inverse),
                y1=int(y * inverse),
                x2=int(round((x + bw) * inverse)),
                y2=int(round((y + bh) * inverse)),
            ).clamp(w, h)
        )
    return blobs


def _rect_gap(blob: PixelBox, px: float, py: float) -> float:
    """Distance from a point to a block, zero when the point is inside it."""
    cx = min(max(px, blob.x1), blob.x2)
    cy = min(max(py, blob.y1), blob.y2)
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _rect_distance(one: PixelBox, two: PixelBox) -> float:
    dx = max(one.x1 - two.x2, two.x1 - one.x2, 0)
    dy = max(one.y1 - two.y2, two.y1 - one.y2, 0)
    return (dx**2 + dy**2) ** 0.5


def _pair_up(costs: list[list[float]], count: int) -> dict[int, int]:
    """Cheapest way to give each label a block of its own."""
    if not costs or count == 0:
        return {}
    from itertools import permutations
    from math import perm

    rows = len(costs)
    if rows <= count and perm(count, rows) <= 200_000:
        order = min(
            permutations(range(count), rows),
            key=lambda pick: sum(costs[i][j] for i, j in enumerate(pick)),
        )
        return dict(enumerate(order))
    taken: set[int] = set()
    pairs: dict[int, int] = {}
    for _cost, i, j in sorted(
        (costs[i][j], i, j) for i in range(rows) for j in range(count)
    ):
        if i in pairs or j in taken:
            continue
        pairs[i] = j
        taken.add(j)
    return pairs


FRAGMENT_REACH = 0.015


def claim_blobs(
    blobs: list[PixelBox],
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[list[int]]:
    """Which block of linework each label owns, as indices into `blobs`.

    A sheet draws one plan per title, so titles and plan blocks pair up one to
    one — which is what tells a title apart from the plan of its neighbour when
    the two are equally close. A title drawn over a block owns that block
    outright, and two titles over one block are two wings drawn against each
    other. Blocks no title claims are the sheet's other furniture: key plans,
    photos, detail insets.
    """
    claims: list[list[int]] = [[] for _ in points]
    if not blobs:
        return claims
    pixels = [(nx * width, ny * height) for nx, ny in points]
    area = [(b.x2 - b.x1) * (b.y2 - b.y1) for b in blobs]
    taken: set[int] = set()
    for index, (px, py) in enumerate(pixels):
        held = [j for j, blob in enumerate(blobs) if _rect_gap(blob, px, py) == 0]
        if held:
            # The tightest block around the title is the one it labels.
            best = min(held, key=lambda j: area[j])
            claims[index] = [best]
            taken.add(best)

    rest = [i for i, owned in enumerate(claims) if not owned]
    free = [j for j in range(len(blobs)) if j not in taken]
    if rest and free:
        costs = [[_rect_gap(blobs[j], *pixels[i]) for j in free] for i in rest]
        for row, column in _pair_up(costs, len(free)).items():
            claims[rest[row]] = [free[column]]

    # More titles than blocks: whoever is left shares the block nearest them.
    for index, (px, py) in enumerate(pixels):
        if not claims[index]:
            claims[index] = [
                min(range(len(blobs)), key=lambda j: _rect_gap(blobs[j], px, py))
            ]

    # Leftover blocks belong to the nearest title: a large plan sitting just
    # past a title bar, or a small caption broken off the plan it names.
    owned = {j for group in claims for j in group}
    diagonal = max(1.0, (width**2 + height**2) ** 0.5)
    total = float(max(1, width * height))
    small_reach = diagonal * FRAGMENT_REACH
    large_reach = diagonal * 0.18
    for j, blob in enumerate(blobs):
        if j in owned or not points:
            continue
        nearest = min(range(len(points)), key=lambda i: _rect_gap(blob, *pixels[i]))
        gap = _rect_gap(blob, *pixels[nearest])
        frac = area[j] / total
        if (frac >= 0.02 and gap <= large_reach) or (
            frac < 0.02 and gap <= small_reach
        ):
            claims[nearest].append(j)
    return claims


def plan_side(
    blobs: list[PixelBox],
    points: list[tuple[float, float]],
    claims: list[list[int]],
    height: int,
) -> int:
    """Which side of its title a plan sits on: +1 below, -1 above, 0 unknown.

    A sheet keeps one habit for every plan on it, so the titles that clearly
    stand outside their block vote for the rest.
    """
    votes = 0
    for (_nx, ny), owned in zip(points, claims):
        if not owned:
            continue
        blob = blobs[owned[0]]
        py = ny * height
        if blob.y1 <= py <= blob.y2:
            continue
        votes += 1 if blob.y1 > py else -1
    return (votes > 0) - (votes < 0)


@dataclass(frozen=True)
class PlanLayout:
    """How the plans on a sheet sit against their titles."""

    blobs: list[PixelBox]
    claims: list[list[int]]
    side: int

    def owned(self, index: int) -> list[PixelBox]:
        return [self.blobs[j] for j in self.claims[index]]


def plan_layout(mask: np.ndarray, points: list[tuple[float, float]]) -> PlanLayout:
    """Read the blocks of linework on a sheet and hand each one to its title."""
    h, w = mask.shape[:2]
    blobs = dense_blobs(mask)
    claims = claim_blobs(blobs, points, w, h)
    return PlanLayout(blobs, claims, plan_side(blobs, points, claims, h))


LABEL_REACH = 0.3


def _labelled_plan_box(
    mask: np.ndarray,
    label_points: list[tuple[float, float]],
) -> PixelBox | None:
    """Union of the plan blocks that belong to a wing label.

    Sheets also carry key plans, photos and detail insets. Keeping only blocks
    at a wing label leaves the plans and drops the rest.
    """
    if not label_points:
        return None
    h, w = mask.shape[:2]
    layout = plan_layout(mask, label_points)
    if not layout.blobs:
        return None
    reach = ((w**2 + h**2) ** 0.5) * LABEL_REACH
    chosen: list[PixelBox] = []
    for index, (nx, ny) in enumerate(label_points):
        owned = layout.owned(index)
        px, py = nx * w, ny * h
        # A label with no plan anywhere near it is not a plan title at all.
        if not owned or _rect_gap(owned[0], px, py) > reach:
            continue
        chosen.extend(owned)
        chosen.append(PixelBox(int(px), int(py), int(px) + 1, int(py) + 1))
    if not chosen:
        return None
    pad = int(max(4, min(w, h) * 0.01))
    return PixelBox(
        x1=max(0, min(b.x1 for b in chosen) - pad),
        y1=max(0, min(b.y1 for b in chosen) - pad),
        x2=min(w, max(b.x2 for b in chosen) + pad),
        y2=min(h, max(b.y2 for b in chosen) + pad),
    )


def _plan_area_box(mask: np.ndarray, pad_pct: float = 0.01) -> PixelBox | None:
    """Bounding box of substantial plan content, ignoring stray marks.

    A single stray line at the paper edge would defeat an extreme-value bbox, so
    a row or column only counts once it carries a real share of the ink.
    """
    h, w = mask.shape[:2]
    rows = (mask > 0).sum(axis=1)
    cols = (mask > 0).sum(axis=0)
    if rows.max() == 0:
        return None
    ys = np.nonzero(rows >= max(1.0, rows.max() * 0.02))[0]
    xs = np.nonzero(cols >= max(1.0, cols.max() * 0.02))[0]
    if ys.size == 0 or xs.size == 0:
        return None
    pad = int(max(4, min(w, h) * pad_pct))
    return PixelBox(
        x1=max(0, int(xs.min()) - pad),
        y1=max(0, int(ys.min()) - pad),
        x2=min(w, int(xs.max()) + 1 + pad),
        y2=min(h, int(ys.max()) + 1 + pad),
    )


def crop_drawing(
    page: Image.Image,
    llama_box: list[float] | None,
    label_points: list[tuple[float, float]] | None = None,
) -> tuple[Image.Image, PixelBox]:
    """Crop the plan drawing area from a full sheet.

    The vision model decides whether a sheet holds plans; the extent comes from
    CV, because the model's box routinely misses a plan at the edge of the sheet.
    Inside the ruled frame, the plans sitting at the wing labels define the area,
    which leaves out key plans, photos and detail insets.
    """
    rgb = page.convert("RGB")
    width, height = rgb.size
    seed = (
        normalized_to_pixels(llama_box, width, height, pad=0.02)
        if llama_box
        else PixelBox(0, 0, width, height)
    )
    frame = _frame_on_page(rgb, seed)

    cropped = rgb.crop(frame.to_tuple())
    mask = _trim_side_strips(plan_mask(cropped))
    local_points = [
        (
            (nx * width - frame.x1) / max(1.0, float(frame.x2 - frame.x1)),
            (ny * height - frame.y1) / max(1.0, float(frame.y2 - frame.y1)),
        )
        for nx, ny in (label_points or [])
    ]
    inner = _labelled_plan_box(
        mask, [p for p in local_points if 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0]
    ) or _plan_area_box(mask)
    if inner is None:
        return rgb.crop(seed.to_tuple()), seed
    seed = frame

    local = cropped.crop(inner.to_tuple())
    abs_box = PixelBox(
        x1=seed.x1 + inner.x1,
        y1=seed.y1 + inner.y1,
        x2=seed.x1 + inner.x2,
        y2=seed.y1 + inner.y2,
    ).clamp(width, height)
    # Reject a collapse that dropped most of the drawing
    seed_area = max(1, (seed.x2 - seed.x1) * (seed.y2 - seed.y1))
    kept = (abs_box.x2 - abs_box.x1) * (abs_box.y2 - abs_box.y1)
    if kept / seed_area < 0.18:
        return cropped, seed
    return local.convert("RGB"), abs_box
