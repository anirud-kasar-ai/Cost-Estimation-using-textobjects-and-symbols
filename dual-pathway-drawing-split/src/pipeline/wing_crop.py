"""Cut one image per wing.

CV owns the boundaries; the vision model and the PDF text layer only say WHICH
wing is where, because their coordinates are far too coarse to crop with.

Each title claims the block of linework it belongs to, and that block is the
crop. Only where several titles claim one block — wings that share walls — is
the block divided, guided by any rule drawn between them, by the room tags of
each area, and finally by the blank paper between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from PIL import Image

from pipeline.diagram_crop import (
    PixelBox,
    PlanLayout,
    _content_bbox,
    ink_ratio,
    plan_layout,
    plan_mask,
)
from pipeline.ink_profile import (
    blank_bands,
    build_mask,
    coverage_profile,
    rule_lines,
    sparsest_line,
    strip_rules,
)

MIN_WING_INK_RATIO = 0.005
MIN_WING_SIDE_PX = 80
# Some valid plan bars are long and thin (e.g. D-1 / corridors). Keep a
# generous aspect bound and rely on plan-content ratio to reject true slivers.
MAX_WING_ASPECT = 50.0
MIN_WING_PLAN_RATIO = 0.05
LABEL_CLEARANCE = 0.06
TITLE_CLEARANCE = 0.004
RULE_LABEL_CLEARANCE = 0.025
EDGE_BAND = 0.12
FACING_BAND = 0.2
FACING_OFFSET = 0.12
SIDE_PREFERENCE = 2.0
MIN_SIDE_INK = 0.12
MIN_TAGS_PER_WING = 3
TAG_INSIDE_RATIO = 0.6
TAG_SYSTEM_RATIO = 0.5


@dataclass(frozen=True)
class WingRegion:
    name: str
    slug: str
    box: PixelBox
    image: Image.Image
    instance_id: str = ""
    vision_box: list[float] | None = None
    source: str = "vision_model"


def slugify(name: str) -> str:
    """Folder name for a wing, spelled as the sheet spells it."""
    cleaned = re.sub(r"[^A-Z0-9]+", "-", name.strip().upper())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:80] or "FULL-PLAN"


def assign_wing_instance_ids(items: list[dict]) -> list[dict]:
    """Assign unique instance_id values; duplicate display names are kept."""
    counts: dict[str, int] = {}
    out: list[dict] = []
    for item in items:
        name = slugify(str(item.get("name") or ""))
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
        idx = counts[name]
        instance_id = name if idx == 1 else f"{name}-{idx}"
        out.append({**item, "name": name, "instance_id": instance_id})
    return out


def _plan_content_ratio(image: Image.Image) -> float:
    """How much of the crop is actual plan linework rather than text."""
    w, h = image.size
    scale = min(1.0, 1000.0 / max(w, h))
    small = (
        image
        if scale >= 1.0
        else image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    )
    return float((plan_mask(small) > 0).mean())


def _is_usable_wing(image: Image.Image) -> bool:
    """Drop blanks, slivers, and label strips that hold no plan of their own.

    Some sheets carry a wing title whose plan is drawn on a different sheet; the
    crop then contains only text, which the plan-content test rejects.
    """
    w, h = image.size
    if w < MIN_WING_SIDE_PX or h < MIN_WING_SIDE_PX:
        return False
    if max(w, h) / max(1, min(w, h)) > MAX_WING_ASPECT:
        return False
    if ink_ratio(image) < MIN_WING_INK_RATIO:
        return False
    return _plan_content_ratio(image) >= MIN_WING_PLAN_RATIO


def _trim_to_ink(image: Image.Image) -> tuple[Image.Image, PixelBox]:
    """Tighten onto the plan linework, dropping notes panels and blank paper."""
    inner = _content_bbox(plan_mask(image), pad_pct=0.015)
    w, h = image.size
    if inner is None:
        return image.convert("RGB"), PixelBox(0, 0, w, h)
    area = (inner.x2 - inner.x1) * (inner.y2 - inner.y1)
    if area / max(1, w * h) < 0.04:
        return image.convert("RGB"), PixelBox(0, 0, w, h)
    return image.crop(inner.to_tuple()).convert("RGB"), inner


@dataclass(frozen=True)
class WingAnchor:
    """Where a wing is, and how we know: a point, not a boundary."""

    name: str
    x: float
    y: float
    source: str
    instance_id: str
    vision_box: list[float] | None = None


def anchors_from_specs(
    wing_specs: list[dict],
    label_points: list[dict] | None = None,
) -> list[WingAnchor]:
    """Build one anchor per wing instance. PDF text labels win over model boxes."""
    specs = assign_wing_instance_ids(wing_specs or [])
    anchors: dict[str, WingAnchor] = {}

    for item in label_points or []:
        name = slugify(str(item.get("name") or ""))
        point = item.get("point")
        if not name or not point:
            continue
        # Match label point to a spec instance by name + proximity when possible.
        instance_id = str(item.get("instance_id") or "")
        if not instance_id:
            px, py = float(point[0]), float(point[1])
            for spec in specs:
                if spec["name"] != name:
                    continue
                box = spec.get("box")
                if box and box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                    instance_id = spec["instance_id"]
                    break
            if not instance_id:
                for spec in specs:
                    if spec["name"] == name and spec["instance_id"] not in anchors:
                        instance_id = spec["instance_id"]
                        break
            if not instance_id:
                instance_id = name
        if instance_id in anchors:
            continue
        anchors[instance_id] = WingAnchor(
            name=name,
            instance_id=instance_id,
            x=float(point[0]),
            y=float(point[1]),
            source="pdf_text",
        )

    for item in specs:
        name = slugify(str(item.get("name") or "FULL-PLAN"))
        instance_id = str(item.get("instance_id") or name)
        box = item.get("box")
        if not box:
            if instance_id not in anchors:
                point = item.get("point")
                if point:
                    anchors[instance_id] = WingAnchor(
                        name=name,
                        instance_id=instance_id,
                        x=float(point[0]),
                        y=float(point[1]),
                        source=str(item.get("source") or "pdf_text"),
                    )
            continue
        norm = [float(v) for v in box]
        covered = next(
            (
                key
                for key, anchor in anchors.items()
                if anchor.source == "pdf_text"
                and norm[0] <= anchor.x <= norm[2]
                and norm[1] <= anchor.y <= norm[3]
            ),
            None,
        )
        if covered is not None:
            anchor = anchors[covered]
            anchors[covered] = WingAnchor(
                name=anchor.name,
                instance_id=anchor.instance_id,
                x=anchor.x,
                y=anchor.y,
                source=anchor.source,
                vision_box=norm,
            )
            continue
        if instance_id in anchors:
            anchor = anchors[instance_id]
            anchors[instance_id] = WingAnchor(
                name=anchor.name,
                instance_id=anchor.instance_id,
                x=anchor.x,
                y=anchor.y,
                source=anchor.source,
                vision_box=norm,
            )
            continue
        anchors[instance_id] = WingAnchor(
            name=name,
            instance_id=instance_id,
            x=(norm[0] + norm[2]) / 2.0,
            y=(norm[1] + norm[3]) / 2.0,
            source=str(item.get("source") or "vision_model"),
            vision_box=norm,
        )
    return list(anchors.values())


def _focus_on_wing(
    cell: Image.Image,
    anchor_x: float,
    anchor_y: float,
    side: int = 0,
) -> PixelBox | None:
    """Tighten a cell onto the plan body belonging to its label.

    A cell can still hold leftovers of the sheet — a neighbour's rooms, a tag
    table, a key plan, a scale bar. The plan itself is the substantial block of
    linework by the label, on the side of it this sheet puts its plans.
    """
    import cv2
    import numpy as np

    width, height = cell.size
    scale = min(1.0, 1200.0 / max(width, height))
    small = (
        cell
        if scale >= 1.0
        else cell.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
    )
    mask = (strip_rules(plan_mask(small)) > 0).astype(np.float32)
    sh, sw = mask.shape[:2]
    if sw < 40 or sh < 40:
        return None
    win = max(9, int(min(sw, sh) * 0.03) | 1)
    dense = (cv2.blur(mask, (win, win)) >= 0.04).astype(np.uint8)
    close = max(3, int(min(sw, sh) * 0.03) | 1)
    dense = cv2.morphologyEx(
        dense, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close))
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dense, connectivity=8)
    total = float(sw * sh)
    diagonal = max(1.0, (sw**2 + sh**2) ** 0.5)
    ax, ay = anchor_x * sw, anchor_y * sh

    best: tuple[float, PixelBox] | None = None
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if area / total < 0.01:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        distance = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 / diagonal
        score = (area / total) / (1.0 + 3.0 * distance)
        # Two plans can sit equally close to a title; the sheet's own habit of
        # putting plans above or below their titles breaks the tie.
        if side and (cy > ay) == (side > 0):
            score *= SIDE_PREFERENCE
        if best is None or score > best[0]:
            best = (score, PixelBox(x, y, x + bw, y + bh))
    if best is None:
        return None

    box = best[1]
    if (box.y2 - box.y1) / max(1, sh) < 0.12 and side >= 0:
        return None
    grow = max(4, int(min(sw, sh) * 0.02))
    label_pad = max(4, int(min(sw, sh) * 0.03))
    box = PixelBox(
        x1=min(box.x1 - grow, int(ax) - label_pad),
        y1=min(box.y1 - grow, int(ay) - label_pad),
        x2=max(box.x2 + grow, int(ax) + label_pad),
        y2=max(box.y2 + grow, int(ay) + label_pad),
    ).clamp(sw, sh)
    inverse = 1.0 / scale if scale > 0 else 1.0
    return PixelBox(
        x1=int(box.x1 * inverse),
        y1=int(box.y1 * inverse),
        x2=int(round(box.x2 * inverse)),
        y2=int(round(box.y2 * inverse)),
    ).clamp(width, height)


def _room_pad(points: list[tuple[float, float]], width: int, height: int) -> float:
    """Roughly one room, measured as the typical spacing between room tags."""
    import numpy as np

    coords = np.array([[p[0] * width, p[1] * height] for p in points], dtype=float)
    gaps = []
    for index in range(len(coords)):
        deltas = coords - coords[index]
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        distances[index] = np.inf
        gaps.append(float(distances.min()))
    spacing = float(np.median(gaps)) if gaps else 0.0
    return max(min(width, height) * 0.02, spacing * 1.2)


def _wing_prefixes(
    anchors: list[WingAnchor],
    tag_points: list[dict] | None,
) -> dict[str, str]:
    """Match each wing to the room-tag prefix its rooms carry.

    Room numbers name their area ("B-13" in B WING, "AD-17" in ADMIN. BLDG.), so
    they mark where one wing ends even where the two share a wall. The prefix
    usually leads the wing name; when it does not, the tags crowding around the
    label decide. A prefix claimed by two wings says nothing about the boundary
    between them, so it is dropped.
    """
    from collections import Counter

    if not tag_points:
        return {}
    known = {str(tag.get("name") or "") for tag in tag_points}
    nearby: dict[str, Counter] = {anchor.instance_id: Counter() for anchor in anchors}
    for tag in tag_points:
        point = tag.get("point") or [0.5, 0.5]
        tx, ty = float(point[0]), float(point[1])
        nearest = min(anchors, key=lambda a: (a.x - tx) ** 2 + (a.y - ty) ** 2)
        nearby[nearest.instance_id][str(tag.get("name") or "")] += 1

    matched: dict[str, str] = {}
    for anchor in anchors:
        head = anchor.name.split("-")[0]
        if head in known:
            matched[anchor.instance_id] = head
            continue
        common = nearby[anchor.instance_id].most_common(1)
        if common and common[0][1] >= MIN_TAGS_PER_WING:
            matched[anchor.instance_id] = common[0][0]
    counts = Counter(matched.values())
    return {iid: prefix for iid, prefix in matched.items() if counts[prefix] == 1}


def _tags_name_areas(
    anchors: list[WingAnchor],
    tag_points: list[dict] | None,
) -> bool:
    """Do this sheet's room numbers say which area a room is in?

    Some sets number rooms by area, so B-13 is a room in the B wing and the tags
    map out where each wing reaches. Others number by room type or use the same
    style for detail markers, where a prefix says nothing about location. The
    tags themselves tell which it is: area numbering agrees with the titles.
    """
    if not tag_points or not anchors:
        return False
    agree = 0
    for tag in tag_points:
        point = tag.get("point") or [0.5, 0.5]
        tx, ty = float(point[0]), float(point[1])
        nearest = min(anchors, key=lambda a: (a.x - tx) ** 2 + (a.y - ty) ** 2)
        prefix = str(tag.get("name") or "")
        if prefix and any(
            word.startswith(prefix) for word in nearest.name.split("-")
        ):
            agree += 1
    return agree / len(tag_points) >= TAG_SYSTEM_RATIO


def _tag_window(
    axis: int,
    extent: int,
    anchor: WingAnchor,
    other: WingAnchor,
    prefixes: dict[str, str],
    tags: dict[str, list[tuple[float, float]]],
    default: tuple[int, int],
) -> tuple[int, int]:
    """Narrow the search for a boundary to the space between two tag clusters."""
    import numpy as np

    mine, theirs = prefixes.get(anchor.instance_id), prefixes.get(other.instance_id)
    if not mine or not theirs or mine == theirs:
        return default
    ours = [point[axis] for point in tags.get(mine, ())]
    yours = [point[axis] for point in tags.get(theirs, ())]
    if len(ours) < MIN_TAGS_PER_WING or len(yours) < MIN_TAGS_PER_WING:
        return default
    here = anchor.x if axis == 0 else anchor.y
    there = other.x if axis == 0 else other.y
    if there < here:
        low, high = np.percentile(yours, 90), np.percentile(ours, 10)
    else:
        low, high = np.percentile(ours, 90), np.percentile(yours, 10)
    lo, hi = int(low * extent), int(high * extent)
    if hi - lo < 4:
        return default
    return lo, hi


def _plan_gutter_window(
    axis: int,
    anchor: WingAnchor,
    other: WingAnchor,
    extent: int,
) -> tuple[int, int]:
    """Paper between two stacked plans, not between their titles.

    A title sits at one end of its plan. The cut belongs in the blank band
    between the two plan bodies, which starts below the upper title and runs
    into the lower wing's drawing.
    """
    here = int((anchor.x if axis == 0 else anchor.y) * extent)
    there = int((other.x if axis == 0 else other.y) * extent)
    near = max(2, int(extent * TITLE_CLEARANCE))
    span = sorted((here, there))
    lo = span[0] + near
    hi = span[1] - near
    if hi - lo < 4:
        return lo, hi
    past_title = int(extent * 0.10)
    into_plan = int(extent * 0.40)
    if axis == 1:
        lo = max(lo, span[0] + past_title)
        hi = max(lo + 4, min(hi, span[1] + into_plan))
    else:
        lo = max(lo, span[0] + past_title)
        hi = max(lo + 4, min(hi, span[1] + into_plan))
    return lo, hi


def _cut_between(
    axis: int,
    anchor: WingAnchor,
    other: WingAnchor,
    mask,
    rules: list[int],
    prefixes: dict[str, str],
    tags: dict[str, list[tuple[float, float]]],
    limits: tuple[int, int, int, int],
    side: int = 0,
) -> int:
    """Where to divide two neighbouring plans, best evidence first."""
    extent = mask.shape[1] if axis == 0 else mask.shape[0]
    here = int((anchor.x if axis == 0 else anchor.y) * extent)
    there = int((other.x if axis == 0 else other.y) * extent)
    span = sorted((here, there))
    near = max(2, int(extent * TITLE_CLEARANCE))
    lo, hi = _tag_window(
        axis, extent, anchor, other, prefixes, tags, (span[0] + near, span[1] - near)
    )
    gutter_lo, gutter_hi = _plan_gutter_window(axis, anchor, other, extent)
    lo = max(lo, gutter_lo)
    hi = max(lo + 4, min(hi, gutter_hi))

    # Read the ink only in the band where these two plans face each other;
    # linework from plans elsewhere on the sheet says nothing about this edge.
    left, top, right, bottom = limits
    across = mask.shape[0] if axis == 0 else mask.shape[1]
    mine = (anchor.y if axis == 0 else anchor.x) * across
    yours = (other.y if axis == 0 else other.x) * across
    reach = across * FACING_BAND
    # Side by side, the two plans face each other over the paper their titles
    # leave free: above the titles on a sheet that titles from below, and below
    # them on one that titles from above.
    behind = across * TITLE_CLEARANCE if axis == 0 and side else reach
    lead, trail = (reach, behind) if side < 0 else (behind, reach)
    band_lo = max(0, int(min(mine, yours) - lead), top if axis == 0 else left)
    band_hi = min(across, int(max(mine, yours) + trail), bottom if axis == 0 else right)
    if band_hi - band_lo < across * 0.05:
        band_lo, band_hi = (top, bottom) if axis == 0 else (left, right)
    coverage = coverage_profile(mask, axis, band_lo, band_hi)

    # Candidates, strongest evidence first: a rule the sheet drew between the
    # two titles, then the blank bands between them, widest first.
    guard = max(4, int(extent * RULE_LABEL_CLEARANCE))
    candidates = [
        line
        for line in rules
        if lo < line < hi and span[0] + guard < line < span[1] - guard
    ]
    bands = blank_bands(coverage, lo, hi)
    candidates += [(start + stop) // 2 for start, stop in bands]

    # An edge with a plan on only one side of it is not an edge between plans:
    # it is a cut through the middle of one. Take the first candidate that
    # leaves real linework on both sides, weighing only the paper the two plans
    # can occupy — which, when titles sit below their plans, reaches back past
    # the upper title and stops at the lower one.
    outer_lo = max(0, span[0] - (near if axis == 1 and side > 0 else int(reach)))
    outer_hi = min(extent, span[1] + (near if axis == 1 and side < 0 else int(reach)))
    total = float(coverage[outer_lo:outer_hi].sum())
    for cut in candidates:
        if not outer_lo < cut < outer_hi:
            continue
        before = float(coverage[outer_lo:cut].sum())
        after = float(coverage[cut:outer_hi].sum())
        if total <= 0 or min(before, after) / total >= MIN_SIDE_INK:
            return cut

    if bands:
        return (bands[0][0] + bands[0][1]) // 2
    found = sparsest_line(coverage, lo, hi)
    return (lo + hi) // 2 if found is None else found


def _cells_in_mask(
    anchors: list[WingAnchor],
    mask,
    rules_x: list[int],
    rules_y: list[int],
    prefixes: dict[str, str],
    tags: dict[str, list[tuple[float, float]]],
    bounds: tuple[int, int, int, int],
    side: int,
) -> dict[str, tuple[int, int, int, int]]:
    """Divide one block of linework between the titles that share it."""
    mh, mw = mask.shape
    shared = len(anchors) > 1
    pad = max(4, int(min(mw, mh) * 0.015))
    cells: dict[str, tuple[int, int, int, int]] = {}
    for anchor in anchors:
        ax, ay = anchor.x * mw, anchor.y * mh
        # Start from the block, widened to keep the printed title with its plan.
        left = max(0, min(bounds[0], int(ax) - pad))
        top = max(0, min(bounds[1], int(ay) - pad))
        right = min(mw, max(bounds[2], int(ax) + pad))
        bottom = min(mh, max(bounds[3], int(ay) + pad))
        for other in anchors:
            if other.instance_id == anchor.instance_id:
                continue
            ox, oy = other.x * mw, other.y * mh
            sideways = abs(other.x - anchor.x) >= abs(other.y - anchor.y)
            # Plans set diagonally across the sheet do not border each other,
            # and a cut between them would slice through whatever lies between.
            offset = abs(other.y - anchor.y) if sideways else abs(other.x - anchor.x)
            if offset > FACING_OFFSET:
                continue
            cut = _cut_between(
                0 if sideways else 1,
                anchor,
                other,
                mask,
                rules_x if sideways else rules_y,
                prefixes,
                tags,
                (left, top, right, bottom),
                side,
            )
            if sideways:
                left, right = (max(left, cut), right) if ox < ax else (left, min(right, cut))
            else:
                top, bottom = (max(top, cut), bottom) if oy < ay else (top, min(bottom, cut))

        own = tags.get(prefixes.get(anchor.instance_id, ""), [])
        if shared and len(own) >= MIN_TAGS_PER_WING:
            # Wings can interlock — one wrapping in an L over another — and no
            # straight cut separates them. A wing reaches as far as its own
            # rooms: the outermost tags plus one room's worth of margin for the
            # walls around them.
            xs = [point[0] * mw for point in own]
            ys = [point[1] * mh for point in own]
            room = _room_pad(own, mw, mh)
            lo_x, hi_x = min(xs) - room, max(xs) + room
            lo_y, hi_y = min(ys) - room, max(ys) + room
            # Keep the title with its plan, unless it stands so far off that
            # reaching for it would swallow a neighbouring wing.
            if lo_x - room <= ax <= hi_x + room:
                lo_x, hi_x = min(lo_x, ax), max(hi_x, ax)
            if lo_y - room <= ay <= hi_y + room:
                lo_y, hi_y = min(lo_y, ay), max(hi_y, ay)
            left, right = max(left, int(lo_x)), min(right, int(hi_x))
            top, bottom = max(top, int(lo_y)), min(bottom, int(hi_y))
        if right - left < 10 or bottom - top < 10:
            continue
        cells[anchor.instance_id] = (left, top, right, bottom)
    return cells


def _title_groups(
    anchors: list[WingAnchor],
    layout: PlanLayout,
) -> list[tuple[tuple[int, int, int, int], list[WingAnchor]]]:
    """The titles sharing each block of linework, with that block's bounds."""
    grouped: dict[int, list[int]] = {}
    for index, owned in enumerate(layout.claims):
        if owned:
            grouped.setdefault(owned[0], []).append(index)
    groups: list[tuple[tuple[int, int, int, int], list[WingAnchor]]] = []
    for members in grouped.values():
        pieces = [layout.blobs[j] for index in members for j in layout.claims[index]]
        bounds = (
            min(b.x1 for b in pieces),
            min(b.y1 for b in pieces),
            max(b.x2 for b in pieces),
            max(b.y2 for b in pieces),
        )
        groups.append((bounds, [anchors[index] for index in members]))
    return groups


def _wing_cells(
    anchors: list[WingAnchor],
    mask,
    plans,
    layout: PlanLayout,
    scale: float,
    width: int,
    height: int,
    tag_points: list[dict] | None = None,
) -> dict[str, PixelBox]:
    """One box per plan title, so no plan is ever clipped.

    A title owns a whole block of linework, and that block is its box. Only
    where two titles share one block — wings drawn against each other — is the
    block divided: along a rule the sheet drew between them where there is one,
    otherwise between the room tags of the two areas, otherwise along the blank
    paper between them.

    Rules are read from the ink, but blank paper is judged on `plans`, where the
    titles and caption bars between two drawings have been taken out: a plan is
    mostly white inside, so on the raw ink a room can look as empty as the paper
    between two plans.
    """
    mh, mw = mask.shape
    rules_x = rule_lines(mask, 1)
    rules_y = rule_lines(mask, 0)
    if not _tags_name_areas(anchors, tag_points):
        tag_points = []
    tags: dict[str, list[tuple[float, float]]] = {}
    for tag in tag_points or []:
        point = tag.get("point") or [0.5, 0.5]
        tags.setdefault(str(tag.get("name") or ""), []).append(
            (float(point[0]), float(point[1]))
        )

    groups = _title_groups(anchors, layout)
    if not groups:
        groups = [((0, 0, mw, mh), anchors)]

    def divide(prefixes: dict[str, str]) -> dict[str, tuple[int, int, int, int]]:
        cells: dict[str, tuple[int, int, int, int]] = {}
        for bounds, members in groups:
            cells.update(
                _cells_in_mask(
                    members, plans, rules_x, rules_y, prefixes, tags, bounds, layout.side
                )
            )
        return cells

    # Room tags only mark an area where the sheet numbers rooms by area. Where
    # they number by room type instead, a prefix lands all over the sheet, so
    # trust a prefix only once its tags prove to sit inside that plan's own cell.
    trusted: dict[str, str] = {}
    provisional = divide({})
    for instance_id, prefix in _wing_prefixes(anchors, tag_points).items():
        cell = provisional.get(instance_id)
        points = tags.get(prefix, [])
        if cell is None or not points:
            continue
        left, top, right, bottom = cell
        inside = sum(
            1
            for px, py in points
            if left <= px * mw <= right and top <= py * mh <= bottom
        )
        if inside / len(points) >= TAG_INSIDE_RATIO:
            trusted[instance_id] = prefix

    inverse = 1.0 / scale if scale > 0 else 1.0
    return {
        name: PixelBox(
            x1=int(left * inverse),
            y1=int(top * inverse),
            x2=int(round(right * inverse)),
            y2=int(round(bottom * inverse)),
        ).clamp(width, height)
        for name, (left, top, right, bottom) in divide(trusted).items()
    }


def crop_wings(
    diagram: Image.Image,
    wing_specs: list[dict],
    label_points: list[dict] | None = None,
    tag_points: list[dict] | None = None,
) -> list[WingRegion]:
    """One complete image per wing, with CV panels defining the boundaries."""
    rgb = diagram.convert("RGB")
    width, height = rgb.size

    def whole_plan(source: str = "vision_model") -> list[WingRegion]:
        crop, inner = _trim_to_ink(rgb)
        if not _is_usable_wing(crop):
            return []
        return [
            WingRegion(
                name="FULL-PLAN",
                slug="FULL-PLAN",
                box=inner,
                image=crop,
                source=source,
            )
        ]

    anchors = anchors_from_specs(wing_specs, label_points)
    real_wings = [a for a in anchors if a.name != "FULL-PLAN"]
    if not real_wings:
        return whole_plan()

    if len(real_wings) == 1:
        anchor = real_wings[0]
        crop, inner = _trim_to_ink(rgb)
        if not _is_usable_wing(crop):
            return []
        return [
            WingRegion(
                name=anchor.name,
                slug=anchor.name,
                instance_id=anchor.instance_id,
                box=inner,
                image=crop,
                vision_box=anchor.vision_box,
                source=anchor.source,
            )
        ]

    mask, scale = build_mask(rgb)
    small = rgb if scale >= 1.0 else rgb.resize(
        (mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR
    )
    plans = (plan_mask(small) > 0).astype("uint8")
    cleaned = strip_rules(plans)
    layout = plan_layout(
        cleaned if cleaned.any() else plans,
        [(anchor.x, anchor.y) for anchor in real_wings],
    )
    cells = _wing_cells(
        real_wings, mask, plans, layout, scale, width, height, tag_points
    )
    if not cells:
        return whole_plan()

    side = layout.side
    regions: list[WingRegion] = []
    for anchor in real_wings:
        box = cells.get(anchor.instance_id)
        if box is None:
            continue
        piece = rgb.crop(box.to_tuple())
        cell_w = max(1.0, float(box.x2 - box.x1))
        cell_h = max(1.0, float(box.y2 - box.y1))
        local_x = (anchor.x * width - box.x1) / cell_w
        local_y = (anchor.y * height - box.y1) / cell_h
        focus = _focus_on_wing(piece, local_x, local_y, side)
        if focus is not None:
            piece = piece.crop(focus.to_tuple())
            box = PixelBox(
                x1=box.x1 + focus.x1,
                y1=box.y1 + focus.y1,
                x2=box.x1 + focus.x2,
                y2=box.y1 + focus.y2,
            ).clamp(width, height)
        trimmed, inner = _trim_to_ink(piece)
        if not _is_usable_wing(trimmed):
            continue
        abs_box = PixelBox(
            x1=box.x1 + inner.x1,
            y1=box.y1 + inner.y1,
            x2=box.x1 + inner.x2,
            y2=box.y1 + inner.y2,
        ).clamp(width, height)
        regions.append(
            WingRegion(
                name=anchor.name,
                slug=anchor.name,
                instance_id=anchor.instance_id,
                box=abs_box,
                image=trimmed,
                vision_box=anchor.vision_box,
                source=anchor.source,
            )
        )

    if not regions:
        return whole_plan()
    return regions
