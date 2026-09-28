"""Suppress YOLO hits that sit on drawing text (labels, titles, notes).

CAD-exported PDFs carry an exact vector text layer. Letters like the "O" in
"RESTORATION" look identical to small symbols (plumbing stacks, drains…), so
YOLO occasionally boxes them. Instead of guessing with OCR — which is
unreliable on sparse line drawings — we read the word boxes straight from the
source PDF and map them through the same crop chain the wing images use:

    PDF points  →  rendered page pixels  →  wing crop  →  ROI space

A detection is dropped only when it is letter-sized AND centered on a text
word of 2+ characters. Real symbols that merely sit next to a label keep
their center outside the word box and survive; oversized detections spanning
a whole label are kept too (they are not letter shaped). Single-character
words (the "R" / "W" reference boxes on roof plans) are never used as masks,
because legitimate symbols can contain a letter glyph.

Many CAD PDFs draw in-plan labels with SHX stroke fonts — pure vector lines
with no text layer at all (the layer only covers border notes / title block).
For those, a geometric detector on the wing image finds text *lines* without
reading them: a run of 4+ letter-sized ink blobs packed edge-to-edge is text;
symbols are never spaced that tightly. Both sources are unioned into one mask.

Scanned PDFs without a text layer simply fall back to the geometric detector.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline import config
from pipeline.yolo_detect import YoloHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextWord:
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    # Weak regions (short dimension strings like 64'-0") only mask hits below
    # the primary YOLO confidence — a confident real symbol is never dropped.
    weak: bool = False

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


def _source_pdf(job_root: Path) -> Path | None:
    src = job_root / "01_source"
    if not src.is_dir():
        return None
    pdfs = sorted(src.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def pdf_words_in_page_pixels(
    pdf_path: Path,
    page_number: int,
    page_image_size: tuple[int, int],
    *,
    min_chars: int = 2,
) -> list[TextWord]:
    """Text word boxes of one PDF page, scaled to rendered-page pixels."""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed — text masking disabled")
        return []

    try:
        with fitz.open(pdf_path) as doc:
            if not (1 <= page_number <= len(doc)):
                return []
            page = doc[page_number - 1]
            rect = page.rect
            raw = page.get_text("words")
    except Exception:
        logger.exception("Text-layer read failed for %s p%d", pdf_path.name, page_number)
        return []

    if rect.width <= 0 or rect.height <= 0:
        return []
    sx = page_image_size[0] / rect.width
    sy = page_image_size[1] / rect.height

    words: list[TextWord] = []
    for x1, y1, x2, y2, text, *_ in raw:
        t = (text or "").strip()
        if len(t) < min_chars:
            continue
        words.append(TextWord(x1 * sx, y1 * sy, x2 * sx, y2 * sy, t))
    return words


def words_in_roi_space(
    words: list[TextWord],
    cv_box_on_page: dict[str, Any] | None,
    roi_box_px: dict[str, Any] | None,
) -> list[TextWord]:
    """Shift page-pixel word boxes into the wing's ROI coordinate system."""
    if not words or not cv_box_on_page:
        return []
    wx = float(cv_box_on_page.get("x1") or 0)
    wy = float(cv_box_on_page.get("y1") or 0)
    wx2 = float(cv_box_on_page.get("x2") or 0)
    wy2 = float(cv_box_on_page.get("y2") or 0)
    rx = float((roi_box_px or {}).get("x1") or 0)
    ry = float((roi_box_px or {}).get("y1") or 0)

    out: list[TextWord] = []
    for w in words:
        cx = (w.x1 + w.x2) / 2.0
        cy = (w.y1 + w.y2) / 2.0
        if not (wx <= cx <= wx2 and wy <= cy <= wy2):
            continue  # word is outside this wing crop
        out.append(
            TextWord(
                w.x1 - wx - rx,
                w.y1 - wy - ry,
                w.x2 - wx - rx,
                w.y2 - wy - ry,
                w.text,
                w.weak,
            )
        )
    return out


def detect_text_lines_on_image(
    image,
    *,
    min_glyph_h: int | None = None,
    max_glyph_h: int | None = None,
    min_glyphs: int | None = None,
) -> list[TextWord]:
    """Find text lines geometrically: runs of tightly packed letter-sized blobs.

    Catches SHX stroke-font labels ("RESTORATION", dimensions…) that have no
    PDF text layer. Detects both horizontal and vertical text lines. Returns
    boxes in image pixel coordinates (== ROI space for full_wing.jpg).
    """
    import cv2
    import numpy as np

    lo = int(config.TEXT_MASK_CV_MIN_GLYPH_H if min_glyph_h is None else min_glyph_h)
    hi = int(config.TEXT_MASK_CV_MAX_GLYPH_H if max_glyph_h is None else max_glyph_h)
    need = int(config.TEXT_MASK_CV_MIN_GLYPHS if min_glyphs is None else min_glyphs)

    gray = np.asarray(image.convert("L"))
    binary = (gray < 160).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    glyphs: list[tuple[float, float, float, float]] = []  # single letters
    marks: list[tuple[float, float, float, float]] = []  # ticks, quotes, dashes
    merged_words: list[TextWord] = []  # whole words fused into one blob

    def _is_fused_word(x: int, y: int, w: int, h: int, area: int, vertical: bool) -> bool:
        """A whole word fused into one elongated blob (incl. rotated dims).

        Distinguish text from symbol outlines / lines:
         - moderate ink fill (outlines are sparse only at edges, solid
           lines are ~100% filled, text is ~10-60%)
         - ink present in the middle band across most of the reading axis
           (letter strokes cross the middle; a hollow rectangle does not)
        """
        long_side, short_side = (h, w) if vertical else (w, h)
        if not (lo <= short_side <= hi):
            return False
        if not (3 * short_side < long_side <= 30 * short_side):
            return False
        fill = area / float(max(1, w * h))
        if not (0.08 <= fill <= 0.60):
            return False
        if vertical:
            band = binary[y : y + h, x + w // 4 : x + (3 * w) // 4]
            cover_axis = 1
        else:
            band = binary[y + h // 4 : y + (3 * h) // 4, x : x + w]
            cover_axis = 0
        if band.size == 0:
            return False
        return float((band.max(axis=cover_axis) > 0).mean()) >= 0.55

    for i in range(1, n):
        x, y, w, h, area = (int(stats[i][k]) for k in range(5))
        box = (float(x), float(y), float(x + w), float(y + h))
        if area >= 2 and max(w, h) <= hi and (h < lo or w < lo) and max(w, h) >= 2:
            marks.append(box)  # apostrophe / inch tick / dash
            continue
        if area < 4:
            continue

        if lo <= h <= hi and w <= 3 * h:
            # Letter-like blob: small and roughly compact.
            if w >= lo:
                glyphs.append(box)
            continue

        if _is_fused_word(x, y, w, h, area, vertical=False) or _is_fused_word(
            x, y, w, h, area, vertical=True
        ):
            merged_words.append(
                TextWord(float(x), float(y), float(x + w), float(y + h), f"cv:word{w}x{h}")
            )

    def chain(boxes, horizontal: bool) -> list[TextWord]:
        # Sort along the reading axis, then greedily grow runs of neighbors.
        if horizontal:
            boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        else:
            boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        used = [False] * len(boxes)
        lines: list[TextWord] = []
        for i, b in enumerate(boxes):
            if used[i]:
                continue
            run = [b]
            used[i] = True
            grown = True
            while grown:
                grown = False
                lx1 = min(r[0] for r in run)
                ly1 = min(r[1] for r in run)
                lx2 = max(r[2] for r in run)
                ly2 = max(r[3] for r in run)
                lh = (ly2 - ly1) if horizontal else (lx2 - lx1)
                gap = max(6.0, 1.2 * lh)
                for j, c in enumerate(boxes):
                    if used[j]:
                        continue
                    if horizontal:
                        # vertical centers aligned, horizontally adjacent
                        cy = (c[1] + c[3]) / 2
                        if not (ly1 - 2 <= cy <= ly2 + 2):
                            continue
                        if c[0] > lx2 + gap or c[2] < lx1 - gap:
                            continue
                    else:
                        cx = (c[0] + c[2]) / 2
                        if not (lx1 - 2 <= cx <= lx2 + 2):
                            continue
                        if c[1] > ly2 + gap or c[3] < ly1 - gap:
                            continue
                    run.append(c)
                    used[j] = True
                    grown = True
            if len(run) < need:
                continue
            x1 = min(r[0] for r in run)
            y1 = min(r[1] for r in run)
            x2 = max(r[2] for r in run)
            y2 = max(r[3] for r in run)
            w, h = x2 - x1, y2 - y1
            long_side, short_side = (w, h) if horizontal else (h, w)
            # A text line is clearly elongated along the reading axis.
            if short_side <= 0 or long_side < 2.2 * short_side:
                continue
            lines.append(TextWord(x1, y1, x2, y2, f"cv:{len(run)}glyphs"))
        return lines

    def dim_strings(vertical: bool) -> list[TextWord]:
        """Short dimension strings (64'-0"): 2+ digits with tick marks packed
        tightly, running parallel to a long dimension line. Flagged weak."""

        def ax(b):  # reading-axis view: (a1, b1, a2, b2)
            return (b[1], b[0], b[3], b[2]) if vertical else b

        def has_parallel_line(a1, a2, b1, b2, bh) -> bool:
            """A near-continuous straight ink run beside/through the chain,
            parallel to the reading axis (the dimension line itself)."""
            pa1 = max(0, int(a1 - bh))
            pa2 = int(a2 + bh)
            pb1 = max(0, int(b1 - 2.5 * bh))
            pb2 = int(b2 + 2.5 * bh)
            if vertical:
                strip = binary[pa1:pa2, pb1:pb2]  # rows = reading axis
                cover = strip.mean(axis=0) if strip.size else None
            else:
                strip = binary[pb1:pb2, pa1:pa2]
                cover = strip.mean(axis=1) if strip.size else None
            return cover is not None and bool((cover >= 0.85).any())

        letters = [ax(g) for g in glyphs]
        ticks = [ax(m) for m in marks]
        comps = [(b, True) for b in letters] + [(b, False) for b in ticks]
        comps.sort(key=lambda cb: cb[0][0])
        used = [False] * len(comps)
        out: list[TextWord] = []
        for i, (b, is_letter) in enumerate(comps):
            if used[i] or not is_letter:
                continue
            run = [(b, True)]
            used[i] = True
            lh = max(1.0, b[3] - b[1])
            grown = True
            while grown:
                grown = False
                a1 = min(r[0][0] for r in run)
                a2 = max(r[0][2] for r in run)
                b1 = min(r[0][1] for r in run)
                b2 = max(r[0][3] for r in run)
                gap = max(4.0, 0.9 * lh)
                for j, (c, c_is_letter) in enumerate(comps):
                    if used[j]:
                        continue
                    cb = (c[1] + c[3]) / 2
                    if not (b1 - 3 <= cb <= b2 + 3):
                        continue
                    if c[0] > a2 + gap or c[2] < a1 - gap:
                        continue
                    run.append((c, c_is_letter))
                    used[j] = True
                    grown = True
            n_letters = sum(1 for _, l in run if l)
            if n_letters < 2 or len(run) < 3:
                continue
            a1 = min(r[0][0] for r in run)
            a2 = max(r[0][2] for r in run)
            b1 = min(r[0][1] for r in run)
            b2 = max(r[0][3] for r in run)
            bh = max(1.0, b2 - b1)
            if (a2 - a1) > 10 * bh:
                continue
            # Must run parallel to a long line (the dimension line itself).
            if not has_parallel_line(a1, a2, b1, b2, bh):
                continue
            if vertical:
                out.append(TextWord(b1, a1, b2, a2, f"cv:dim{len(run)}", weak=True))
            else:
                out.append(TextWord(a1, b1, a2, b2, f"cv:dim{len(run)}", weak=True))
        return out

    return (
        merged_words
        + chain(glyphs, horizontal=True)
        + chain(glyphs, horizontal=False)
        + dim_strings(vertical=False)
        + dim_strings(vertical=True)
    )


def load_wing_text_words(
    job_root: Path,
    page: dict[str, Any],
    wing: dict[str, Any],
    zoom_folder: Path,
) -> list[TextWord]:
    """All text regions of this wing in ROI coordinates (PDF layer + CV lines)."""
    from PIL import Image

    words: list[TextWord] = []

    # Source 1: the PDF text layer (exact boxes for real-font text).
    pdf_path = _source_pdf(job_root)
    page_number = int(page.get("page") or 0)
    page_image_rel = page.get("page_image")
    cv_box = wing.get("cv_box_on_page")
    if pdf_path and page_number and page_image_rel and cv_box:
        page_size = None
        try:
            with Image.open(job_root / page_image_rel) as im:
                page_size = im.size
        except OSError:
            page_size = None

        if page_size:
            roi_box = None
            manifest = zoom_folder / "zooms_manifest.json"
            if manifest.is_file():
                try:
                    roi_box = json.loads(
                        manifest.read_text(encoding="utf-8")
                    ).get("roi_box_px")
                except (OSError, json.JSONDecodeError):
                    roi_box = None
            page_words = pdf_words_in_page_pixels(
                pdf_path,
                page_number,
                page_size,
                min_chars=config.TEXT_MASK_MIN_WORD_CHARS,
            )
            words.extend(words_in_roi_space(page_words, cv_box, roi_box))

    # Source 2: geometric text lines on the wing image itself — catches SHX
    # stroke-font labels that have no text layer (drawn as vector lines).
    if config.TEXT_MASK_CV_ENABLED:
        full = zoom_folder / "full_wing.jpg"
        if full.is_file():
            try:
                with Image.open(full) as im:
                    words.extend(detect_text_lines_on_image(im.convert("RGB")))
            except Exception:
                logger.exception("CV text-line detection failed for %s", full)

    return words


def dense_text_regions(
    words: list[TextWord],
    *,
    min_words: int | None = None,
    radius_px: float | None = None,
    pad_px: float | None = None,
) -> list[TextWord]:
    """Boxes around dense clusters of text — legend tables, title blocks,
    note columns, schedules.

    A drawing area carries scattered labels; a legend table stacks dozens of
    words tightly. Any YOLO detection centered inside such a region is not a
    countable plan symbol — the legend's own glyph pictures are the classic
    case (they match the legend, so the legend gate keeps them, and they are
    confident real symbol shapes, so thresholds keep them too).
    """
    need = int(config.TEXT_DENSE_MIN_WORDS if min_words is None else min_words)
    radius = float(config.TEXT_DENSE_RADIUS_PX if radius_px is None else radius_px)
    pad = float(config.TEXT_DENSE_PAD_PX if pad_px is None else pad_px)

    strong = [w for w in words if not w.weak]
    if len(strong) < need:
        return []

    centers = [((w.x1 + w.x2) / 2.0, (w.y1 + w.y2) / 2.0) for w in strong]

    # Single-link clustering on center distance <= radius (union-find).
    parent = list(range(len(strong)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    r2 = radius * radius
    for i in range(len(strong)):
        xi, yi = centers[i]
        for j in range(i + 1, len(strong)):
            dx = centers[j][0] - xi
            dy = centers[j][1] - yi
            if dx * dx + dy * dy <= r2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    clusters: dict[int, list[TextWord]] = {}
    for i, w in enumerate(strong):
        clusters.setdefault(find(i), []).append(w)

    regions: list[TextWord] = []
    for members in clusters.values():
        if len(members) < need:
            continue
        regions.append(
            TextWord(
                min(w.x1 for w in members) - pad,
                min(w.y1 for w in members) - pad,
                max(w.x2 for w in members) + pad,
                max(w.y2 for w in members) + pad,
                f"dense:{len(members)}words",
            )
        )
    return regions


def filter_hits_in_regions(
    hits: list[YoloHit],
    regions: list[TextWord],
) -> tuple[list[YoloHit], list[YoloHit]]:
    """Split hits into (kept, masked): a hit whose center falls inside any
    region is dropped regardless of confidence or size — these regions
    (legend tables / title blocks) contain no countable plan symbols."""
    if not hits or not regions:
        return hits, []
    kept: list[YoloHit] = []
    masked: list[YoloHit] = []
    for hit in hits:
        cx = (hit.x1 + hit.x2) / 2.0
        cy = (hit.y1 + hit.y2) / 2.0
        inside = any(r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2 for r in regions)
        if inside:
            logger.debug(
                "Region mask: dropped %s conf=%.2f inside dense text region",
                hit.class_name,
                hit.confidence,
            )
        (masked if inside else kept).append(hit)
    return kept, masked


def filter_hits_on_text(
    hits: list[YoloHit],
    words: list[TextWord],
    *,
    pad_px: float | None = None,
    max_height_ratio: float | None = None,
) -> tuple[list[YoloHit], list[YoloHit]]:
    """Split hits into (kept, masked-as-text).

    A hit is masked when its center lies inside a (padded) word box and the
    hit is letter-sized relative to that word (height ≤ ratio × word height).
    """
    if not hits or not words:
        return hits, []
    pad = float(config.TEXT_MASK_PAD_PX if pad_px is None else pad_px)
    ratio = float(
        config.TEXT_MASK_HEIGHT_RATIO if max_height_ratio is None else max_height_ratio
    )

    kept: list[YoloHit] = []
    masked: list[YoloHit] = []
    for hit in hits:
        cx = (hit.x1 + hit.x2) / 2.0
        cy = (hit.y1 + hit.y2) / 2.0
        h = max(1.0, hit.y2 - hit.y1)
        on_text = False
        for w in words:
            if w.height <= 0:
                continue
            if w.weak and hit.confidence >= config.YOLO_CONF:
                # Weak (dimension-string) regions never veto a confident hit.
                continue
            # For vertical text the letter size is the box width.
            letter = min(w.height, max(1.0, w.x2 - w.x1))
            if (
                w.x1 - pad <= cx <= w.x2 + pad
                and w.y1 - pad <= cy <= w.y2 + pad
                and h <= ratio * (letter + 2 * pad) + (w.height - letter)
            ):
                on_text = True
                logger.debug(
                    "Text mask: dropped %s conf=%.2f on word %r",
                    hit.class_name,
                    hit.confidence,
                    w.text,
                )
                break
        (masked if on_text else kept).append(hit)
    return kept, masked
