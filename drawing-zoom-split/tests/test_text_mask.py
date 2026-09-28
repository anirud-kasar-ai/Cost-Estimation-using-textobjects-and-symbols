"""Text-mask filter: detections on drawing text must not be counted."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.text_mask import (  # noqa: E402
    TextWord,
    dense_text_regions,
    filter_hits_in_regions,
    filter_hits_on_text,
    words_in_roi_space,
)
from pipeline.yolo_detect import YoloHit  # noqa: E402


def _hit(x1, y1, x2, y2, conf=0.9, name="PLUMBING STACK") -> YoloHit:
    return YoloHit(
        class_id=0,
        class_name=name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_name="t.jpg",
    )


# "RESTORATION" label: 110px wide, 14px tall
WORD = TextWord(100, 100, 210, 114, "RESTORATION")


def test_letter_sized_hit_on_word_is_masked():
    # letter "O" boxed as a plumbing stack, centered inside the word
    fp = _hit(130, 100, 144, 114)
    kept, masked = filter_hits_on_text([fp], [WORD])
    assert kept == []
    assert len(masked) == 1


def test_symbol_next_to_label_survives():
    # real symbol left of the label — center outside the word box
    real = _hit(70, 98, 92, 116)
    kept, masked = filter_hits_on_text([real], [WORD])
    assert len(kept) == 1
    assert masked == []


def test_large_symbol_overlapping_label_survives():
    # a big curb outline behind the text is not letter-shaped
    big = _hit(120, 60, 200, 160)
    kept, masked = filter_hits_on_text([big], [WORD])
    assert len(kept) == 1
    assert masked == []


def _legend_table_words(x0=1000.0, y0=100.0, rows=6, cols=2) -> list[TextWord]:
    """Tightly stacked words like a legend table / title block."""
    words = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * 120
            y = y0 + r * 22
            words.append(TextWord(x, y, x + 100, y + 14, f"ENTRY{r}{c}"))
    return words


def test_dense_word_cluster_masks_confident_hit_inside():
    """A legend table's glyph picture is a real symbol shape at high conf —
    only the dense-region mask can drop it."""
    words = _legend_table_words()
    regions = dense_text_regions(words, min_words=10, radius_px=140)
    assert len(regions) == 1

    glyph_in_table = _hit(1050, 140, 1080, 170, conf=0.93)
    real_on_plan = _hit(200, 300, 230, 330, conf=0.93)
    kept, masked = filter_hits_in_regions([glyph_in_table, real_on_plan], regions)
    assert len(masked) == 1 and masked[0].x1 == 1050
    assert len(kept) == 1 and kept[0].x1 == 200


def test_scattered_plan_labels_form_no_dense_region():
    """Normal in-plan labels are spread out — never masked as a region."""
    words = [
        TextWord(i * 600.0, (i % 3) * 500.0, i * 600.0 + 90, (i % 3) * 500.0 + 14, f"LBL{i}")
        for i in range(12)
    ]
    assert dense_text_regions(words, min_words=10, radius_px=140) == []


def test_weak_words_do_not_build_dense_regions():
    """Dimension strings (weak) must not conjure a mask region."""
    words = [
        TextWord(100 + i * 30.0, 100.0, 120 + i * 30.0, 112.0, f"dim{i}", weak=True)
        for i in range(15)
    ]
    assert dense_text_regions(words, min_words=10, radius_px=140) == []


def test_no_words_keeps_everything():
    hits = [_hit(0, 0, 20, 20)]
    kept, masked = filter_hits_on_text(hits, [])
    assert kept == hits
    assert masked == []


def test_cv_detector_finds_stroke_text_not_symbol_outlines():
    """A fused word blob is text; a hollow rectangle symbol is not."""
    from PIL import Image, ImageDraw

    from pipeline.text_mask import detect_text_lines_on_image

    img = Image.new("RGB", (400, 200), "white")
    dr = ImageDraw.Draw(img)
    # Fake fused word: dense vertical strokes across a 70x10 area (like
    # "RESTORATION" rendered at small scale where letters touch).
    for x in range(100, 170, 4):
        dr.line((x, 60, x, 70), fill="black", width=2)
    dr.line((100, 65, 170, 65), fill="black", width=1)
    # Hollow rectangle symbol outline 36x20 (like a curb) — must NOT match.
    dr.rectangle((250, 100, 286, 120), outline="black", width=2)

    lines = detect_text_lines_on_image(img, min_glyph_h=5, max_glyph_h=26, min_glyphs=4)
    assert any(
        w.x1 <= 135 <= w.x2 and w.y1 <= 65 <= w.y2 for w in lines
    ), f"word not detected: {lines}"
    assert not any(
        w.x1 <= 268 <= w.x2 and w.y1 <= 110 <= w.y2 for w in lines
    ), f"symbol outline wrongly masked: {lines}"


def test_cv_detector_finds_vertical_dimension_text():
    """Rotated dimension strings (6'-0" written vertically) are text too."""
    from PIL import Image, ImageDraw

    from pipeline.text_mask import detect_text_lines_on_image

    img = Image.new("RGB", (200, 300), "white")
    dr = ImageDraw.Draw(img)
    # Fused vertical word: dense horizontal strokes stacked in a 10x70 column.
    for y in range(100, 170, 4):
        dr.line((60, y, 70, y), fill="black", width=2)
    dr.line((65, 100, 65, 170), fill="black", width=1)
    # A plain vertical wall line — must NOT match (fill ~100%).
    dr.line((140, 50, 140, 250), fill="black", width=3)

    lines = detect_text_lines_on_image(img, min_glyph_h=5, max_glyph_h=26, min_glyphs=4)
    assert any(
        w.x1 <= 65 <= w.x2 and w.y1 <= 135 <= w.y2 for w in lines
    ), f"vertical word not detected: {lines}"
    assert not any(
        w.x1 <= 140 <= w.x2 and w.y1 <= 150 <= w.y2 for w in lines
    ), f"wall line wrongly masked: {lines}"


def _draw_vertical_dim_string(dr):
    """Vertical 64'-0'' beside a long vertical dimension line at x=100."""
    dr.line((100, 20, 100, 280), fill="black", width=2)  # dimension line
    # digit-sized blobs stacked vertically at x~112 (rotated digits)
    for cy in (120, 138, 156):
        dr.ellipse((108, cy - 7, 122, cy + 7), outline="black", width=2)
    dr.line((110, 130, 120, 130), fill="black", width=2)  # tick between digits
    dr.line((110, 148, 120, 148), fill="black", width=2)


def test_dimension_string_masks_low_conf_only():
    """Digits on a dimension line mask rescue-band hits, never confident ones."""
    from PIL import Image, ImageDraw

    from pipeline.text_mask import detect_text_lines_on_image

    img = Image.new("RGB", (300, 300), "white")
    _draw_vertical_dim_string(ImageDraw.Draw(img))
    lines = detect_text_lines_on_image(img, min_glyph_h=5, max_glyph_h=26, min_glyphs=4)
    dims = [w for w in lines if w.weak]
    assert dims, f"dimension string not detected: {lines}"
    d = dims[0]
    assert d.x1 <= 115 <= d.x2 and d.y1 <= 138 <= d.y2

    low = _hit(106, 112, 124, 128, conf=0.20)  # rescue-band FP on the "6"
    high = _hit(106, 112, 124, 128, conf=0.90)  # confident hit, same spot
    kept, masked = filter_hits_on_text([low, high], dims)
    assert [m.confidence for m in masked] == [0.20]
    assert [k.confidence for k in kept] == [0.90]


def test_symbol_pair_without_dimension_line_not_masked():
    """Two adjacent real symbols with no long line nearby are never text."""
    from PIL import Image, ImageDraw

    from pipeline.text_mask import detect_text_lines_on_image

    img = Image.new("RGB", (300, 300), "white")
    dr = ImageDraw.Draw(img)
    dr.rectangle((100, 100, 118, 118), outline="black", width=2)
    dr.rectangle((100, 124, 118, 142), outline="black", width=2)
    lines = detect_text_lines_on_image(img, min_glyph_h=5, max_glyph_h=26, min_glyphs=4)
    assert not any(
        w.x1 <= 109 <= w.x2 and w.y1 <= 120 <= w.y2 for w in lines
    ), f"symbol pair wrongly masked: {lines}"


def test_words_map_into_roi_space_and_clip_to_wing():
    words = [
        TextWord(1000, 2000, 1100, 2014, "RESTORATION"),  # inside wing
        TextWord(50, 50, 90, 64, "TITLE"),  # outside wing crop
    ]
    cv_box = {"x1": 171, "y1": 1066, "x2": 2479, "y2": 2108}
    roi_box = {"x1": 10, "y1": 20, "x2": 2300, "y2": 1040}
    mapped = words_in_roi_space(words, cv_box, roi_box)
    assert len(mapped) == 1
    w = mapped[0]
    assert w.text == "RESTORATION"
    assert w.x1 == 1000 - 171 - 10
    assert w.y1 == 2000 - 1066 - 20
