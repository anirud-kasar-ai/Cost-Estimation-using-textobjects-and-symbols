from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.classify.glyph_identify import identify_candidates_by_glyphs
from pipeline.cv.nms import BoundingBox
from pipeline.cv.template_match import score_glyph_on_crop
from pipeline.detect.candidates import Candidate
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.taxonomy.builder import taxonomy_from_legend


def _png_bytes(draw_fn) -> bytes:
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    draw_fn(ImageDraw.Draw(img), img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ap_png() -> bytes:
    def draw(d, img):
        d.ellipse([4, 4, 28, 28], outline=(0, 0, 0), width=3)
        d.polygon([(16, 8), (24, 24), (8, 24)], fill=(0, 0, 0))

    return _png_bytes(draw)


def _g_png() -> bytes:
    def draw(d, img):
        d.text((8, 6), "G", fill=(0, 0, 0))

    return _png_bytes(draw)


def _four_png() -> bytes:
    def draw(d, img):
        d.text((10, 6), "4", fill=(0, 0, 0))

    return _png_bytes(draw)


def test_score_glyph_on_crop_identical_is_high():
    glyph = Image.new("RGB", (24, 24), (255, 255, 255))
    ImageDraw.Draw(glyph).rectangle([6, 6, 18, 18], fill=(0, 0, 0))
    assert score_glyph_on_crop(glyph, glyph) >= 0.9


def test_identify_does_not_hunt_other_legend_glyphs():
    """A named mark is checked only against its own glyph, not the full legend."""
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="AP", description="ACCESS POINT", symbol_image_png=_ap_png()),
            SymbolEntry(symbol="G", description="GROUND BOX", symbol_image_png=_g_png()),
        ]
    )
    taxonomy = taxonomy_from_legend(legend)
    ap_img = Image.open(BytesIO(_ap_png())).convert("RGB")
    canvas = Image.new("RGB", (80, 80), (255, 255, 255))
    canvas.paste(ap_img, (20, 20))
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(20, 20, 52, 52),
        shape_class_guess="letter_tag",
        score=0.8,
        source="ocr",
        ocr_text="G",
        resolved_key="G",
        status="resolved",
        candidate_keys=["G"],
        classify_source="context_rule",
    )
    out, stats = identify_candidates_by_glyphs(
        canvas,
        [cand],
        legend=legend,
        glyph_dir=None,
        taxonomy=taxonomy,
        valid_keys={"AP", "G"},
    )
    assert out[0].resolved_key == "G"
    assert stats["relabeled"] == 0
    assert stats["compared"] == 1


def test_identify_names_unresolved_number_tag():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="4", description="TYPE 4 DEVICE", symbol_image_png=_four_png()),
            SymbolEntry(symbol="#", description="DATA DROP", symbol_image_png=_ap_png()),
        ]
    )
    taxonomy = taxonomy_from_legend(legend)
    four_img = Image.open(BytesIO(_four_png())).convert("RGB")
    canvas = Image.new("RGB", (80, 80), (255, 255, 255))
    canvas.paste(four_img, (20, 20))
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(20, 20, 52, 52),
        shape_class_guess="digit_label",
        score=0.7,
        source="ocr",
        ocr_text="4",
        status="unresolved",
        classify_source="unmatched",
    )
    out, stats = identify_candidates_by_glyphs(
        canvas,
        [cand],
        legend=legend,
        glyph_dir=None,
        taxonomy=taxonomy,
        valid_keys={"4", "#"},
    )
    assert out[0].resolved_key == "4"
    assert stats["assigned"] == 1


def test_identify_does_not_turn_triangle_qty_into_number_symbol():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="4", description="TYPE 4 DEVICE", symbol_image_png=_four_png()),
            SymbolEntry(symbol="#", description="DATA DROP", symbol_image_png=_ap_png()),
        ]
    )
    taxonomy = taxonomy_from_legend(legend)
    canvas = Image.new("RGB", (80, 80), (255, 255, 255))
    ImageDraw.Draw(canvas).polygon([(30, 22), (42, 48), (18, 48)], fill=(0, 0, 0))
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(16, 20, 44, 50),
        shape_class_guess="triangle_like",
        score=0.85,
        source="triangle",
        resolved_key="#",
        status="resolved",
        candidate_keys=["#"],
        classify_source="context_rule",
    )
    out, _stats = identify_candidates_by_glyphs(
        canvas,
        [cand],
        legend=legend,
        glyph_dir=None,
        taxonomy=taxonomy,
        valid_keys={"4", "#"},
    )
    assert out[0].resolved_key == "#"


def test_identify_skips_when_detected_key_missing_from_folder(tmp_path: Path):
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="AP", description="ACCESS POINT", symbol_image_png=_ap_png()),
            SymbolEntry(symbol="G", description="GROUND BOX", symbol_image_png=_g_png()),
        ]
    )
    taxonomy = taxonomy_from_legend(legend)
    glyph_dir = tmp_path / "07_glyphs"
    glyph_dir.mkdir()
    Image.open(BytesIO(_ap_png())).save(glyph_dir / "AP_13.png")
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(20, 20, 52, 52),
        shape_class_guess="letter_tag",
        score=0.8,
        source="ocr",
        ocr_text="G",
        resolved_key="G",
        status="resolved",
        candidate_keys=["G"],
        classify_source="context_rule",
    )
    canvas = Image.new("RGB", (80, 80), (255, 255, 255))
    out, stats = identify_candidates_by_glyphs(
        canvas,
        [cand],
        legend=legend,
        glyph_dir=glyph_dir,
        taxonomy=taxonomy,
        valid_keys={"AP", "G"},
    )
    assert out[0].resolved_key == "G"
    assert stats["no_glyph"] == 1
    assert stats["compared"] == 0


def test_identify_confirms_only_the_folder_glyph_for_detected_key(tmp_path: Path):
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="AP", description="ACCESS POINT", symbol_image_png=_ap_png()),
            SymbolEntry(symbol="G", description="GROUND BOX", symbol_image_png=_g_png()),
        ]
    )
    taxonomy = taxonomy_from_legend(legend)
    glyph_dir = tmp_path / "07_glyphs"
    glyph_dir.mkdir()
    Image.open(BytesIO(_g_png())).save(glyph_dir / "G_1.png")
    Image.open(BytesIO(_ap_png())).save(glyph_dir / "AP_13.png")
    g_img = Image.open(BytesIO(_g_png())).convert("RGB")
    canvas = Image.new("RGB", (80, 80), (255, 255, 255))
    canvas.paste(g_img, (20, 20))
    cand = Candidate(
        tile_id="t",
        bbox=BoundingBox(20, 20, 52, 52),
        shape_class_guess="letter_tag",
        score=0.8,
        source="ocr",
        ocr_text="G",
        resolved_key="G",
        status="resolved",
        candidate_keys=["G"],
        classify_source="context_rule",
    )
    out, stats = identify_candidates_by_glyphs(
        canvas,
        [cand],
        legend=legend,
        glyph_dir=glyph_dir,
        taxonomy=taxonomy,
        valid_keys={"AP", "G"},
    )
    assert out[0].resolved_key == "G"
    assert stats["compared"] == 1
    assert stats["confirmed"] == 1
    assert stats["no_glyph"] == 0
