from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline import symbol_library
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo


def _png(color: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    img = Image.new("RGB", (40, 40), "white")
    for x in range(10, 30):
        for y in range(10, 30):
            img.putpixel((x, y), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _use_tmp_library(monkeypatch, tmp_path: Path) -> None:
    lib_dir = tmp_path / "symbol_library"
    monkeypatch.setattr(symbol_library, "LIBRARY_DIR", lib_dir)
    monkeypatch.setattr(symbol_library, "INDEX_PATH", lib_dir / "index.json")


def test_library_roundtrip_resolves_missing_glyph(monkeypatch, tmp_path: Path):
    _use_tmp_library(monkeypatch, tmp_path)

    rich = SymbolTableInfo(
        entries=[
            SymbolEntry(
                description="CAT6A DATA DROP LOCATION (QTY = 1) - NETWORK CAMERA",
                symbol_image_png=_png(),
            )
        ]
    )
    assert symbol_library.add_from_info(rich, source="rich.pdf") == 1

    # Same symbol described with slightly different punctuation, no artwork.
    poor = SymbolTableInfo(
        entries=[
            SymbolEntry(
                description="CAT6A DATA DROP LOCATION (QTY = 1) NETWORK CAMERA",
            )
        ]
    )
    assert symbol_library.resolve_info(poor) == 1
    entry = poor.entries[0]
    assert entry.symbol_image_png == _png()
    assert entry.glyph_source == "library"


def test_library_skips_synthetic_glyphs(monkeypatch, tmp_path: Path):
    _use_tmp_library(monkeypatch, tmp_path)

    info = SymbolTableInfo(
        entries=[
            SymbolEntry(
                description="FIRE ALARM CONTROL PANEL",
                symbol="FACP",
                symbol_image_png=_png(),
                glyph_source="synthetic",
            )
        ]
    )
    assert symbol_library.add_from_info(info, source="badges.pdf") == 0

    lookup = SymbolTableInfo(
        entries=[SymbolEntry(description="FIRE ALARM CONTROL PANEL", symbol="FACP")]
    )
    assert symbol_library.resolve_info(lookup) == 0


def test_variant_picked_by_part_number(monkeypatch, tmp_path: Path):
    _use_tmp_library(monkeypatch, tmp_path)

    red = _png((200, 0, 0))
    blue = _png((0, 0, 200))
    rich = SymbolTableInfo(
        entries=[
            SymbolEntry(
                description="SURFACE RACEWAY",
                part_number="WM2300",
                symbol_image_png=red,
            ),
            SymbolEntry(
                description="SURFACE RACEWAY",
                part_number="WM5400",
                symbol_image_png=blue,
            ),
        ]
    )
    assert symbol_library.add_from_info(rich, source="rich.pdf") == 2

    poor = SymbolTableInfo(
        entries=[SymbolEntry(description="SURFACE RACEWAY", part_number="WM5400")]
    )
    assert symbol_library.resolve_info(poor) == 1
    assert poor.entries[0].symbol_image_png == blue
