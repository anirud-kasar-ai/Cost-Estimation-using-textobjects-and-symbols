"""Tests for PDF -> image conversion and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ml.pdf_to_image import PdfValidationError, convert_pdf_to_images, validate_pdf


def test_converts_sample_pdf_to_png(sample_pdf: Path, tmp_path: Path) -> None:
    paths = convert_pdf_to_images(sample_pdf, tmp_path / "out", dpi=150)

    assert len(paths) == 1
    assert paths[0].name == "page_001.png"
    with Image.open(paths[0]) as image:
        # A3 landscape at 150 DPI is roughly 2455 x 1737 px
        assert image.width > 2000
        assert image.height > 1500


def test_rejects_non_pdf_file(tmp_path: Path) -> None:
    fake = tmp_path / "not_a_pdf.pdf"
    fake.write_bytes(b"hello, definitely not a pdf")

    with pytest.raises(PdfValidationError, match="does not look like a PDF"):
        validate_pdf(fake)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PdfValidationError, match="not found"):
        validate_pdf(tmp_path / "missing.pdf")


def test_rejects_corrupt_pdf_body(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7\ngarbage garbage garbage")

    with pytest.raises(PdfValidationError):
        convert_pdf_to_images(corrupt, tmp_path / "out")


def test_truncates_when_over_max_pages(sample_pdf: Path, tmp_path: Path) -> None:
    """Large bid sets truncate rendering instead of hard-failing."""
    from scripts.generate_sample_pdf import generate_sample_pdf

    multi = tmp_path / "multi.pdf"
    generate_sample_pdf(multi, pages=3)
    paths = convert_pdf_to_images(multi, tmp_path / "out", dpi=72, max_pages=2)
    assert len(paths) == 2
