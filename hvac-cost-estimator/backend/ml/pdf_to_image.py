"""PDF -> high-resolution page image conversion.

Primary backend is ``pdf2image`` (requires the poppler binaries, per the
project spec). When poppler is not installed — common on Windows — we fall
back to PyMuPDF, which is pure pip and renders equivalently for our purposes.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"


class PdfValidationError(ValueError):
    """The uploaded file is not a readable PDF (wrong type, corrupt, too long)."""


class PdfConversionError(RuntimeError):
    """PDF rendering failed for an unexpected reason."""


def validate_pdf(pdf_path: Path) -> None:
    """Cheap sanity checks before rendering. Raises :class:`PdfValidationError`."""
    if not pdf_path.exists():
        raise PdfValidationError(f"File not found: {pdf_path}")
    header = pdf_path.open("rb").read(len(PDF_MAGIC))
    if header != PDF_MAGIC:
        raise PdfValidationError(
            "File does not look like a PDF (missing %PDF header). "
            "Please upload a valid PDF drawing."
        )


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
    max_pages: int = 40,
    poppler_path: Path | None = None,
) -> list[Path]:
    """Render PDF pages to PNGs in ``output_dir``.

    Pages beyond ``max_pages`` are truncated (not rejected) so large bid
    sets still process. Returns the ordered list of written image paths.
    """
    validate_pdf(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _convert_with_pdf2image(pdf_path, output_dir, dpi, max_pages, poppler_path)
    except _PopplerMissing:
        logger.info("poppler not available, falling back to PyMuPDF for %s", pdf_path.name)
        return _convert_with_pymupdf(pdf_path, output_dir, dpi, max_pages)


class _PopplerMissing(Exception):
    pass


def _page_path(output_dir: Path, page_number: int) -> Path:
    return output_dir / f"page_{page_number:03d}.png"


def _convert_with_pdf2image(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    max_pages: int,
    poppler_path: Path | None,
) -> list[Path]:
    from pdf2image import convert_from_path, pdfinfo_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
    )

    poppler = str(poppler_path) if poppler_path else None
    try:
        info = pdfinfo_from_path(str(pdf_path), poppler_path=poppler)
    except PDFInfoNotInstalledError as exc:
        raise _PopplerMissing from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise PdfValidationError(f"PDF appears to be corrupt or unreadable: {exc}") from exc

    page_count = int(info.get("Pages", 0))
    if page_count == 0:
        raise PdfValidationError("PDF contains no pages.")
    last_page = min(page_count, max_pages)
    if page_count > max_pages:
        logger.warning(
            "PDF has %d pages; rendering only the first %d for the CV pipeline",
            page_count,
            max_pages,
        )

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        poppler_path=poppler,
        first_page=1,
        last_page=last_page,
    )
    paths: list[Path] = []
    for index, image in enumerate(images, start=1):
        path = _page_path(output_dir, index)
        image.save(path, format="PNG")
        paths.append(path)
    return paths


def _convert_with_pymupdf(
    pdf_path: Path, output_dir: Path, dpi: int, max_pages: int
) -> list[Path]:
    import fitz  # PyMuPDF

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:  # fitz raises various error types for bad files
        raise PdfValidationError(f"PDF appears to be corrupt or unreadable: {exc}") from exc

    with document:
        if document.page_count == 0:
            raise PdfValidationError("PDF contains no pages.")
        render_count = min(document.page_count, max_pages)
        if document.page_count > max_pages:
            logger.warning(
                "PDF has %d pages; rendering only the first %d for the CV pipeline",
                document.page_count,
                max_pages,
            )
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        paths: list[Path] = []
        for index in range(render_count):
            page = document.load_page(index)
            # alpha=False cuts raster cost/size; PNG is kept for API/UI compatibility.
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            path = _page_path(output_dir, index + 1)
            pixmap.save(str(path))
            paths.append(path)
    if not paths:
        raise PdfConversionError("PDF contains no renderable pages.")
    return paths
