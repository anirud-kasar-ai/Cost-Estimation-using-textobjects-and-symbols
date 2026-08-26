"""PDF / image → page JPEGs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def render_pages(source: Path, pages_dir: Path) -> list[Path]:
    """Write ``page_001.jpg`` … into ``pages_dir``. Returns written paths."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf(source, pages_dir)
    if suffix in IMAGE_SUFFIXES:
        image = Image.open(source).convert("RGB")
        dest = pages_dir / "page_001.jpg"
        image.save(dest, format="JPEG", quality=90, optimize=True)
        return [dest]
    raise ValueError(f"Unsupported file type: {suffix}")


def _render_pdf(pdf_path: Path, pages_dir: Path) -> list[Path]:
    import fitz

    dpi = max(72, int(config.VISION_PDF_DPI))
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    written: list[Path] = []
    doc = fitz.open(pdf_path)
    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            dest = pages_dir / f"page_{index + 1:03d}.jpg"
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            image.convert("RGB").save(dest, format="JPEG", quality=90, optimize=True)
            written.append(dest)
    finally:
        doc.close()
    return written
