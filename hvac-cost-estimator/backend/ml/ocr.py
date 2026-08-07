"""OCR over the title-block crop.

Real backend: PaddleOCR. Mock backend: returns realistic title-block lines
(label line followed by value line, like a real drawing template) so the
metadata mapper is genuinely exercised without the paddle dependency.
"""

from __future__ import annotations

import logging

from PIL import Image

from config import Settings
from ml.base import ModelNotAvailableError, OcrLine

logger = logging.getLogger(__name__)


class MockOcrEngine:
    """Deterministic stand-in emitting typical title-block text lines.

    Field labels intentionally vary from the canonical schema names
    ("CONSULTANT" instead of "Engineer", "ISSUE DATE" instead of "Due date")
    to exercise the fuzzy metadata mapping.
    """

    MOCK_LINES: tuple[str, ...] = (
        "PROJECT TITLE",
        "Riverside Office Tower - Level 3 HVAC Layout",
        "CLIENT",
        "Meridian Property Group",
        "ARCHITECT",
        "Atelier North Architects",
        "CONSULTANT",
        "Vector Building Services Ltd.",
        "PROJECT ADDRESS",
        "128 Riverside Drive, Springfield",
        "ISSUE DATE",
        "2026-08-15",
        "DRAWING NO",
        "M-301",
        "SCALE",
        "1:100",
    )

    def extract(self, image: Image.Image) -> list[OcrLine]:
        return [OcrLine(text=line, confidence=0.98) for line in self.MOCK_LINES]


class PaddleOcrEngine:
    """PaddleOCR-backed engine. Downloads its default detection/recognition
    weights on first use; no custom training required."""

    def __init__(self, settings: Settings) -> None:
        try:
            import numpy as np
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ModelNotAvailableError(
                "paddleocr is not installed. Install requirements-ml.txt "
                "or set HVAC_USE_MOCK_MODELS=true."
            ) from exc
        self._np = np
        self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def extract(self, image: Image.Image) -> list[OcrLine]:
        from ml.base import BoundingBox

        array = self._np.asarray(image.convert("RGB"))
        result = self._ocr.ocr(array, cls=True)
        lines: list[OcrLine] = []
        for page in result or []:
            for entry in page or []:
                quad, (text, confidence) = entry
                xs = [point[0] for point in quad]
                ys = [point[1] for point in quad]
                lines.append(
                    OcrLine(
                        text=str(text),
                        confidence=float(confidence),
                        box=BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
                    )
                )
        # Preserve top-to-bottom reading order for the label/value pairing logic
        lines.sort(key=lambda line: (line.box.y1 if line.box else 0.0))
        return lines


def get_ocr_engine(settings: Settings):
    """Factory: mock or PaddleOCR backend depending on configuration."""
    if settings.use_mock_models:
        return MockOcrEngine()
    return PaddleOcrEngine(settings)


def ocr_logo_image(settings: Settings, image: Image.Image) -> list[OcrLine]:
    """OCR a logo/stamp crop.

    Returns an empty list in mock mode so we never invent mock title-block
    firm names (e.g. "Atelier North Architects") into Company / Firm.
    """
    if settings.use_mock_models:
        return []
    try:
        return PaddleOcrEngine(settings).extract(image)
    except ModelNotAvailableError:
        logger.warning("ocr_logo_image: PaddleOCR unavailable; skipping logo OCR")
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("ocr_logo_image: failed: %s", exc)
        return []
