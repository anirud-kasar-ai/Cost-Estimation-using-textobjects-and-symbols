"""Shared datatypes and interfaces for the ML pipeline.

Every model wrapper implements one of the small Protocols below. Two backends
exist per wrapper: a deterministic mock (default) and a real model wrapper
that lazily imports its heavy dependencies. The rest of the pipeline only
depends on these interfaces, so swapping mock -> real is a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image


class ModelNotAvailableError(RuntimeError):
    """Raised when a real model backend is requested but deps/weights are missing."""


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
        )

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class Detection:
    """A detected region with confidence and optional label."""

    box: BoundingBox
    score: float
    label: str | None = None


@dataclass(frozen=True)
class OcrLine:
    """One line of OCR output."""

    text: str
    confidence: float
    box: BoundingBox | None = None


@dataclass(frozen=True)
class ClassifiedDevice:
    """A detected symbol assigned to a device type by the classifier."""

    device_type: str
    confidence: float
    detection: Detection
    page_number: int = 1


@dataclass
class PageResult:
    """Per-page intermediate pipeline output (useful for debugging/UI overlays)."""

    page_number: int
    image_path: str
    roi: Detection | None = None
    ocr_lines: list[OcrLine] = field(default_factory=list)
    devices: list[ClassifiedDevice] = field(default_factory=list)


class RoiDetector(Protocol):
    """Locates the title-block / text region of interest on a page image."""

    def detect(self, image: Image.Image) -> Detection | None: ...


class SymbolDetector(Protocol):
    """Detects HVAC device symbol bounding boxes on a page image."""

    def detect(self, image: Image.Image) -> list[Detection]: ...


class DeviceClassifier(Protocol):
    """Classifies a cropped symbol image into a device type."""

    def classify(self, crop: Image.Image) -> tuple[str, float]: ...


class OcrEngine(Protocol):
    """Extracts text lines from an image region."""

    def extract(self, image: Image.Image) -> list[OcrLine]: ...
