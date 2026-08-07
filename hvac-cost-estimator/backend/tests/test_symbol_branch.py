"""Tests for the symbol-extraction branch: symbol detection + classification."""

from __future__ import annotations

import random

from PIL import Image

from ml.classifier import DEVICE_TYPES, MockDeviceClassifier
from ml.symbol_detector import MockSymbolDetector


def _noise_image(width: int = 1200, height: int = 800, seed: int = 7) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("L", (width, height))
    image.putdata([rng.randint(0, 255) for _ in range(width * height)])
    return image.convert("RGB")


class TestMockSymbolDetector:
    def test_boxes_within_drawing_area(self) -> None:
        image = _noise_image()
        detections = MockSymbolDetector().detect(image)

        assert MockSymbolDetector.MIN_SYMBOLS <= len(detections) <= MockSymbolDetector.MAX_SYMBOLS
        for detection in detections:
            assert 0 <= detection.box.x1 < detection.box.x2
            assert detection.box.x2 <= image.width * MockSymbolDetector.DRAWING_AREA_FRACTION
            assert 0 <= detection.box.y1 < detection.box.y2 <= image.height
            assert 0.0 < detection.score <= 1.0
            assert detection.label == "device_symbol"

    def test_deterministic_for_same_image(self) -> None:
        image = _noise_image(seed=42)
        first = MockSymbolDetector().detect(image)
        second = MockSymbolDetector().detect(image)
        assert first == second

    def test_different_images_differ(self) -> None:
        boxes_a = MockSymbolDetector().detect(_noise_image(seed=1))
        boxes_b = MockSymbolDetector().detect(_noise_image(seed=2))
        assert boxes_a != boxes_b


class TestMockDeviceClassifier:
    def test_returns_known_device_type(self) -> None:
        crop = _noise_image(64, 64, seed=3)
        device_type, confidence = MockDeviceClassifier().classify(crop)

        assert device_type in DEVICE_TYPES
        assert 0.80 <= confidence <= 0.99

    def test_deterministic_for_same_crop(self) -> None:
        crop = _noise_image(64, 64, seed=4)
        classifier = MockDeviceClassifier()
        assert classifier.classify(crop) == classifier.classify(crop)

    def test_varied_crops_cover_multiple_types(self) -> None:
        classifier = MockDeviceClassifier()
        types = {
            classifier.classify(_noise_image(64, 64, seed=seed))[0] for seed in range(30)
        }
        assert len(types) >= 3  # mock spreads detections across device types
