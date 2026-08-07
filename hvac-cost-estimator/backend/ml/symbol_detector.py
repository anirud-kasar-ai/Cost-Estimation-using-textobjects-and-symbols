"""HVAC device symbol detection across the drawing area.

Real backend: a Detectron2 model trained on standard HVAC symbol shapes.
Mock backend: deterministic pseudo-random boxes seeded from the image content,
so the same PDF always yields the same detections (stable tests, stable demo).
"""

from __future__ import annotations

import hashlib
import random

from PIL import Image

from config import Settings
from ml.base import BoundingBox, Detection, ModelNotAvailableError


def _image_seed(image: Image.Image) -> int:
    """Stable seed derived from image size + a sparse pixel sample."""
    thumb = image.convert("L").resize((32, 32))
    digest = hashlib.sha256(bytes([image.width % 256, image.height % 256]) + thumb.tobytes())
    return int.from_bytes(digest.digest()[:8], "big")


class MockSymbolDetector:
    """Deterministic stand-in emitting plausible device boxes.

    Boxes are placed within the drawing area (left portion of the sheet,
    excluding the title-block strip) with symbol-scale sizes.
    """

    MIN_SYMBOLS = 12
    MAX_SYMBOLS = 24
    DRAWING_AREA_FRACTION = 0.70  # left side of the sheet; title block is right

    def detect(self, image: Image.Image) -> list[Detection]:
        width, height = image.size
        rng = random.Random(_image_seed(image))
        count = rng.randint(self.MIN_SYMBOLS, self.MAX_SYMBOLS)

        max_x = width * self.DRAWING_AREA_FRACTION

        detections: list[Detection] = []
        for _ in range(count):
            # Vary box size so crops differ even over blank drawing regions,
            # which spreads the mock classifier's output across device types.
            symbol_size = max(20.0, width * rng.uniform(0.012, 0.022))
            x1 = rng.uniform(0.05 * width, max_x - symbol_size)
            y1 = rng.uniform(0.05 * height, 0.95 * height - symbol_size)
            detections.append(
                Detection(
                    box=BoundingBox(x1=x1, y1=y1, x2=x1 + symbol_size, y2=y1 + symbol_size),
                    score=round(rng.uniform(0.75, 0.99), 3),
                    label="device_symbol",
                )
            )
        return detections


class Detectron2SymbolDetector:
    """Detectron2-backed symbol detector. Requires weights + config in backend/models/."""

    SCORE_THRESHOLD = 0.5

    def __init__(self, settings: Settings) -> None:
        if not settings.symbol_model_path.exists() or not settings.symbol_model_config.exists():
            raise ModelNotAvailableError(
                f"Symbol detector weights/config not found at {settings.symbol_model_path} / "
                f"{settings.symbol_model_config}. Train the model "
                "(scripts/train_symbol_detector.py) or set HVAC_USE_MOCK_MODELS=true."
            )
        try:
            import numpy as np
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise ModelNotAvailableError(
                "detectron2 is not installed. Install requirements-ml.txt and build "
                "detectron2 from source, or set HVAC_USE_MOCK_MODELS=true."
            ) from exc

        self._np = np
        cfg = get_cfg()
        cfg.merge_from_file(str(settings.symbol_model_config))
        cfg.MODEL.WEIGHTS = str(settings.symbol_model_path)
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.SCORE_THRESHOLD
        self._predictor = DefaultPredictor(cfg)

    def detect(self, image: Image.Image) -> list[Detection]:
        array = self._np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        outputs = self._predictor(array)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        return [
            Detection(
                box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                score=float(score),
                label="device_symbol",
            )
            for (x1, y1, x2, y2), score in zip(boxes, scores)
        ]


def get_symbol_detector(settings: Settings):
    """Factory: mock or Detectron2 backend depending on configuration."""
    if settings.use_mock_models:
        return MockSymbolDetector()
    return Detectron2SymbolDetector(settings)
