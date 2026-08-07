"""Title-block / region-of-interest detection.

Real backend: a Detectron2 model trained to locate the title-block strip on a
drawing sheet. Mock backend: assumes the standard sheet layout where the
title block occupies the right-hand strip of the page (true for the vast
majority of HVAC drawing templates, and exactly how the synthetic sample PDF
is laid out).
"""

from __future__ import annotations

import logging

from PIL import Image

from config import Settings
from ml.base import BoundingBox, Detection, ModelNotAvailableError

logger = logging.getLogger(__name__)


class MockRoiDetector:
    """Deterministic stand-in: right-hand strip of the sheet is the title block."""

    ROI_WIDTH_FRACTION = 0.28

    def detect(self, image: Image.Image) -> Detection | None:
        width, height = image.size
        box = BoundingBox(
            x1=width * (1 - self.ROI_WIDTH_FRACTION),
            y1=0,
            x2=width,
            y2=height,
        )
        return Detection(box=box, score=0.99, label="title_block")


class Detectron2RoiDetector:
    """Detectron2-backed ROI detector. Requires weights + config in backend/models/."""

    def __init__(self, settings: Settings) -> None:
        if not settings.roi_model_path.exists() or not settings.roi_model_config.exists():
            raise ModelNotAvailableError(
                f"ROI detector weights/config not found at {settings.roi_model_path} / "
                f"{settings.roi_model_config}. Train the model (scripts/train_roi_detector.py) "
                "or set HVAC_USE_MOCK_MODELS=true."
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
        cfg.merge_from_file(str(settings.roi_model_config))
        cfg.MODEL.WEIGHTS = str(settings.roi_model_path)
        cfg.MODEL.DEVICE = "cpu"
        self._predictor = DefaultPredictor(cfg)

    def detect(self, image: Image.Image) -> Detection | None:
        array = self._np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        outputs = self._predictor(array)
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            return None
        scores = instances.scores.numpy()
        best = int(scores.argmax())
        x1, y1, x2, y2 = instances.pred_boxes.tensor.numpy()[best].tolist()
        return Detection(
            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            score=float(scores[best]),
            label="title_block",
        )


def get_roi_detector(settings: Settings):
    """Factory: mock or Detectron2 backend depending on configuration."""
    if settings.use_mock_models:
        return MockRoiDetector()
    return Detectron2RoiDetector(settings)


def crop_roi(image: Image.Image, detection: Detection) -> Image.Image:
    """Crop the detected ROI out of the page image."""
    width, height = image.size
    return image.crop(detection.box.clamp(width, height).to_int_tuple())
