"""Device sub-type classification of detected symbol crops.

Real backend: a fine-tuned CNN (ResNet18 / EfficientNet-B0 via timm) exported
to ONNX for fast local CPU inference. Mock backend: deterministic label
derived from the crop content hash, so identical inputs always classify the
same way.
"""

from __future__ import annotations

import hashlib

from PIL import Image

from config import Settings
from ml.base import ModelNotAvailableError

# Class order must match the exported classifier's output layer
# (scripts/export_model.py writes weights with this exact ordering).
DEVICE_TYPES: tuple[str, ...] = (
    "supply_air_diffuser",
    "return_air_grille",
    "exhaust_grille",
    "co2_sensor",
    "temperature_sensor",
    "thermostat",
    "vav_box",
)


class MockDeviceClassifier:
    """Deterministic stand-in: device type derived from the crop's pixel hash."""

    def classify(self, crop: Image.Image) -> tuple[str, float]:
        thumb = crop.convert("L").resize((16, 16))
        size_tag = f"{crop.width}x{crop.height}".encode()
        digest = hashlib.sha256(size_tag + thumb.tobytes()).digest()
        index = digest[0] % len(DEVICE_TYPES)
        confidence = 0.80 + (digest[1] / 255) * 0.19  # stable value in [0.80, 0.99]
        return DEVICE_TYPES[index], round(confidence, 3)


class OnnxDeviceClassifier:
    """ONNX Runtime-backed classifier. Requires an exported model file."""

    INPUT_SIZE = 224
    # ImageNet normalization, matching the fine-tuning preprocessing
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, settings: Settings) -> None:
        if not settings.classifier_model_path.exists():
            raise ModelNotAvailableError(
                f"Classifier model not found at {settings.classifier_model_path}. "
                "Export it (scripts/export_model.py) or set HVAC_USE_MOCK_MODELS=true."
            )
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelNotAvailableError(
                "onnxruntime is not installed. Install requirements-ml.txt "
                "or set HVAC_USE_MOCK_MODELS=true."
            ) from exc
        self._np = np
        self._session = ort.InferenceSession(
            str(settings.classifier_model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def classify(self, crop: Image.Image) -> tuple[str, float]:
        np = self._np
        resized = crop.convert("RGB").resize((self.INPUT_SIZE, self.INPUT_SIZE))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.array(self.MEAN, dtype=np.float32)) / np.array(
            self.STD, dtype=np.float32
        )
        batch = array.transpose(2, 0, 1)[np.newaxis, :]  # HWC -> NCHW

        logits = self._session.run(None, {self._input_name: batch})[0][0]
        exp = np.exp(logits - logits.max())
        probabilities = exp / exp.sum()
        index = int(probabilities.argmax())
        return DEVICE_TYPES[index], float(probabilities[index])


def get_device_classifier(settings: Settings):
    """Factory: mock or ONNX backend depending on configuration."""
    if settings.use_mock_models:
        return MockDeviceClassifier()
    return OnnxDeviceClassifier(settings)
