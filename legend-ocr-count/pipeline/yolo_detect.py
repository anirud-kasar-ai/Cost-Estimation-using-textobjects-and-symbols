"""YOLO symbol detector (Ultralytics) for legend-ocr-count."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model = None
_model_path: Path | None = None


@dataclass(frozen=True)
class YoloHit:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_name: str = ""
    legend_number: int | None = None
    # How many legend units this detection contributes (CONDUIT STUB 2 → 2)
    count_weight: int = 1
    # True when found only by the low-conf verification (rescue) pass
    rescued: bool = False

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cx"] = self.cx
        d["cy"] = self.cy
        return d


def model_path() -> Path:
    return Path(config.YOLO_MODEL_PATH)


def yolo_available() -> bool:
    path = model_path()
    if not path.is_file():
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def get_model():
    """Lazy-load and cache the YOLO weights."""
    global _model, _model_path
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    with _model_lock:
        if _model is not None and _model_path == path.resolve():
            return _model
        from ultralytics import YOLO

        logger.info("Loading YOLO model from %s", path)
        _model = YOLO(str(path))
        _model_path = path.resolve()
        return _model


def class_names() -> dict[int, str]:
    if not yolo_available():
        return {}
    try:
        model = get_model()
    except Exception:  # noqa: BLE001
        return {}
    names = getattr(model, "names", None) or {}
    return {int(k): str(v) for k, v in names.items()}


def _predict_hits(
    model,
    arr: np.ndarray,
    *,
    conf: float,
    iou: float,
    image_name: str,
    augment: bool = False,
    rescued: bool = False,
) -> list[YoloHit]:
    """One Ultralytics predict call parsed into YoloHit boxes."""
    results = model.predict(
        source=arr,
        conf=conf,
        iou=iou,
        verbose=False,
        imgsz=int(config.YOLO_IMGSZ),
        augment=augment,
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = model.names or {}
    hits: list[YoloHit] = []
    for box in boxes:
        cls_id = int(box.cls.item())
        score = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        name = str(names.get(cls_id, f"class_{cls_id}"))
        # Skip generic catch-all if configured
        if name.strip().lower() in {"object", "objects"} and not config.YOLO_KEEP_OBJECT_CLASS:
            continue
        hits.append(
            YoloHit(
                class_id=cls_id,
                class_name=name,
                confidence=score,
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                image_name=image_name,
                rescued=rescued,
            )
        )
    return hits


def detect_symbols(
    image: Image.Image,
    *,
    image_name: str = "",
    conf: float | None = None,
    iou: float | None = None,
) -> list[YoloHit]:
    """Run YOLO on a plan image; return boxes with class labels.

    When YOLO_RESCUE_ENABLED, a second verification pass re-runs the same image
    at YOLO_RESCUE_CONF (optionally with TTA) and appends its below-threshold
    boxes tagged rescued=True. Downstream NMS drops any rescue box overlapping
    a primary detection, so no symbol is counted twice.
    """
    if not yolo_available():
        return []

    conf = float(config.YOLO_CONF if conf is None else conf)
    iou = float(config.YOLO_IOU if iou is None else iou)
    model = get_model()

    # Ultralytics accepts numpy RGB
    arr = np.array(image.convert("RGB"))
    hits = _predict_hits(model, arr, conf=conf, iou=iou, image_name=image_name)
    logger.info("YOLO on %s: %d detections (conf>=%.2f)", image_name or "image", len(hits), conf)

    rescue_conf = float(config.YOLO_RESCUE_CONF)
    if config.YOLO_RESCUE_ENABLED and rescue_conf < conf:
        candidates = _predict_hits(
            model,
            arr,
            conf=rescue_conf,
            iou=iou,
            image_name=image_name,
            augment=bool(config.YOLO_RESCUE_TTA),
            rescued=True,
        )
        # Boxes at/above the primary threshold are re-finds of pass-1 symbols;
        # keep only the below-threshold ones as rescue candidates.
        candidates = [h for h in candidates if h.confidence < conf]
        if candidates:
            logger.info(
                "YOLO rescue pass on %s: %d candidates (%.2f<=conf<%.2f, tta=%s)",
                image_name or "image",
                len(candidates),
                rescue_conf,
                conf,
                config.YOLO_RESCUE_TTA,
            )
            hits.extend(candidates)
    return hits
