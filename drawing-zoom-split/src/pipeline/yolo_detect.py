"""YOLO symbol detector (Ultralytics) for drawing-zoom-split zoom tiles."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_models: dict[str, Any] = {}

# Test-only classes baked into some checkpoints; never detected or reported.
IGNORED_CLASSES = {"object", "objects", "classroom 15"}


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


_class_conf_cache: dict[str, dict[str, float]] | None = None


def class_conf_overrides(weights: Path | None = None) -> dict[str, float]:
    """Per-class minimum confidence floors (eval-derived) for one model.

    The JSON at config.YOLO_CLASS_CONF_PATH is keyed by model weight stem
    ({"symbol_detector_tech": {"LADDER": 0.7, ...}, ...}) because the same
    class name can need different floors in different models. Returns the
    overrides for the given weights (default: configured model); empty when
    the file or the model entry is missing.
    """
    global _class_conf_cache
    if _class_conf_cache is None:
        data: dict[str, dict[str, float]] = {}
        path = Path(config.YOLO_CLASS_CONF_PATH)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                data = {
                    str(model): {str(k): float(v) for k, v in entries.items()}
                    for model, entries in raw.items()
                }
                logger.info(
                    "Loaded per-class conf overrides for %d models from %s",
                    len(data),
                    path.name,
                )
            except (OSError, ValueError, AttributeError) as exc:
                logger.warning("Ignoring unreadable class conf overrides %s: %s", path, exc)
        _class_conf_cache = data
    stem = (weights or model_path()).stem.lower()
    return _class_conf_cache.get(stem, {})


def available_model_paths() -> list[Path]:
    """All usable weight files: the configured default plus every *.pt in model/."""
    seen: set[str] = set()
    out: list[Path] = []
    candidates = [model_path()]
    model_dir = getattr(config, "YOLO_MODEL_DIR", None)
    if model_dir and Path(model_dir).is_dir():
        candidates.extend(sorted(Path(model_dir).glob("*.pt")))
    for path in candidates:
        if not path.is_file():
            continue
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def yolo_available(weights: Path | None = None) -> bool:
    path = weights or model_path()
    if not path.is_file():
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def get_model(weights: Path | None = None):
    """Lazy-load and cache YOLO weights (supports multiple models)."""
    path = weights or model_path()
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    key = str(path.resolve()).lower()
    with _model_lock:
        model = _models.get(key)
        if model is not None:
            return model
        from ultralytics import YOLO

        logger.info("Loading YOLO model from %s", path)
        model = YOLO(str(path))
        _models[key] = model
        return model


def class_names(weights: Path | None = None) -> dict[int, str]:
    if not yolo_available(weights):
        return {}
    try:
        model = get_model(weights)
    except Exception:  # noqa: BLE001
        return {}
    names = getattr(model, "names", None) or {}
    return {
        int(k): str(v)
        for k, v in names.items()
        if str(v).strip().lower() not in IGNORED_CLASSES
    }


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
        # Skip test-only / catch-all classes
        if name.strip().lower() in IGNORED_CLASSES:
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
    weights: Path | None = None,
) -> list[YoloHit]:
    """Run YOLO on a plan image; return boxes with class labels.

    When YOLO_RESCUE_ENABLED, a second verification pass re-runs the same image
    at YOLO_RESCUE_CONF (optionally with TTA) and appends its below-threshold
    boxes tagged rescued=True. Downstream NMS drops any rescue box overlapping
    a primary detection, so no symbol is counted twice.
    """
    if not yolo_available(weights):
        return []

    conf = float(config.YOLO_CONF if conf is None else conf)
    iou = float(config.YOLO_IOU if iou is None else iou)
    model = get_model(weights)

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
        # TTA can find symbols the plain pass missed outright (e.g. small
        # PLUMBING STACKs in rows). Strong hits (conf >= primary threshold)
        # are as trustworthy as pass-1 detections: un-flag them so they skip
        # the rescue gates, and let downstream NMS dedupe true re-finds.
        # Only the low-confidence band stays flagged rescued.
        strong = [
            replace(h, rescued=False) for h in candidates if h.confidence >= conf
        ]
        candidates = [h for h in candidates if h.confidence < conf]
        if strong:
            logger.info(
                "YOLO rescue pass on %s: %d strong TTA-only hits promoted (conf>=%.2f)",
                image_name or "image",
                len(strong),
                conf,
            )
            hits.extend(strong)
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

    # Eval-derived per-class floors: noisy classes need more confidence.
    overrides = class_conf_overrides(weights)
    if overrides:
        kept = [h for h in hits if h.confidence >= overrides.get(h.class_name, 0.0)]
        if len(kept) != len(hits):
            logger.info(
                "Per-class conf floors dropped %d low-confidence hits on %s",
                len(hits) - len(kept),
                image_name or "image",
            )
        hits = kept
    return hits
