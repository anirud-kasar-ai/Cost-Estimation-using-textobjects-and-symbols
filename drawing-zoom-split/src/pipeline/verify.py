"""Second-stage verification of low-confidence YOLO detections.

A small binary classifier (true_symbol vs false_symbol), trained on crops of
the detector's own correct and incorrect detections, re-judges every kept
detection below VERIFY_MAX_CONF. Crops are taken from the wing image with
context margin — the same preparation used at training time
(scripts/build_verifier_dataset.py at the repo root).

Confident detections are never second-guessed, and the whole stage is a no-op
when the weights file is missing, so the pipeline runs unchanged without it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import config
from pipeline.yolo_detect import YoloHit

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any = None
_model_failed = False


def verifier_available() -> bool:
    if not config.VERIFY_ENABLED or _model_failed:
        return False
    if not Path(config.VERIFY_MODEL_PATH).is_file():
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    global _model, _model_failed
    with _lock:
        if _model is not None:
            return _model
        try:
            from ultralytics import YOLO

            logger.info("Loading detection verifier from %s", config.VERIFY_MODEL_PATH)
            _model = YOLO(str(config.VERIFY_MODEL_PATH))
        except Exception:
            logger.exception("Verifier failed to load — verification disabled")
            _model_failed = True
            return None
        return _model


def _crop(image: Image.Image, hit: YoloHit) -> Image.Image:
    w, h = image.size
    bw, bh = hit.x2 - hit.x1, hit.y2 - hit.y1
    mx, my = config.VERIFY_CROP_MARGIN * bw, config.VERIFY_CROP_MARGIN * bh
    x1, y1 = hit.x1 - mx, hit.y1 - my
    x2, y2 = hit.x2 + mx, hit.y2 + my
    if x2 - x1 < 32:
        cx = (x1 + x2) / 2
        x1, x2 = cx - 16, cx + 16
    if y2 - y1 < 32:
        cy = (y1 + y2) / 2
        y1, y2 = cy - 16, cy + 16
    return image.crop(
        (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
    )


def p_false(image: Image.Image, hit: YoloHit) -> float | None:
    """P(false_symbol) for one detection, or None when unavailable."""
    model = _get_model()
    if model is None:
        return None
    try:
        results = model.predict(
            source=_crop(image, hit),
            imgsz=int(config.VERIFY_IMGSZ),
            verbose=False,
        )
        names = model.names or {}
        false_idx = next(
            (int(i) for i, n in names.items() if str(n) == "false_symbol"), None
        )
        if false_idx is None or not results:
            return None
        return float(results[0].probs.data[false_idx])
    except Exception:
        logger.exception("Verifier prediction failed")
        return None


def verify_hits(
    hits: list[YoloHit],
    wing_image: Image.Image,
) -> tuple[list[YoloHit], list[YoloHit]]:
    """Split hits into (kept, dropped-as-false).

    Only hits below VERIFY_MAX_CONF are checked; a checked hit is dropped when
    P(false_symbol) >= VERIFY_DROP_PROB. Boxes must be in wing-image pixels.
    """
    if not hits or not verifier_available():
        return hits, []

    kept: list[YoloHit] = []
    dropped: list[YoloHit] = []
    for hit in hits:
        if hit.confidence >= float(config.VERIFY_MAX_CONF):
            kept.append(hit)
            continue
        prob = p_false(wing_image, hit)
        if prob is not None and prob >= float(config.VERIFY_DROP_PROB):
            logger.debug(
                "Verifier dropped %s conf=%.2f (P(false)=%.2f)",
                hit.class_name,
                hit.confidence,
                prob,
            )
            dropped.append(hit)
        else:
            kept.append(hit)
    return kept, dropped
