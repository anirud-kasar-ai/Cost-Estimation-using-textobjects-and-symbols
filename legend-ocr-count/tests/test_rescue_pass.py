"""Two-pass rescue detection: recover threshold misses, never double-count."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from pipeline import config
from pipeline import yolo_detect
from pipeline.nms import nms_yolo_hits
from pipeline.tile_overlap import TileRoi, merge_overlapping_tile_hits
from pipeline.yolo_detect import YoloHit, detect_symbols


def _hit(x1, y1, x2, y2, conf=0.9, cls=0, name="CAM", img="t.jpg", rescued=False) -> YoloHit:
    return YoloHit(
        class_id=cls,
        class_name=name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_name=img,
        rescued=rescued,
    )


def test_rescue_overlapping_primary_is_dropped():
    """A rescue box on an already-detected symbol never adds a second count."""
    primary = _hit(0, 0, 20, 20, conf=0.80)
    rescue = _hit(1, 1, 21, 21, conf=0.20, rescued=True)
    kept = nms_yolo_hits([primary, rescue], same_class_iou=0.35)
    assert len(kept) == 1
    assert kept[0].confidence == 0.80
    assert kept[0].rescued is False


def test_rescue_on_empty_spot_survives_once():
    """A rescue box far from any primary detection is kept exactly once."""
    primary = _hit(0, 0, 20, 20, conf=0.80)
    rescue = _hit(200, 200, 220, 220, conf=0.20, rescued=True)
    kept = nms_yolo_hits([primary, rescue], same_class_iou=0.35)
    assert len(kept) == 2
    assert sum(1 for h in kept if h.rescued) == 1


def test_rescue_duplicated_across_tiles_counts_once():
    """Same rescued mark seen in two overlapping tiles collapses to one."""
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {
        "plan_zoom_r00_c00.jpg": t0,
        "plan_zoom_r00_c01.jpg": t1,
    }
    # Same physical mark centered at ROI x=90 (tile0 local 80-100, tile1 local 0-20)
    h0 = _hit(80, 40, 100, 60, conf=0.2, img="plan_zoom_r00_c00.jpg", rescued=True)
    h1 = _hit(0, 40, 20, 60, conf=0.2, img="plan_zoom_r00_c01.jpg", rescued=True)
    merged = merge_overlapping_tile_hits([h0, h1], rois)
    merged = nms_yolo_hits(merged, same_class_iou=0.35)
    assert len(merged) == 1
    assert merged[0].rescued is True


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = SimpleNamespace(item=lambda: cls_id)
        self.conf = SimpleNamespace(item=lambda: conf)
        self.xyxy = [SimpleNamespace(tolist=lambda: list(xyxy))]


class _FakeBoxes(list):
    pass


class _FakeModel:
    """Returns preset boxes filtered by conf; records augment per call."""

    names = {0: "CAM"}

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def predict(self, source, conf, iou, verbose, imgsz, augment=False):
        self.calls.append({"conf": conf, "augment": augment})
        kept = _FakeBoxes(b for b in self._boxes if b.conf.item() >= conf)
        return [SimpleNamespace(boxes=kept if kept else None)]


def test_detect_symbols_two_pass_tags_rescued(monkeypatch):
    """Pass 2 keeps only below-threshold boxes, tagged rescued=True."""
    boxes = [
        _FakeBox(0, 0.80, (0, 0, 20, 20)),     # primary detection
        _FakeBox(0, 0.20, (200, 200, 220, 220)),  # only visible at rescue conf
    ]
    model = _FakeModel(boxes)
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda: model)
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", True)
    monkeypatch.setattr(config, "YOLO_RESCUE_CONF", 0.15)
    monkeypatch.setattr(config, "YOLO_RESCUE_TTA", True)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg")

    assert len(model.calls) == 2
    assert model.calls[0] == {"conf": 0.35, "augment": False}
    assert model.calls[1] == {"conf": 0.15, "augment": True}

    # Primary box appears once un-rescued; pass-2 re-find (conf>=0.35) filtered out
    primaries = [h for h in hits if not h.rescued]
    rescues = [h for h in hits if h.rescued]
    assert len(primaries) == 1 and primaries[0].confidence == 0.80
    assert len(rescues) == 1 and rescues[0].confidence == 0.20
    # After NMS nothing is double-counted
    assert len(nms_yolo_hits(hits, same_class_iou=0.35)) == 2


def test_detect_symbols_rescue_disabled_single_pass(monkeypatch):
    boxes = [_FakeBox(0, 0.80, (0, 0, 20, 20))]
    model = _FakeModel(boxes)
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda: model)
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", False)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg")
    assert len(model.calls) == 1
    assert len(hits) == 1 and hits[0].rescued is False
