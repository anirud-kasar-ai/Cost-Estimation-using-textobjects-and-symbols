"""Two-pass rescue detection: recover threshold misses, never double-count."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline import config  # noqa: E402
from pipeline import yolo_detect  # noqa: E402
from pipeline.nms import nms_yolo_hits  # noqa: E402
from pipeline.tile_overlap import TileRoi, merge_overlapping_tile_hits  # noqa: E402
from pipeline.yolo_detect import YoloHit, detect_symbols  # noqa: E402


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


def test_hit_in_overlap_strip_missed_by_neighbor_survives():
    """A mark in the overlap strip detected ONLY by the second tile must
    survive (the old exclusive-region rule silently deleted it)."""
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {
        "plan_zoom_r00_c00.jpg": t0,
        "plan_zoom_r00_c01.jpg": t1,
    }
    only_t1 = _hit(0, 40, 20, 60, conf=0.95, img="plan_zoom_r00_c01.jpg")
    merged = merge_overlapping_tile_hits([only_t1], rois)
    assert len(merged) == 1
    assert merged[0].x1 == 80  # mapped into ROI space
    assert merged[0].confidence == 0.95


def test_touching_small_symbols_both_counted():
    """Two connected 14px symbols: centers only 14px apart. The old 16px
    center floor merged them into one count — they are two real symbols."""
    a = _hit(100, 100, 114, 114, conf=0.90)
    b = _hit(114, 100, 128, 114, conf=0.85)  # touching, centers 14px apart
    kept = nms_yolo_hits([a, b])
    assert len(kept) == 2


def test_duplicate_boxes_on_one_glyph_still_merged():
    """Same glyph boxed twice (near-identical centers) keeps collapsing."""
    a = _hit(100, 100, 130, 130, conf=0.90)
    b = _hit(102, 101, 132, 131, conf=0.60)  # IoU ~0.8, centers ~2px apart
    kept = nms_yolo_hits([a, b])
    assert len(kept) == 1
    assert kept[0].confidence == 0.90


def test_mid_iou_duplicate_caught_by_center_rule():
    """Two sloppy boxes on one glyph with shared center are one symbol."""
    a = _hit(100, 100, 130, 130, conf=0.90)
    b = _hit(96, 106, 134, 126, conf=0.55)  # same center, offset shape
    kept = nms_yolo_hits([a, b])
    assert len(kept) == 1


def test_elongated_overlaps_still_merged_by_iou():
    """Raceway-like elongated boxes keep merging on IoU alone."""
    a = _hit(100, 100, 300, 120, conf=0.90)  # 200x20 — elongated
    b = _hit(140, 100, 340, 120, conf=0.70)  # IoU 0.67 >= 0.35 → merged
    kept = nms_yolo_hits([a, b])
    assert len(kept) == 1


def test_unconfirmed_rescue_in_overlap_zone_is_dropped():
    """A rescue-band hit in territory two tiles can see, seen by only one of
    them, is noise — the consensus rule removes it."""
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {t0.file: t0, t1.file: t1}
    # Center at ROI x=90 → inside both tiles; only tile 1 reports it, rescued.
    lonely = _hit(5, 40, 15, 60, conf=0.28, img="plan_zoom_r00_c01.jpg", rescued=True)
    merged = merge_overlapping_tile_hits([lonely], rois)
    assert merged == []


def test_confirmed_rescue_in_overlap_zone_survives():
    """Same spot, but the neighbouring tile also sees it → kept (once)."""
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {t0.file: t0, t1.file: t1}
    rescue_t1 = _hit(5, 40, 15, 60, conf=0.28, img="plan_zoom_r00_c01.jpg", rescued=True)
    confirm_t0 = _hit(85, 40, 95, 60, conf=0.30, img="plan_zoom_r00_c00.jpg", rescued=True)
    merged = merge_overlapping_tile_hits([rescue_t1, confirm_t0], rois)
    assert len(merged) == 1
    assert merged[0].rescued is True


def test_rescue_in_single_tile_territory_kept_without_consensus():
    """Where only one tile covers the spot, consensus is impossible — keep."""
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {t0.file: t0, t1.file: t1}
    # Center at ROI x=20 → covered by tile 0 only.
    interior = _hit(15, 40, 25, 60, conf=0.28, img="plan_zoom_r00_c00.jpg", rescued=True)
    merged = merge_overlapping_tile_hits([interior], rois)
    assert len(merged) == 1


def test_primary_hits_never_need_consensus():
    """Confident detections are exempt from the consensus rule."""
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {t0.file: t0, t1.file: t1}
    confident = _hit(5, 40, 15, 60, conf=0.95, img="plan_zoom_r00_c01.jpg")
    merged = merge_overlapping_tile_hits([confident], rois)
    assert len(merged) == 1


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
        _FakeBox(0, 0.80, (0, 0, 20, 20)),        # primary detection
        _FakeBox(0, 0.20, (200, 200, 220, 220)),  # only visible at rescue conf
    ]
    model = _FakeModel(boxes)
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda weights=None: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda weights=None: model)
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", True)
    monkeypatch.setattr(config, "YOLO_RESCUE_CONF", 0.15)
    monkeypatch.setattr(config, "YOLO_RESCUE_TTA", True)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg", weights=Path("fake.pt"))

    assert len(model.calls) == 2
    assert model.calls[0] == {"conf": 0.35, "augment": False}
    assert model.calls[1] == {"conf": 0.15, "augment": True}

    # Primary box appears un-rescued (once from pass 1, once as a promoted
    # pass-2 re-find on the same spot); the low-conf box stays rescued.
    primaries = [h for h in hits if not h.rescued]
    rescues = [h for h in hits if h.rescued]
    assert all(h.confidence == 0.80 for h in primaries)
    assert len(rescues) == 1 and rescues[0].confidence == 0.20
    # After NMS the re-find collapses onto the pass-1 box: nothing double-counted
    assert len(nms_yolo_hits(hits, same_class_iou=0.35)) == 2


def test_detect_symbols_strong_tta_only_hit_promoted(monkeypatch):
    """TTA sometimes sees a symbol the plain pass misses outright. A strong
    (conf >= primary threshold) TTA-only box must survive as a trusted,
    non-rescued detection instead of being discarded as a 're-find'."""

    class _TtaOnlyModel(_FakeModel):
        def predict(self, source, conf, iou, verbose, imgsz, augment=False):
            self.calls.append({"conf": conf, "augment": augment})
            if not augment:  # plain pass sees nothing
                return [SimpleNamespace(boxes=None)]
            kept = _FakeBoxes(b for b in self._boxes if b.conf.item() >= conf)
            return [SimpleNamespace(boxes=kept if kept else None)]

    model = _TtaOnlyModel([_FakeBox(0, 0.75, (100, 100, 120, 120))])
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda weights=None: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda weights=None: model)
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", True)
    monkeypatch.setattr(config, "YOLO_RESCUE_CONF", 0.15)
    monkeypatch.setattr(config, "YOLO_RESCUE_TTA", True)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg", weights=Path("fake.pt"))

    assert len(hits) == 1
    assert hits[0].confidence == 0.75
    assert hits[0].rescued is False  # exempt from legend gate and consensus


def test_detect_symbols_rescue_disabled_single_pass(monkeypatch):
    boxes = [_FakeBox(0, 0.80, (0, 0, 20, 20))]
    model = _FakeModel(boxes)
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda weights=None: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda weights=None: model)
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", False)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg")
    assert len(model.calls) == 1
    assert len(hits) == 1 and hits[0].rescued is False


def test_per_class_conf_floor_drops_noisy_class(monkeypatch):
    """Eval-derived per-class floors filter low-conf hits of noisy classes only."""
    boxes = [
        _FakeBox(0, 0.40, (0, 0, 20, 20)),        # CAM at 0.40 — below its 0.50 floor
        _FakeBox(0, 0.80, (50, 50, 70, 70)),      # CAM at 0.80 — passes
    ]
    model = _FakeModel(boxes)
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda weights=None: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda weights=None: model)
    monkeypatch.setattr(yolo_detect, "class_conf_overrides", lambda weights=None: {"CAM": 0.50})
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", False)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg")
    assert len(hits) == 1
    assert hits[0].confidence == 0.80


def test_legend_gate_drops_only_unmatched_rescues():
    """Rescued hits without a legend match are dropped; everything else stays."""
    from pipeline.count_stage import drop_unmatched_rescues

    def with_legend(h: YoloHit, number: int | None) -> YoloHit:
        return YoloHit(
            class_id=h.class_id,
            class_name=h.class_name,
            confidence=h.confidence,
            x1=h.x1,
            y1=h.y1,
            x2=h.x2,
            y2=h.y2,
            image_name=h.image_name,
            legend_number=number,
            rescued=h.rescued,
        )

    primary_matched = with_legend(_hit(0, 0, 10, 10, conf=0.8), 1)
    primary_unmatched = with_legend(_hit(20, 0, 30, 10, conf=0.8), None)
    rescued_matched = with_legend(_hit(40, 0, 50, 10, conf=0.2, rescued=True), 2)
    rescued_unmatched = with_legend(_hit(60, 0, 70, 10, conf=0.2, rescued=True), None)

    kept, dropped = drop_unmatched_rescues(
        [primary_matched, primary_unmatched, rescued_matched, rescued_unmatched]
    )
    assert dropped == 1
    assert rescued_unmatched not in kept
    assert primary_matched in kept
    assert primary_unmatched in kept  # confident hits stand alone
    assert rescued_matched in kept


def test_ignored_classes_filtered_in_rescue_pass(monkeypatch):
    """Test-only classes stay excluded even in the rescue pass."""
    model = _FakeModel([_FakeBox(1, 0.20, (0, 0, 10, 10))])
    model.names = {0: "CAM", 1: "object"}
    monkeypatch.setattr(yolo_detect, "yolo_available", lambda weights=None: True)
    monkeypatch.setattr(yolo_detect, "get_model", lambda weights=None: model)
    monkeypatch.setattr(config, "YOLO_CONF", 0.35)
    monkeypatch.setattr(config, "YOLO_RESCUE_ENABLED", True)
    monkeypatch.setattr(config, "YOLO_RESCUE_CONF", 0.15)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    hits = detect_symbols(img, image_name="t.jpg")
    assert hits == []
