from pipeline.nms import nms_yolo_hits
from pipeline.tile_overlap import TileRoi, merge_overlapping_tile_hits
from pipeline.yolo_detect import YoloHit


def _hit(x1, y1, x2, y2, conf=0.9, cls=0, name="CAM", img="t.jpg") -> YoloHit:
    return YoloHit(
        class_id=cls,
        class_name=name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_name=img,
    )


def test_same_class_iou_drops_duplicate():
    a = _hit(0, 0, 20, 20, conf=0.9)
    b = _hit(2, 2, 22, 22, conf=0.6)
    kept = nms_yolo_hits([a, b], same_class_iou=0.35)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_far_boxes_same_class_kept():
    a = _hit(0, 0, 20, 20, conf=0.9)
    b = _hit(80, 80, 100, 100, conf=0.8)
    kept = nms_yolo_hits([a, b], same_class_iou=0.35)
    assert len(kept) == 2


def test_cross_class_overlap_keeps_higher_conf():
    a = _hit(0, 0, 30, 10, conf=0.8, cls=0, name="WM5400")
    b = _hit(1, 0, 31, 10, conf=0.5, cls=1, name="WM5500")
    kept = nms_yolo_hits([a, b], cross_class_iou=0.5)
    assert len(kept) == 1
    assert kept[0].class_name == "WM5400"


def test_tile_overlap_same_mark_counts_once():
    # Two 100px tiles with 20px overlap; same physical mark seen by both.
    t0 = TileRoi("plan_zoom_r00_c00.jpg", 0, 0, 0, 0, 100, 100)
    t1 = TileRoi("plan_zoom_r00_c01.jpg", 0, 1, 80, 0, 180, 100)
    rois = {
        "plan_zoom_r00_c00.jpg": t0,
        "plan_zoom_r00_c01.jpg": t1,
    }
    # Same physical mark: center 90 in tile0, 10 in tile1 (90 in ROI)
    h0 = _hit(80, 40, 100, 60, conf=0.9, img="plan_zoom_r00_c00.jpg")
    h1 = _hit(0, 40, 20, 60, conf=0.7, img="plan_zoom_r00_c01.jpg")
    merged = merge_overlapping_tile_hits([h0, h1], rois)
    assert len(merged) == 1
    assert merged[0].confidence == 0.9


def test_tile_overlap_keeps_hit_missed_by_neighbor_tile():
    # A mark in the overlap strip detected ONLY by the second tile must
    # survive (the old exclusive-region rule silently deleted it).
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
