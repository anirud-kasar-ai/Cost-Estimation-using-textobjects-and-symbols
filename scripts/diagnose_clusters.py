"""Where do connected symbol clusters (2-3 touching symbols) get lost?

Uses the cached raw detections (scripts/tune_thresholds.py) to split cluster
misses into their real causes:

  A. model-missed   — no raw box on the member at all (training-data problem)
  B. merged-away    — raw box existed but our NMS/merge deleted it
  C. one-box-over-N — a single box covers several members (model fused them)

Also compares recall on cluster members vs isolated symbols to size the gap.

Usage (from repo root):  python scripts/diagnose_clusters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DZS = ROOT / "drawing-zoom-split"
sys.path.insert(0, str(DZS / "src"))

from pipeline import config  # noqa: E402
from pipeline.nms import nms_yolo_hits  # noqa: E402
from pipeline.yolo_detect import YoloHit  # noqa: E402

MATCH_IOU = 0.50
GAP_FACTOR = 0.40  # boxes closer than this fraction of their size are "touching"

CACHES = {
    "main 52c": ROOT / "eval_data" / "report_misses" / "cache_main_52c.json",
    "roof 9c": ROOT / "eval_data" / "report_misses" / "cache_roof_9c.json",
}


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def touching(a, b) -> bool:
    """Same-class GT boxes whose (padded) boxes intersect."""
    if a[0] != b[0]:
        return False
    ab, bb = a[1:], b[1:]
    pa = GAP_FACTOR * min(ab[2] - ab[0], ab[3] - ab[1])
    pb = GAP_FACTOR * min(bb[2] - bb[0], bb[3] - bb[1])
    return not (
        ab[2] + pa < bb[0] - pb
        or bb[2] + pb < ab[0] - pa
        or ab[3] + pa < bb[1] - pb
        or bb[3] + pb < ab[1] - pa
    )


def to_hits(raw_hits) -> list[YoloHit]:
    return [
        YoloHit(
            class_id=h["class_id"],
            class_name=h["class_name"],
            confidence=h["conf"],
            x1=h["box"][0],
            y1=h["box"][1],
            x2=h["box"][2],
            y2=h["box"][3],
            rescued=h["rescued"],
        )
        for h in raw_hits
    ]


def greedy_match(gts, hits) -> list[bool]:
    """Class-aware IoU>=0.5 matching; returns matched flag per GT."""
    taken = [False] * len(gts)
    for h in sorted(hits, key=lambda x: x.confidence, reverse=True):
        best_i, best_v = -1, MATCH_IOU
        for i, g in enumerate(gts):
            if taken[i] or g[0] != h.class_id:
                continue
            v = iou(g[1:], (h.x1, h.y1, h.x2, h.y2))
            if v >= best_v:
                best_i, best_v = i, v
        if best_i >= 0:
            taken[best_i] = True
    return taken


def main() -> None:
    for name, cache_path in CACHES.items():
        if not cache_path.is_file():
            continue
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

        stats = {
            "cluster_members": 0,
            "singletons": 0,
            "cluster_found_raw": 0,
            "cluster_found_final": 0,
            "singleton_found_raw": 0,
            "singleton_found_final": 0,
            "merged_away": 0,
            "one_box_over_n": 0,
        }

        for entry in cache["images"].values():
            gts = [tuple(g) for g in entry["gt"]]
            if not gts:
                continue
            raw = to_hits(entry["hits"])
            final = nms_yolo_hits(
                raw,
                same_class_iou=config.YOLO_NMS_IOU,
                cross_class_iou=config.YOLO_CROSS_NMS_IOU,
                compact_center_px=config.YOLO_NMS_CENTER_PX,
            )

            # cluster membership per GT
            in_cluster = [False] * len(gts)
            for i in range(len(gts)):
                for j in range(i + 1, len(gts)):
                    if touching(gts[i], gts[j]):
                        in_cluster[i] = in_cluster[j] = True

            raw_matched = greedy_match(gts, raw)
            final_matched = greedy_match(gts, final)

            for i, g in enumerate(gts):
                key = "cluster" if in_cluster[i] else "singleton"
                stats[f"{key}s" if key == "singleton" else "cluster_members"] += 1
                if raw_matched[i]:
                    stats[f"{key}_found_raw"] += 1
                if final_matched[i]:
                    stats[f"{key}_found_final"] += 1
                if in_cluster[i] and raw_matched[i] and not final_matched[i]:
                    stats["merged_away"] += 1

            # one box covering >=2 cluster members (same class)
            for h in raw:
                covered = 0
                for i, g in enumerate(gts):
                    if not in_cluster[i] or g[0] != h.class_id:
                        continue
                    gcx = (g[1] + g[3]) / 2
                    gcy = (g[2] + g[4]) / 2
                    if h.x1 <= gcx <= h.x2 and h.y1 <= gcy <= h.y2:
                        covered += 1
                if covered >= 2:
                    stats["one_box_over_n"] += 1

        cm, sg = stats["cluster_members"], stats["singletons"]
        print(f"\n=== {name} ===")
        print(f"GT symbols: {cm} in clusters (touching), {sg} isolated")
        if cm:
            print(
                f"cluster members : raw recall {stats['cluster_found_raw'] / cm:.1%}"
                f" -> final recall {stats['cluster_found_final'] / cm:.1%}"
                f"   (merged away by our NMS: {stats['merged_away']})"
            )
        if sg:
            print(
                f"isolated symbols: raw recall {stats['singleton_found_raw'] / sg:.1%}"
                f" -> final recall {stats['singleton_found_final'] / sg:.1%}"
            )
        print(f"raw boxes covering 2+ cluster members (model fused): {stats['one_box_over_n']}")


if __name__ == "__main__":
    main()
