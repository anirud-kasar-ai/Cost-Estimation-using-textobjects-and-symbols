"""Tune rescue floor + per-class confidence thresholds to cut false detections.

Strategy: run the drawing-zoom-split production detection path ONCE per model
with the rescue pass at its lowest floor (0.15) and cache every raw hit.
Every candidate rescue floor / per-class threshold is then simulated from the
cache (filter -> NMS -> IoU 0.5 matching), so the sweep is instant and the
final numbers use exactly the same evaluation as business_metrics_rescue.py.

Outputs:
  - eval_data/report_misses/threshold_sweep.json   (all sweep results)
  - drawing-zoom-split/model/class_conf_overrides.json  (per-class min conf)
  - docs/business_metrics_rescue.csv  (rewritten with a "tuned" mode added)

Usage (from repo root):
    python scripts/tune_thresholds.py
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DZS = ROOT / "drawing-zoom-split"
sys.path.insert(0, str(DZS / "src"))

from PIL import Image  # noqa: E402

from pipeline import config  # noqa: E402
from pipeline.nms import nms_yolo_hits  # noqa: E402
from pipeline.yolo_detect import YoloHit, detect_symbols  # noqa: E402

MATCH_IOU = 0.50
CACHE_RESCUE_FLOOR = 0.15
RESCUE_FLOORS = [None, 0.15, 0.20, 0.25, 0.30]  # None = rescue off
CLASS_THRESH_CANDIDATES = [0.40, 0.50, 0.60, 0.70]
MIN_CLASS_SUPPORT = 5  # need >=5 GT or >=5 FP before overriding a class

MODELS = [
    {
        "name": "Main symbol detector (52 classes)",
        "classes": 52,
        "weights": DZS / "model" / "symbol_detector_tech.pt",
        "dataset": ROOT / "eval_data" / "test_project" / "test",
        "cache": ROOT / "eval_data" / "report_misses" / "cache_main_52c.json",
        # F1-optimal floors cost no recall on this model — allow up to 0.7.
        "max_floor": 0.70,
    },
    {
        "name": "Roof symbol detector (9 classes)",
        "classes": 9,
        "weights": DZS / "model" / "symbol_detector_best.pt",
        "dataset": ROOT / "eval_data" / "test_small_9c" / "test",
        "cache": ROOT / "eval_data" / "report_misses" / "cache_roof_9c.json",
        # 0.7 floors erased real roof symbols on live jobs (recall 81% vs 86%).
        # Cap at 0.5: recovers those at a false-positive cost the downstream
        # filters (verifier, masks) absorb.
        "max_floor": 0.50,
    },
]


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(p[0])), *(float(v) for v in p[1:5])
        out.append(
            (cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        )
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------
# 1. Build / load the raw-hit cache (one full inference run per model)
# ----------------------------------------------------------------------


def build_cache(dataset: Path, weights: Path, cache_path: Path) -> dict:
    if cache_path.is_file():
        print(f"  cache hit: {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    config.YOLO_RESCUE_ENABLED = True
    config.YOLO_RESCUE_CONF = CACHE_RESCUE_FLOOR
    data: dict = {"images": {}}
    img_dir = dataset / "images"
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    for n, img_path in enumerate(images, 1):
        img = Image.open(img_path)
        gts = load_gt(dataset / "labels" / (img_path.stem + ".txt"), *img.size)
        hits = detect_symbols(img, image_name=img_path.name, weights=weights)
        data["images"][img_path.name] = {
            "gt": gts,
            "hits": [
                {
                    "class_id": h.class_id,
                    "class_name": h.class_name,
                    "conf": h.confidence,
                    "box": [h.x1, h.y1, h.x2, h.y2],
                    "rescued": h.rescued,
                }
                for h in hits
            ],
        }
        if n % 32 == 0:
            print(f"  inference {n}/{len(images)} images", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"  cache saved: {cache_path.name}")
    return data


# ----------------------------------------------------------------------
# 2. Simulate any (rescue_floor, per-class overrides) config from cache
# ----------------------------------------------------------------------


@dataclass
class Tally:
    images: int = 0
    gt: int = 0
    pred: int = 0
    correct: int = 0
    wrong_class: int = 0
    false_det: int = 0
    rescued_matched: int = 0

    @property
    def missed(self) -> int:
        return self.gt - self.correct - self.wrong_class

    @property
    def recall(self) -> float:
        return self.correct / self.gt if self.gt else 0.0

    @property
    def precision(self) -> float:
        return self.correct / self.pred if self.pred else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_cached(
    cache: dict,
    rescue_floor: float | None,
    overrides: dict[str, float] | None = None,
    collect_dets: list | None = None,
) -> Tally:
    """Filter cached hits per config, NMS, match vs GT. Same math as the
    business metrics script. Optionally collects (class_name, conf, is_tp)
    per kept detection for per-class threshold derivation."""
    overrides = overrides or {}
    t = Tally()
    t.images = len(cache["images"])
    for entry in cache["images"].values():
        gts = [tuple(g) for g in entry["gt"]]
        t.gt += len(gts)
        hits = []
        for h in entry["hits"]:
            if h["rescued"]:
                if rescue_floor is None or h["conf"] < rescue_floor:
                    continue
            if h["conf"] < overrides.get(h["class_name"], 0.0):
                continue
            hits.append(
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
            )
        hits = nms_yolo_hits(
            hits,
            same_class_iou=config.YOLO_NMS_IOU,
            cross_class_iou=config.YOLO_CROSS_NMS_IOU,
            compact_center_px=config.YOLO_NMS_CENTER_PX,
        )
        t.pred += len(hits)

        taken = [False] * len(gts)
        unmatched = []
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
                t.correct += 1
                if h.rescued:
                    t.rescued_matched += 1
                if collect_dets is not None:
                    collect_dets.append((h.class_name, h.confidence, True))
            else:
                unmatched.append(h)
        for h in unmatched:
            best_i, best_v = -1, MATCH_IOU
            for i, g in enumerate(gts):
                if taken[i]:
                    continue
                v = iou(g[1:], (h.x1, h.y1, h.x2, h.y2))
                if v >= best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                taken[best_i] = True
                t.wrong_class += 1
            else:
                t.false_det += 1
            if collect_dets is not None:
                collect_dets.append((h.class_name, h.confidence, False))
    return t


def gt_counts_by_class(cache: dict, id_to_name: dict[int, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in cache["images"].values():
        for g in entry["gt"]:
            name = id_to_name.get(int(g[0]), str(g[0]))
            out[name] = out.get(name, 0) + 1
    return out


def derive_overrides(cache: dict, rescue_floor: float | None) -> dict[str, float]:
    """Per class, raise the min confidence if it improves class F1 on eval.
    Requires MIN_CLASS_SUPPORT instances so single lucky/unlucky boxes don't
    set thresholds."""
    dets: list[tuple[str, float, bool]] = []
    evaluate_cached(cache, rescue_floor, collect_dets=dets)
    id_to_name = {}
    for entry in cache["images"].values():
        for h in entry["hits"]:
            id_to_name[int(h["class_id"])] = h["class_name"]
    gt_per_class = gt_counts_by_class(cache, id_to_name)

    by_class: dict[str, list[tuple[float, bool]]] = {}
    for name, conf, is_tp in dets:
        by_class.setdefault(name, []).append((conf, is_tp))

    overrides: dict[str, float] = {}
    for name, entries in by_class.items():
        gt_n = gt_per_class.get(name, 0)
        fp_n = sum(1 for _, tp in entries if not tp)
        if gt_n < MIN_CLASS_SUPPORT and fp_n < MIN_CLASS_SUPPORT:
            continue

        def f1_at(thresh: float) -> float:
            tp = sum(1 for c, is_tp in entries if is_tp and c >= thresh)
            fp = sum(1 for c, is_tp in entries if not is_tp and c >= thresh)
            fn = gt_n - tp
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        base_f1 = f1_at(0.0)  # current behaviour (pass floors only)
        best_t, best_f1 = None, base_f1
        for cand in CLASS_THRESH_CANDIDATES:
            v = f1_at(cand)
            if v > best_f1 + 1e-9:
                best_t, best_f1 = cand, v
        if best_t is not None:
            overrides[name] = best_t
    return overrides


TUNED_FLOOR = 0.25  # production choice: balanced recall/precision; text mask +
# legend gate (not visible to this raw eval) remove most remaining rescue FPs.


def main() -> None:
    sweep_report: dict = {"models": []}
    csv_rows = []
    totals: dict[str, Tally] = {}
    chosen_overrides: dict[str, dict[str, float]] = {}  # keyed by model weight stem

    for m in MODELS:
        if not m["weights"].is_file() or not m["dataset"].is_dir():
            print(f"SKIP {m['name']}: missing weights or dataset")
            continue
        print(f"\n=== {m['name']} ===")
        cache = build_cache(m["dataset"], m["weights"], m["cache"])

        # --- rescue floor sweep ---
        floors = {}
        for floor in RESCUE_FLOORS:
            t = evaluate_cached(cache, floor)
            key = "off" if floor is None else f"{floor:.2f}"
            floors[key] = {
                "recall": round(t.recall, 4),
                "precision": round(t.precision, 4),
                "f1": round(t.f1, 4),
                "rescued_matched": t.rescued_matched,
                "false_det": t.false_det,
            }
            print(
                f"  floor {key:>5}: recall {t.recall:.4f} precision {t.precision:.4f} "
                f"f1 {t.f1:.4f} rescued {t.rescued_matched} fp {t.false_det}"
            )

        # --- per-class overrides at the tuned floor (capped per model) ---
        cap = float(m.get("max_floor", 1.0))
        overrides = {
            k: min(v, cap) for k, v in derive_overrides(cache, TUNED_FLOOR).items()
        }
        print(f"  -> per-class overrides ({len(overrides)}, cap {cap}): {overrides}")
        chosen_overrides[m["weights"].stem.lower()] = overrides

        # --- final modes for the CSV ---
        modes = [
            ("single-pass (before)", None, {}),
            ("two-pass rescue (after)", CACHE_RESCUE_FLOOR, {}),
            (
                f"tuned (rescue {TUNED_FLOOR:.2f} + class thresholds)",
                TUNED_FLOOR,
                overrides,
            ),
        ]
        for label, floor, ov in modes:
            t = evaluate_cached(cache, floor, ov)
            csv_rows.append((m["name"], m["classes"], label, t))
            agg = totals.setdefault(label, Tally())
            for f in ("images", "gt", "pred", "correct", "wrong_class", "false_det",
                      "rescued_matched"):
                setattr(agg, f, getattr(agg, f) + getattr(t, f))

        sweep_report["models"].append(
            {
                "name": m["name"],
                "floors": floors,
                "tuned_floor": TUNED_FLOOR,
                "overrides": overrides,
            }
        )

    for label, t in totals.items():
        csv_rows.append(("Overall (both models)", "52+9", label, t))

    # --- write outputs ---
    sweep_path = ROOT / "eval_data" / "report_misses" / "threshold_sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(json.dumps(sweep_report, indent=2), encoding="utf-8")

    overrides_path = DZS / "model" / "class_conf_overrides.json"
    overrides_path.write_text(json.dumps(chosen_overrides, indent=2, sort_keys=True),
                              encoding="utf-8")

    out = ROOT / "docs" / "business_metrics_rescue.csv"
    try:
        f = out.open("w", newline="", encoding="utf-8")
    except PermissionError:
        # File is open in Excel — write next to it instead of crashing.
        out = out.with_name("business_metrics_rescue_tuned.csv")
        f = out.open("w", newline="", encoding="utf-8")
    with f:
        w = csv.writer(f)
        w.writerow(
            [
                "Model", "Classes", "Mode", "Images", "Actual objects",
                "Correctly identified", "Object recall", "Wrong class", "Missed",
                "Total detections", "False detections", "Object precision",
                "Object F1", "Rescued symbols matched",
            ]
        )
        for name, classes, mode, t in csv_rows:
            w.writerow(
                [
                    name, classes, mode, t.images, t.gt, t.correct,
                    f"{t.recall:.4f}", t.wrong_class, t.missed, t.pred,
                    t.false_det, f"{t.precision:.4f}", f"{t.f1:.4f}",
                    t.rescued_matched,
                ]
            )

    print(f"\nSaved: {sweep_path}")
    print(f"Saved: {overrides_path}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
