"""Per-key precision / recall / F1 + confusion table (IoU >= 0.5)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix1 = max(float(a["x1"]), float(b["x1"]))
    iy1 = max(float(a["y1"]), float(b["y1"]))
    ix2 = min(float(a["x2"]), float(b["x2"]))
    iy2 = min(float(a["y2"]), float(b["y2"]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(
        0.0, float(a["y2"]) - float(a["y1"])
    )
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(
        0.0, float(b["y2"]) - float(b["y1"])
    )
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _norm_key(key: str) -> str:
    k = (key or "").strip()
    if k.upper().startswith("DATA POLE"):
        return "DATA POLE"
    if "CAMERA" in k.upper() and "MOUNT" not in k.upper():
        return "NETWORK CAMERA"
    if "HOOK" in k.upper():
        return "J-HOOK"
    if "RACEWAY" in k.upper() or k.upper().startswith("WM"):
        return "SURFACE RACEWAY"
    return k


def load_gt(gt_dir: Path) -> list[dict[str, Any]]:
    index = gt_dir / "index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(gt_dir.glob("*.json"))]


def match_crop(
    gt_instances: list[dict[str, Any]],
    pred_dets: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Return (matches as (gt_key, pred_key), unmatched_gt keys, unmatched_pred keys)."""
    gt = list(gt_instances)
    pred = list(pred_dets)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[str, str]] = []
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = box_iou(g["box"], p["box"])
            if iou >= iou_thresh:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    for _iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((_norm_key(gt[gi]["key"]), _norm_key(pred[pi]["symbol"])))
    fn = [_norm_key(gt[i]["key"]) for i in range(len(gt)) if i not in used_g]
    fp = [_norm_key(pred[i]["symbol"]) for i in range(len(pred)) if i not in used_p]
    return matches, fn, fp


def score(
    gt_records: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = defaultdict(int)  # "gt -> pred"

    for rec in gt_records:
        crop_id = rec["id"]
        preds = predictions.get(crop_id) or predictions.get(rec.get("image", "")) or []
        matches, miss_gt, extra_pred = match_crop(
            rec["instances"], preds, iou_thresh=iou_thresh
        )
        for gk, pk in matches:
            if gk == pk:
                tp[gk] += 1
            else:
                fp[pk] += 1
                fn[gk] += 1
                confusion[f"{gk} predicted as {pk}"] += 1
        for k in miss_gt:
            fn[k] += 1
        for k in extra_pred:
            fp[k] += 1

    keys = sorted(set(tp) | set(fp) | set(fn))
    per_key = {}
    for k in keys:
        t, fpos, fneg = tp[k], fp[k], fn[k]
        prec = t / (t + fpos) if (t + fpos) else 0.0
        rec = t / (t + fneg) if (t + fneg) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_key[k] = {
            "tp": t,
            "fp": fpos,
            "fn": fneg,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    return {
        "iou_threshold": iou_thresh,
        "per_key": per_key,
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
        "totals": {
            "tp": sum(tp.values()),
            "fp": sum(fp.values()),
            "fn": sum(fn.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True, help="JSON {crop_id: [dets]}")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    gt = load_gt(args.gt)
    pred = json.loads(args.pred.read_text(encoding="utf-8"))
    result = score(gt, pred, iou_thresh=args.iou)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
