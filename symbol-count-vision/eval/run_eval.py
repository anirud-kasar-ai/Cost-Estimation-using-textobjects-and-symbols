"""Run the current detect_symbols_raw pipeline on eval crops and score it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.score import load_gt, score
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo
from pipeline.symbol_count import detect_symbols_raw


def eval_legend() -> SymbolTableInfo:
    return SymbolTableInfo(
        entries=[
            SymbolEntry(symbol="#", description="DATA PERMANENT LINK"),
            SymbolEntry(symbol="AP", description="INTERIOR WIRELESS ACCESS POINT"),
            SymbolEntry(symbol="", description="NETWORK CAMERA"),
            SymbolEntry(symbol="", description="DATA POLE", part_number="30TC-4**V"),
            SymbolEntry(symbol="J", description="JUNCTION BOX"),
            SymbolEntry(symbol="", description="J-HOOK (SINGLE/STACKED)"),
            SymbolEntry(
                symbol="", description="SURFACE RACEWAY", part_number="WM5400"
            ),
        ]
    )


def run(gt_dir: Path, crops_dir: Path) -> tuple[dict, dict]:
    legend = eval_legend()
    records = load_gt(gt_dir)
    predictions: dict[str, list[dict]] = {}
    for rec in records:
        img_path = ROOT / "eval" / rec["image"]
        if not img_path.is_file():
            img_path = crops_dir / Path(rec["image"]).name
        image = Image.open(img_path).convert("RGB")
        dets, _notes, _meta = detect_symbols_raw(
            image=image,
            legend=legend,
            glyph_dir=None,
            nms_iou=0.3,
            apply_local_nms=True,
            use_vision_ocr=False,
            exclude_top_pct=0.0,
        )
        predictions[rec["id"]] = [
            {
                "symbol": d.symbol,
                "source": d.source,
                "classify_source": d.classify_source,
                "qty": d.qty,
                "box": {
                    "x1": d.box.x1,
                    "y1": d.box.y1,
                    "x2": d.box.x2,
                    "y2": d.box.y2,
                },
            }
            for d in dets
        ]
    result = score(records, predictions, iou_thresh=0.5)
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="after", help="before|after label for output")
    args = parser.parse_args()
    eval_root = ROOT / "eval"
    result, predictions = run(eval_root / "gt", eval_root / "crops")
    (eval_root / f"predictions_{args.tag}.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    (eval_root / f"scores_{args.tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
