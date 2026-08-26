"""Unit tests for wing symbol counting (NMS, coords, report pivot)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo  # noqa: E402
from pipeline.symbol_count import (
    BoundingBox,
    SymbolDetection,
    _legend_label_map,
    box_iou,
    discover_wing_instances,
    nms_by_class,
    translate_norm_box_to_roi,
)
from pipeline.symbol_count_cv import aggregate_results_by_wing_name
from pipeline.symbol_count_report import build_pivot_rows, write_symbol_count_csv, write_symbol_count_json


def test_box_iou_identical():
    box = BoundingBox(10, 10, 30, 30)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_nms_removes_overlap_duplicates():
    detections = [
        SymbolDetection("R", BoundingBox(100, 100, 120, 120), 0.95),
        SymbolDetection("R", BoundingBox(102, 102, 122, 122), 0.80),
        SymbolDetection("AP", BoundingBox(300, 300, 320, 320), 0.90),
    ]
    kept = nms_by_class(detections, 0.5)
    assert len(kept) == 2
    symbols = {det.symbol for det in kept}
    assert symbols == {"R", "AP"}
    r_det = next(det for det in kept if det.symbol == "R")
    assert r_det.score == pytest.approx(0.95)


def test_translate_norm_box_to_roi():
    roi_box = translate_norm_box_to_roi(
        [0.1, 0.2, 0.3, 0.4],
        tile_bbox_roi={"x1": 100, "y1": 200, "x2": 1252, "y2": 1352},
        image_width=1152,
        image_height=1152,
    )
    assert roi_box.x1 == pytest.approx(100 + 0.1 * 1152)
    assert roi_box.y1 == pytest.approx(200 + 0.2 * 1152)
    assert roi_box.x2 == pytest.approx(100 + 0.3 * 1152)
    assert roi_box.y2 == pytest.approx(200 + 0.4 * 1152)


def test_build_pivot_rows_shape():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(description="DATA RACK", symbol="R", mfg_model="ACME", part_number="X1"),
            SymbolEntry(description="ACCESS POINT", symbol="AP"),
        ]
    )
    instance_results = [
        {
            "column_key": "B-WING-EAST (page_020)",
            "counts": {"R": 2, "AP": 1},
        },
        {
            "column_key": "C-WING-EAST (page_020)",
            "counts": {"R": 1},
        },
    ]
    columns, rows = build_pivot_rows(legend, instance_results)
    assert columns == [
        "B-WING-EAST (page_020)",
        "C-WING-EAST (page_020)",
    ]
    assert len(rows) == 2
    assert rows[0]["symbol"] == "R"
    assert rows[0]["wing_counts"]["B-WING-EAST (page_020)"] == 2
    assert rows[0]["wing_counts"]["C-WING-EAST (page_020)"] == 1
    assert rows[0]["total"] == 3
    assert rows[1]["total"] == 1


def test_write_symbol_count_csv(tmp_path: Path):
    legend = SymbolTableInfo(
        entries=[SymbolEntry(description="DATA RACK", symbol="R")],
    )
    instance_results = [
        {"column_key": "B-WING-EAST (page_020)", "counts": {"R": 3}},
    ]
    csv_path = tmp_path / "report.csv"
    write_symbol_count_csv(
        legend=legend,
        instance_results=instance_results,
        output_path=csv_path,
    )
    with csv_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Symbol"] == "R"
    assert rows[0]["B-WING-EAST (page_020)"] == "3"
    assert rows[0]["Total"] == "3"


def test_discover_wing_instances(tmp_path: Path):
    meta_dir = tmp_path / "05_metadata"
    meta_dir.mkdir(parents=True)
    zooms_dir = tmp_path / "04_wings" / "B-WING-EAST" / "zooms" / "page_020"
    zooms_dir.mkdir(parents=True)
    manifest_path = zooms_dir / "zooms_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "wing_name": "B-WING-EAST",
                "tile_size": 1152,
                "overlap_pct": 0.2,
                "count": 1,
                "tiles": [
                    {
                        "r": 0,
                        "c": 0,
                        "bbox_roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                        "file": "plan_zoom_r00_c00.jpg",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "page_020.json").write_text(
        json.dumps(
            {
                "page_key": "page_020",
                "wings": [
                    {
                        "name": "B-WING-EAST",
                        "image": "04_wings/B-WING-EAST/page_020.jpg",
                        "zooms_manifest": "04_wings/B-WING-EAST/zooms/page_020/zooms_manifest.json",
                        "zooms_full_image": "04_wings/B-WING-EAST/zooms/page_020/full_wing.jpg",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    instances = discover_wing_instances(tmp_path)
    assert len(instances) == 1
    assert instances[0].column_key == "B-WING-EAST (page_020)"
    assert instances[0].manifest_path == manifest_path


def test_symbol_table_json_roundtrip():
    original = SymbolTableInfo(
        entries=[
            SymbolEntry(
                description="DATA RACK",
                symbol="R",
                mfg_model="ACME",
                part_number="WM5500",
                symbol_image_png=b"fake",
            )
        ],
        legend_pages=[2],
        total_pages=10,
        notes=["test"],
    )
    restored = SymbolTableInfo.from_json_dict(original.to_json_dict())
    assert restored.entry_count == 1
    assert restored.entries[0].symbol == "R"
    assert restored.entries[0].mfg_model == "ACME"
    assert restored.entries[0].symbol_image_png is None


def test_write_symbol_count_json(tmp_path: Path):
    legend = SymbolTableInfo(
        entries=[SymbolEntry(description="DATA RACK", symbol="R")],
    )
    instance_results = [
        {"column_key": "B-WING-EAST (page_020)", "counts": {"R": 2}},
    ]
    json_path = tmp_path / "report.json"
    write_symbol_count_json(
        legend=legend,
        instance_results=instance_results,
        output_path=json_path,
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["columns"] == ["B-WING-EAST (page_020)"]
    assert payload["rows"][0]["symbol"] == "R"
    assert payload["rows"][0]["counts"]["B-WING-EAST (page_020)"] == 2
    assert payload["rows"][0]["total"] == 2


def test_legend_label_map_uses_catalog_id_with_part():
    legend = SymbolTableInfo(
        entries=[
            SymbolEntry(description="ACCESS POINT", symbol="AP"),
            SymbolEntry(
                description="SURFACE RACEWAY",
                symbol=None,
                mfg_model="WIREMOLD",
                part_number="WM2300",
            ),
        ]
    )
    mapping = _legend_label_map(legend)
    assert mapping["AP"] == "AP"
    assert mapping["SURFACE RACEWAY (WM2300)"] == "SURFACE RACEWAY (WM2300)"
    assert mapping["SURFACE RACEWAY"] == "SURFACE RACEWAY (WM2300)"


def test_aggregate_results_by_wing_name_sums_sheets():
    per_sheet = [
        {
            "wing_name": "A-WING-EAST",
            "counts": {"AP": 1, "R": 1},
            "wing_image": "04_wings/A-WING-EAST/page_007.jpg",
        },
        {
            "wing_name": "A-WING-EAST",
            "counts": {"AP": 2, "J": 1},
            "wing_image": "04_wings/A-WING-EAST/page_020.jpg",
        },
    ]
    aggregated = aggregate_results_by_wing_name(per_sheet)
    assert len(aggregated) == 1
    assert aggregated[0]["column_key"] == "A-WING-EAST"
    assert aggregated[0]["counts"] == {"AP": 3, "R": 1, "J": 1}
    assert aggregated[0]["sheets_processed"] == 2
