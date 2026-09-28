"""Tests for YOLO annotation → legend mapping."""

from __future__ import annotations

from pipeline.legend_match import attach_legend_match, match_class_to_legend
from pipeline.legend_ocr import LegendEntry
from pipeline.yolo_class_map import resolve_class_map


def _legend(*pairs: tuple[int, str]) -> list[LegendEntry]:
    return [LegendEntry(number=n, description=d, detail_ref=None, raw_line=f"{n}. {d}") for n, d in pairs]


def test_resolve_conduit_stub_2_weight():
    m = resolve_class_map("CONDUIT STUB 2")
    assert m is not None
    assert m.legend_phrase == "CONDUIT STUB"
    assert m.count_weight == 2


def test_resolve_surface_raceway_typo():
    m = resolve_class_map("SURFACE REACEWAY WM 5400")
    assert m is not None
    assert m.legend_phrase == "SURFACE RACEWAY"


def test_match_j_hook():
    legend = _legend((12, "J-HOOK (SINGLE/STACKED)"))
    m = match_class_to_legend("J HOOK", legend)
    assert m is not None
    assert m.legend_number == 12


def test_match_conduit_stub_2_into_stub():
    legend = _legend((3, "CONDUIT STUB"))
    m = attach_legend_match("CONDUIT STUB 2", legend)
    assert m is not None
    assert m.legend_number == 3
    assert m.count_weight == 2


def test_unmapped_class_returns_none():
    legend = _legend((1, "DATA RACK"))
    assert match_class_to_legend("classroom 15", legend) is None
    assert match_class_to_legend("NQ-CC", legend) is None


def test_single_network_camera_maps_to_network_camera():
    legend = _legend((8, "NETWORK CAMERA"), (9, "MULTI-SENSOR EXTERIOR NETWORK CAMERA (4-SENSOR)"))
    m = match_class_to_legend("SINGEL NETWORK CAMERA", legend)
    assert m is not None
    assert m.legend_number == 8


def test_network_camera_maps_to_multisensor():
    legend = _legend((8, "NETWORK CAMERA"), (9, "MULTI-SENSOR EXTERIOR NETWORK CAMERA (4-SENSOR)"))
    m = match_class_to_legend("NETWORK CAMERA", legend)
    assert m is not None
    assert m.legend_number == 9


def test_exhaust_fan_with_dtl():
    legend = _legend((14, "EXHAUST FAN"))
    m = match_class_to_legend("EXHAUST FAN", legend)
    assert m is not None
    assert m.legend_number == 14
