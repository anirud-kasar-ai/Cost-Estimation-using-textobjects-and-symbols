"""Tests for room tagging, scale calibration, and pricing join."""

from __future__ import annotations

from ml.base import BoundingBox, ClassifiedDevice, Detection, OcrLine
from ml.cost_calculator import CostRates, DeviceRate, calculate_costs
from ml.location_tagger import looks_like_room_label, tag_devices_with_rooms
from ml.pricing import PricingLookup, resolve_price
from ml.scale_calibration import (
    apply_linear_feet,
    calibrate_from_ocr,
    feet_per_pixel_from_scale,
    is_line_type_box,
    parse_graphic_scale,
)


def _device(
    device_type: str,
    box: BoundingBox,
    *,
    page: int = 1,
    qty: float = 1.0,
    unit: str = "EA",
) -> ClassifiedDevice:
    return ClassifiedDevice(
        device_type=device_type,
        confidence=0.9,
        detection=Detection(box=box, score=0.9),
        page_number=page,
        quantity=qty,
        unit=unit,
    )


class TestLocationTagger:
    def test_room_label_heuristic(self) -> None:
        assert looks_like_room_label("ROOM 101")
        assert looks_like_room_label("Mech 1")
        assert not looks_like_room_label("SEE NOTE 3")

    def test_nearest_boxed_room(self) -> None:
        devices = [
            _device("thermostat", BoundingBox(100, 100, 120, 120)),
            _device("vav_box", BoundingBox(500, 500, 520, 520)),
        ]
        ocr = [
            OcrLine("ROOM 101", 0.95, BoundingBox(110, 80, 180, 95)),
            OcrLine("CORRIDOR A", 0.9, BoundingBox(490, 460, 580, 480)),
        ]
        tagged = tag_devices_with_rooms(devices, ocr)
        assert tagged[0].room_label == "ROOM 101"
        assert tagged[1].room_label == "CORRIDOR A"

    def test_mock_quadrant_fallback(self) -> None:
        devices = [_device("thermostat", BoundingBox(10, 10, 30, 30))]
        tagged = tag_devices_with_rooms(
            devices, [], page_width=900, page_height=900
        )
        assert tagged[0].room_label is not None


class TestScaleCalibration:
    def test_parse_imperial_scale(self) -> None:
        parsed = parse_graphic_scale('SCALE: 1/8" = 1\'-0"')
        assert parsed is not None
        drawing_in_per_ft, label = parsed
        assert abs(drawing_in_per_ft - 0.125) < 1e-9
        assert "1/8" in label

    def test_feet_per_pixel(self) -> None:
        # 1/8" = 1'-0" at 96 DPI → 0.125 * 96 = 12 px per foot
        fpp = feet_per_pixel_from_scale(0.125, 96.0)
        assert abs(fpp - (1.0 / 12.0)) < 1e-9

    def test_calibrate_from_ocr_line(self) -> None:
        cal = calibrate_from_ocr(
            [OcrLine('1/4" = 1\'-0"', 0.99)],
            dpi=120.0,
            use_mock_fallback=False,
        )
        assert cal.source == "ocr"
        assert cal.feet_per_pixel > 0

    def test_raceway_to_linear_feet(self) -> None:
        box = BoundingBox(0, 0, 240, 10)  # 240 px long
        assert is_line_type_box(box)
        devices = [_device("raceway", box)]
        cal = calibrate_from_ocr([], dpi=96.0, use_mock_fallback=True)
        out = apply_linear_feet(devices, cal)
        assert out[0].unit == "LF"
        assert out[0].quantity >= 1.0


class TestPricingJoin:
    def test_join_groups_by_category(self) -> None:
        class _Item:
            mfg = "Titus"
            part_number = "TMS-AA"
            display_name = "Supply Air Diffuser"
            category = "Diffusers & Grilles"
            unit = "EA"
            unit_cost = 185.0
            device_type = "supply_air_diffuser"

        item = _Item()
        lookup = PricingLookup(
            currency="USD",
            default_unit_cost=100.0,
            by_mfg_part={("titus", "tms-aa"): item},  # type: ignore[dict-item]
            by_device_type={"supply_air_diffuser": item},  # type: ignore[dict-item]
        )
        matched, cost, review = resolve_price(
            lookup, mfg=None, part_number=None, device_type="supply_air_diffuser"
        )
        assert matched is item
        assert cost == 185.0
        assert review is False

        rates = CostRates(
            currency="USD",
            default_unit_cost=100.0,
            rates={"supply_air_diffuser": DeviceRate("Supply Air Diffuser", 185.0)},
        )
        devices = [
            _device("supply_air_diffuser", BoundingBox(0, 0, 20, 20)),
            _device("supply_air_diffuser", BoundingBox(40, 40, 60, 60)),
        ]
        # stamp catalog fields as pipeline does
        from dataclasses import replace

        devices = [
            replace(d, mfg="Titus", part_number="TMS-AA", category="Diffusers & Grilles")
            for d in devices
        ]
        summary = calculate_costs(devices, rates, pricing=lookup)
        assert len(summary.lines) == 1
        assert summary.lines[0].count == 2
        assert summary.lines[0].category == "Diffusers & Grilles"
        assert summary.lines[0].line_total == 370.0
