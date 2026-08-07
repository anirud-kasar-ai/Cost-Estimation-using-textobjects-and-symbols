"""Tests for cost aggregation and the rate table loader."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from config import BACKEND_DIR
from ml.cost_calculator import (
    CostRates,
    CostRatesError,
    DeviceRate,
    calculate_costs,
    load_cost_rates,
)

RATES = CostRates(
    currency="USD",
    default_unit_cost=100.0,
    rates={
        "supply_air_diffuser": DeviceRate("Supply Air Diffuser", 185.0),
        "co2_sensor": DeviceRate("CO2 Sensor", 320.0),
    },
)


class TestCalculateCosts:
    def test_line_totals_and_grand_total(self) -> None:
        summary = calculate_costs(
            Counter({"supply_air_diffuser": 6, "co2_sensor": 2}), RATES
        )

        by_type = {line.device_type: line for line in summary.lines}
        assert by_type["supply_air_diffuser"].line_total == 6 * 185.0
        assert by_type["co2_sensor"].line_total == 2 * 320.0
        assert summary.grand_total == 6 * 185.0 + 2 * 320.0
        assert summary.currency == "USD"

    def test_unknown_device_gets_default_rate_and_review_flag(self) -> None:
        summary = calculate_costs(Counter({"smoke_damper": 3}), RATES)

        (line,) = summary.lines
        assert line.unit_cost == RATES.default_unit_cost
        assert line.needs_review is True
        assert line.display_name == "Smoke Damper"

    def test_known_devices_not_flagged(self) -> None:
        summary = calculate_costs(Counter({"co2_sensor": 1}), RATES)
        assert summary.lines[0].needs_review is False

    def test_empty_input_yields_empty_report(self) -> None:
        summary = calculate_costs(Counter(), RATES)
        assert summary.lines == []
        assert summary.grand_total == 0.0

    def test_lines_sorted_by_device_type(self) -> None:
        summary = calculate_costs(
            Counter({"co2_sensor": 1, "supply_air_diffuser": 1}), RATES
        )
        assert [line.device_type for line in summary.lines] == [
            "co2_sensor",
            "supply_air_diffuser",
        ]

    def test_rounding_of_totals(self) -> None:
        rates = CostRates(
            currency="USD",
            default_unit_cost=0.0,
            rates={"thermostat": DeviceRate("Thermostat", 0.1)},
        )
        summary = calculate_costs(Counter({"thermostat": 3}), rates)
        assert summary.lines[0].line_total == 0.3


class TestLoadCostRates:
    def test_loads_seed_rate_table(self) -> None:
        rates = load_cost_rates(BACKEND_DIR / "data" / "cost_rates.json")

        assert rates.currency == "USD"
        assert rates.rates["supply_air_diffuser"].unit_cost > 0
        assert rates.default_unit_cost > 0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CostRatesError, match="not found"):
            load_cost_rates(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(CostRatesError, match="not valid JSON"):
            load_cost_rates(bad)

    def test_malformed_schema_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"rates": {"x": {"unit_cost": "abc"}}}), encoding="utf-8")
        with pytest.raises(CostRatesError, match="malformed"):
            load_cost_rates(bad)
