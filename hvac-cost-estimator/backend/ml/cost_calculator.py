"""Aggregate detected devices into a costed line-item report.

Pure logic: counts per device type are joined against the rate table
(``data/cost_rates.json``). Device types missing from the rate table get the
configured default unit cost and are flagged ``needs_review`` so a human
verifies them in the dashboard.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ml.base import ClassifiedDevice


class CostRatesError(ValueError):
    """The cost rate table is missing or malformed."""


@dataclass(frozen=True)
class DeviceRate:
    display_name: str
    unit_cost: float


@dataclass(frozen=True)
class CostRates:
    currency: str
    default_unit_cost: float
    rates: dict[str, DeviceRate]


@dataclass(frozen=True)
class CostLine:
    device_type: str
    display_name: str
    count: int
    unit_cost: float
    needs_review: bool = False

    @property
    def line_total(self) -> float:
        return round(self.count * self.unit_cost, 2)


@dataclass(frozen=True)
class CostingSummary:
    currency: str
    lines: list[CostLine] = field(default_factory=list)

    @property
    def grand_total(self) -> float:
        return round(sum(line.line_total for line in self.lines), 2)


def load_cost_rates(path: Path) -> CostRates:
    """Load and validate the device -> unit cost lookup table."""
    if not path.exists():
        raise CostRatesError(f"Cost rate table not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CostRatesError(f"Cost rate table is not valid JSON: {exc}") from exc

    try:
        rates = {
            device_type: DeviceRate(
                display_name=str(entry["display_name"]),
                unit_cost=float(entry["unit_cost"]),
            )
            for device_type, entry in raw["rates"].items()
        }
        return CostRates(
            currency=str(raw.get("currency", "USD")),
            default_unit_cost=float(raw.get("default_unit_cost", 0.0)),
            rates=rates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CostRatesError(f"Cost rate table is malformed: {exc}") from exc


def _prettify(device_type: str) -> str:
    return device_type.replace("_", " ").title()


def calculate_costs(
    devices: Iterable[ClassifiedDevice] | Counter[str],
    rates: CostRates,
) -> CostingSummary:
    """Aggregate device detections into costed line items with a grand total.

    Accepts either raw pipeline detections or a pre-computed
    ``Counter[device_type]`` (useful for tests and recalculation).
    """
    if isinstance(devices, Counter):
        counts = devices
    else:
        counts = Counter(device.device_type for device in devices)

    lines: list[CostLine] = []
    for device_type in sorted(counts):
        count = counts[device_type]
        if count <= 0:
            continue
        rate = rates.rates.get(device_type)
        if rate is not None:
            lines.append(
                CostLine(
                    device_type=device_type,
                    display_name=rate.display_name,
                    count=count,
                    unit_cost=rate.unit_cost,
                )
            )
        else:
            lines.append(
                CostLine(
                    device_type=device_type,
                    display_name=_prettify(device_type),
                    count=count,
                    unit_cost=rates.default_unit_cost,
                    needs_review=True,
                )
            )
    return CostingSummary(currency=rates.currency, lines=lines)
