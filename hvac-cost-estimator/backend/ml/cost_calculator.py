"""Aggregate detected devices into a costed line-item report.

Joins quantity rows (detections) against the SQLite pricing catalog keyed by
``(mfg, part_number)`` (with ``device_type`` fallback). Falls back to the JSON
rate table when no DB session / pricing lookup is supplied so existing tests
and offline scripts keep working.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ml.base import ClassifiedDevice
from ml.pricing import PricingLookup, resolve_price


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
    category: str = "Uncategorized"
    unit: str = "EA"
    mfg: str | None = None
    part_number: str | None = None
    locations: str | None = None
    # Index into the devices list used for review crop (set by pipeline persist)
    sample_device_index: int | None = None

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


def _quantity_as_count(device: ClassifiedDevice) -> int:
    """Integer quantity for report rows (EA count or rounded linear feet)."""
    if device.unit == "LF":
        return max(1, int(round(device.quantity)))
    return max(1, int(round(device.quantity)))


def calculate_costs(
    devices: Iterable[ClassifiedDevice] | Counter[str],
    rates: CostRates,
    pricing: PricingLookup | None = None,
) -> CostingSummary:
    """Aggregate device detections into costed line items with a grand total.

    Accepts either raw pipeline detections or a pre-computed
    ``Counter[device_type]`` (useful for tests and recalculation).

    When ``pricing`` is provided, rows are joined on (mfg, part_number) /
    device_type and grouped by category for the dashboard report.
    """
    if isinstance(devices, Counter):
        return _calculate_from_counter(devices, rates)

    device_list = list(devices)
    if pricing is not None:
        return _calculate_from_pricing(device_list, rates, pricing)
    return _calculate_from_counter(
        Counter(device.device_type for device in device_list), rates
    )


def _calculate_from_counter(counts: Counter[str], rates: CostRates) -> CostingSummary:
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


def _calculate_from_pricing(
    devices: list[ClassifiedDevice],
    rates: CostRates,
    pricing: PricingLookup,
) -> CostingSummary:
    """Join detections × pricing → costing rows grouped by category / SKU."""

    @dataclass
    class _Bucket:
        device_type: str
        display_name: str
        category: str
        unit: str
        mfg: str | None
        part_number: str | None
        unit_cost: float
        needs_review: bool
        quantity: float = 0.0
        locations: list[str] = field(default_factory=list)
        sample_index: int | None = None

    buckets: dict[tuple[str, str, str, str], _Bucket] = {}

    for index, device in enumerate(devices):
        item, unit_cost, needs_review = resolve_price(
            pricing,
            mfg=device.mfg,
            part_number=device.part_number,
            device_type=device.device_type,
        )
        mfg = (item.mfg if item else device.mfg) or None
        part = (item.part_number if item else device.part_number) or None
        display = (
            item.display_name
            if item
            else rates.rates.get(device.device_type, DeviceRate(_prettify(device.device_type), 0)).display_name
            if device.device_type in rates.rates
            else _prettify(device.device_type)
        )
        if item is None and device.device_type in rates.rates:
            display = rates.rates[device.device_type].display_name
            unit_cost = rates.rates[device.device_type].unit_cost
            needs_review = False
        category = (
            (item.category if item else None)
            or device.category
            or "Uncategorized"
        )
        unit = device.unit if device.unit == "LF" else ((item.unit if item else None) or "EA")
        key = (category, device.device_type, mfg or "", part or "")
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                device_type=device.device_type,
                display_name=display,
                category=category,
                unit=unit,
                mfg=mfg,
                part_number=part,
                unit_cost=unit_cost,
                needs_review=needs_review,
            )
            buckets[key] = bucket
        qty = device.quantity if device.unit == "LF" else max(1.0, device.quantity)
        bucket.quantity += qty
        if device.room_label and device.room_label not in bucket.locations:
            bucket.locations.append(device.room_label)
        if bucket.sample_index is None:
            bucket.sample_index = index
        bucket.needs_review = bucket.needs_review or needs_review or device.confidence < 0.85

    lines: list[CostLine] = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], k[2], k[3])):
        bucket = buckets[key]
        count = max(1, int(round(bucket.quantity)))
        locations = " · ".join(bucket.locations) if bucket.locations else None
        lines.append(
            CostLine(
                device_type=bucket.device_type,
                display_name=bucket.display_name,
                count=count,
                unit_cost=bucket.unit_cost,
                needs_review=bucket.needs_review,
                category=bucket.category,
                unit=bucket.unit,
                mfg=bucket.mfg,
                part_number=bucket.part_number,
                locations=locations,
                sample_device_index=bucket.sample_index,
            )
        )
    return CostingSummary(currency=pricing.currency or rates.currency, lines=lines)
