"""SQLite pricing catalog: (mfg, part_number) → unit_cost.

Seeded from ``data/pricing_seed.json`` (manual — no free public HVAC price
feed). Dashboard CRUD updates the same table used by the join/aggregation
service when building costing report rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import PricingItem

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SEED_PATH = BACKEND_DIR / "data" / "pricing_seed.json"


@dataclass(frozen=True)
class PricingLookup:
    """In-memory join helpers keyed by mfg/part and by device_type alias."""

    currency: str
    default_unit_cost: float
    by_mfg_part: dict[tuple[str, str], PricingItem]
    by_device_type: dict[str, PricingItem]


def load_seed_file(path: Path | None = None) -> dict:
    seed_path = path or DEFAULT_SEED_PATH
    if not seed_path.exists():
        return {"currency": "USD", "default_unit_cost": 100.0, "items": []}
    return json.loads(seed_path.read_text(encoding="utf-8"))


def seed_pricing_table(session: Session, path: Path | None = None) -> int:
    """Insert seed rows when the pricing table is empty. Returns rows inserted."""
    existing = session.scalar(select(PricingItem.id).limit(1))
    if existing is not None:
        return 0

    raw = load_seed_file(path)
    inserted = 0
    for entry in raw.get("items", []):
        session.add(
            PricingItem(
                mfg=str(entry["mfg"]),
                part_number=str(entry["part_number"]),
                display_name=str(entry["display_name"]),
                category=str(entry.get("category") or "Uncategorized"),
                unit=str(entry.get("unit") or "EA"),
                unit_cost=float(entry["unit_cost"]),
                device_type=(
                    str(entry["device_type"]) if entry.get("device_type") else None
                ),
            )
        )
        inserted += 1
    if inserted:
        session.commit()
        logger.info("Seeded %d pricing rows from %s", inserted, path or DEFAULT_SEED_PATH)
    return inserted


def build_pricing_lookup(session: Session, *, currency: str = "USD", default_unit_cost: float = 100.0) -> PricingLookup:
    items = list(session.scalars(select(PricingItem).order_by(PricingItem.category, PricingItem.mfg)).all())
    by_mfg_part = {(i.mfg.casefold(), i.part_number.casefold()): i for i in items}
    by_device_type: dict[str, PricingItem] = {}
    for item in items:
        if item.device_type:
            by_device_type[item.device_type] = item
    return PricingLookup(
        currency=currency,
        default_unit_cost=default_unit_cost,
        by_mfg_part=by_mfg_part,
        by_device_type=by_device_type,
    )


def resolve_price(
    lookup: PricingLookup,
    *,
    mfg: str | None,
    part_number: str | None,
    device_type: str,
) -> tuple[PricingItem | None, float, bool]:
    """Return (matched_item_or_None, unit_cost, needs_review)."""
    if mfg and part_number:
        item = lookup.by_mfg_part.get((mfg.casefold(), part_number.casefold()))
        if item is not None:
            return item, float(item.unit_cost), False
    item = lookup.by_device_type.get(device_type)
    if item is not None:
        return item, float(item.unit_cost), False
    return None, lookup.default_unit_cost, True
