"""Synthetic realtime device pricing CSV + invoice from YOLO symbol counts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline import config

logger = logging.getLogger(__name__)

# Base catalog used to (re)seed pricing/device_prices.csv
_BASE_CATALOG: list[tuple[str, str, str, float]] = [
    # device_key, description, unit, base_usd
    ("CURB", "Curb / curb detail", "EA", 185.00),
    ("CURB DETAIL", "Curb detail", "EA", 185.00),
    ("VENT CURB", "Vent curb", "EA", 210.00),
    ("EXHAUST FAN", "Exhaust fan", "EA", 1240.00),
    # HEAT DETAIL is the same drawn symbol as HATCH DETAIL (see _DEVICE_ALIASES)
    ("HATCH DETAIL", "Hatch / heat detail", "EA", 2680.00),
    ("ROOF HATCH", "Roof hatch", "EA", 2680.00),
    ("ROOF ACCESS HATCH", "Roof access hatch", "EA", 2750.00),
    ("HEAT STACK", "Heat stack", "EA", 420.00),
    ("PIPE HOUSING", "Pipe housing", "EA", 390.00),
    ("PIPE PENETRATION", "Pipe penetration", "EA", 275.00),
    ("PIPE/TUBE PENETRATION", "Pipe/tube penetration", "EA", 295.00),
    ("PLUMBING STACK", "Plumbing stack", "EA", 510.00),
    ("ROOF DRAIN", "Roof drain", "EA", 640.00),
    ("ROOF DRAIN OVERFLOW", "Roof drain overflow", "EA", 720.00),
    ("SKYLIGHT", "Skylight", "EA", 1850.00),
    ("VENT INTAKE", "Vent intake", "EA", 480.00),
    ("PASSIVE AIR VENT", "Passive air vent", "EA", 360.00),
    ("DATA RACK", "Data rack", "EA", 2100.00),
    ("GROUND BOX", "Ground box", "EA", 145.00),
    ("JUNCTION BOX", "Junction box", "EA", 95.00),
    ("CONDUIT STUB", "Conduit stub", "EA", 55.00),
    ("SURFACE RACEWAY", "Surface raceway", "LF", 28.50),
    ("J-HOOK", "J-hook", "EA", 12.00),
    ("DATA POLE", "Data pole", "EA", 890.00),
    ("NETWORK CAMERA", "Network camera", "EA", 780.00),
    ("DEFAULT", "Unlisted detected device", "EA", 150.00),
]

CSV_HEADERS = [
    "device_key",
    "description",
    "unit",
    "unit_price_usd",
    "currency",
    "updated_at",
    # Filled only for manually updated rows (from the Symbol Pricing page)
    "updated_price_usd",
    "update_note",
]


def _norm_key(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# YOLO class names that are the SAME drawn symbol — folded into one canonical
# device key so pricing/invoicing never lists them as separate devices.
_DEVICE_ALIASES: dict[str, str] = {
    "HEAT DETAIL": "HATCH DETAIL",
}


def _canon_key(text: str) -> str:
    """Normalized key with same-symbol aliases collapsed."""
    key = _norm_key(text)
    return _DEVICE_ALIASES.get(key, key)


def pricing_csv_path() -> Path:
    return Path(config.PRICING_CSV_PATH)


def _overrides_path() -> Path:
    """Manual price edits (from the UI) that survive the synthetic drift."""
    return pricing_csv_path().parent / "price_overrides.json"


def load_price_overrides() -> dict[str, dict]:
    path = _overrides_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Collapse same-symbol aliases so old overrides still apply to the merged key.
    return {_canon_key(k): v for k, v in data.items()}


def set_price_override(
    device_key: str,
    unit_price_usd: float,
    *,
    description: str | None = None,
    unit: str | None = None,
    reason: str | None = None,
) -> "PriceRow":
    """Persist a manual price edit and rewrite the CSV with it applied."""
    key = _canon_key(device_key)
    if not key:
        raise ValueError("Empty device key")
    overrides = load_price_overrides()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    entry = dict(overrides.get(key) or {})
    entry["unit_price_usd"] = round(float(unit_price_usd), 2)
    if description:
        entry["description"] = str(description)
    if unit:
        entry["unit"] = str(unit)
    entry["reason"] = str(reason or "").strip()
    entry["updated_at"] = stamp
    overrides[key] = entry
    path = _overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    # Probe writability first so a locked CSV (e.g. open in Excel) gives a clear error.
    csv_path = pricing_csv_path()
    if csv_path.is_file():
        try:
            with csv_path.open("a", encoding="utf-8"):
                pass
        except PermissionError:
            raise ValueError(
                "device_prices.csv is open in another program (e.g. Excel) — "
                "close it and save the price again."
            ) from None
    ensure_pricing_csv(refresh=False)
    row = load_price_index().get(key)
    if row is None:
        raise ValueError(f"Failed to persist price for {key}")
    return row


@dataclass
class PriceRow:
    device_key: str
    description: str
    unit: str
    unit_price_usd: float
    currency: str = "USD"
    updated_at: str = ""
    # Set only when the price was manually updated in the UI
    updated_price_usd: float | None = None
    update_note: str = ""


@dataclass
class InvoiceLine:
    description: str
    device_key: str
    quantity: int
    unit: str
    unit_price_usd: float
    line_total_usd: float
    yolo_classes: list[str]


def _synthetic_factor(base: float, device_key: str, when: datetime | None = None) -> float:
    """Synthetic but 'live' price: small deterministic drift from UTC date/time."""
    when = when or datetime.now(timezone.utc)
    day = when.timetuple().tm_yday
    minute = when.hour * 60 + when.minute
    digest = hashlib.sha256(f"{device_key}|{when.date().isoformat()}".encode()).hexdigest()
    wave = int(digest[:6], 16) / float(0xFFFFFF)
    # ±4% daily wave + tiny minute tick so refreshes look live
    factor = 1.0 + 0.04 * math.sin((day + wave * 10.0) * 0.35) + 0.002 * math.sin(minute / 17.0)
    price = max(1.0, round(base * factor, 2))
    return price


_DEFAULT_BASE_USD = 150.0


def _trained_class_names() -> list[str]:
    """Union of class names across every available YOLO weight file.

    Ensures the pricing catalog covers ALL symbols the models are trained on,
    not just the hand-curated base catalog.
    """
    try:
        from pipeline.yolo_detect import available_model_paths, class_names
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    seen: set[str] = set()
    try:
        paths = available_model_paths()
    except Exception:  # noqa: BLE001
        return []
    for path in paths:
        try:
            names = class_names(path)
        except Exception:  # noqa: BLE001
            logger.warning("Could not read class names from %s", path)
            continue
        for name in names.values():
            nk = _norm_key(name)
            if nk and nk not in seen:
                seen.add(nk)
                out.append(name)
    return sorted(out, key=_norm_key)


def ensure_pricing_csv(*, refresh: bool = True) -> Path:
    """Create or refresh pricing/device_prices.csv.

    Synthetic drift only applies when PRICE_DRIFT_ENABLED is set; by default
    prices stay stable so estimates are reproducible between runs.
    """
    refresh = refresh and bool(getattr(config, "PRICE_DRIFT_ENABLED", False))
    path = pricing_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds").replace("+00:00", "Z")

    existing: dict[str, PriceRow] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = _canon_key(row.get("device_key") or "")
                if not key or key in existing:
                    continue
                try:
                    price = float(row.get("unit_price_usd") or 0)
                except ValueError:
                    price = 0.0
                existing[key] = PriceRow(
                    device_key=key,
                    description=str(row.get("description") or key),
                    unit=str(row.get("unit") or "EA"),
                    unit_price_usd=price,
                    currency=str(row.get("currency") or "USD"),
                    updated_at=str(row.get("updated_at") or ""),
                )

    rows: list[PriceRow] = []
    seen: set[str] = set()
    for key, desc, unit, base in _BASE_CATALOG:
        nk = _canon_key(key)
        if nk in seen:
            continue
        seen.add(nk)
        price = _synthetic_factor(base, nk, now) if refresh else (
            existing[nk].unit_price_usd
            if nk in existing and existing[nk].unit_price_usd > 0
            else base
        )
        rows.append(
            PriceRow(
                device_key=nk,
                description=desc,
                unit=unit,
                unit_price_usd=float(price),
                currency="USD",
                updated_at=stamp,
            )
        )

    # One row for every symbol class the YOLO models are trained on, so the
    # Symbol Pricing page lists ALL trained symbols (base catalog covers only some).
    for name in _trained_class_names():
        nk = _canon_key(name)
        if not nk or nk in seen:
            continue
        seen.add(nk)
        prev = existing.get(nk)
        base = prev.unit_price_usd if prev and prev.unit_price_usd > 0 else _DEFAULT_BASE_USD
        price = _synthetic_factor(base, nk, now) if refresh else base
        rows.append(
            PriceRow(
                device_key=nk,
                description=(
                    prev.description
                    if prev and prev.description and prev.description != nk
                    else name.strip().title()
                ),
                unit=(prev.unit if prev else "EA") or "EA",
                unit_price_usd=float(price),
                currency=(prev.currency if prev else "USD") or "USD",
                updated_at=stamp if refresh else (prev.updated_at if prev else stamp),
            )
        )

    # Keep any extra keys previously added
    for key, row in existing.items():
        if key in seen:
            continue
        base_extra = row.unit_price_usd or 150.0
        price = _synthetic_factor(base_extra, key, now) if refresh else base_extra
        rows.append(
            PriceRow(
                device_key=key,
                description=row.description,
                unit=row.unit or "EA",
                unit_price_usd=float(price),
                currency=row.currency or "USD",
                updated_at=stamp if refresh else row.updated_at,
            )
        )

    # Manual UI edits win over both the base catalog and the synthetic drift.
    overrides = load_price_overrides()
    if overrides:
        final: list[PriceRow] = []
        covered: set[str] = set()
        for row in rows:
            ov = overrides.get(row.device_key)
            if ov and ov.get("unit_price_usd") is not None:
                row = PriceRow(
                    device_key=row.device_key,
                    description=str(ov.get("description") or row.description),
                    unit=str(ov.get("unit") or row.unit),
                    unit_price_usd=float(ov["unit_price_usd"]),
                    currency=row.currency,
                    updated_at=str(ov.get("updated_at") or row.updated_at),
                    updated_price_usd=float(ov["unit_price_usd"]),
                    update_note=str(ov.get("reason") or ""),
                )
            final.append(row)
            covered.add(row.device_key)
        for key, ov in overrides.items():
            if key in covered or ov.get("unit_price_usd") is None:
                continue
            final.append(
                PriceRow(
                    device_key=key,
                    description=str(ov.get("description") or key),
                    unit=str(ov.get("unit") or "EA"),
                    unit_price_usd=float(ov["unit_price_usd"]),
                    currency="USD",
                    updated_at=str(ov.get("updated_at") or stamp),
                    updated_price_usd=float(ov["unit_price_usd"]),
                    update_note=str(ov.get("reason") or ""),
                )
            )
        rows = final

    try:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    except PermissionError:
        # CSV open in Excel etc. — keep serving the existing file instead of crashing.
        logger.warning(
            "device_prices.csv is locked by another program — skipping refresh, using existing file"
        )
    return path


def load_price_index() -> dict[str, PriceRow]:
    path = ensure_pricing_csv(refresh=False)
    index: dict[str, PriceRow] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = _canon_key(row.get("device_key") or "")
            if not key or key in index:
                continue
            try:
                price = float(row.get("unit_price_usd") or 0)
            except ValueError:
                price = 0.0
            index[key] = PriceRow(
                device_key=key,
                description=str(row.get("description") or key),
                unit=str(row.get("unit") or "EA"),
                unit_price_usd=price,
                currency=str(row.get("currency") or "USD"),
                updated_at=str(row.get("updated_at") or ""),
            )
    return index


def lookup_price(index: dict[str, PriceRow], *candidates: str) -> PriceRow:
    for raw in candidates:
        key = _canon_key(raw)
        if key in index:
            return index[key]
        # Try core before comma / DTL
        core = _canon_key(re.split(r"[,–—\-]", raw or "", maxsplit=1)[0])
        if core in index:
            return index[core]
    return index.get(
        "DEFAULT",
        PriceRow("DEFAULT", "Unlisted detected device", "EA", 150.0),
    )


def build_invoice(
    count_rows: list[dict[str, Any]],
    *,
    job_id: str,
    refresh_prices: bool = True,
    tax_rate: float | None = None,
) -> dict[str, Any]:
    """Join symbol counts × CSV unit prices → invoice payload."""
    if refresh_prices:
        ensure_pricing_csv(refresh=True)
    index = load_price_index()
    lines: list[InvoiceLine] = []
    for row in count_rows:
        qty = int(row.get("count") or row.get("yolo_count") or 0)
        if qty <= 0:
            continue
        desc = str(row.get("description") or "").strip() or "Unknown"
        classes = [str(c) for c in (row.get("yolo_classes") or []) if c]
        price = lookup_price(index, desc, *(classes or [desc]))
        line_total = round(price.unit_price_usd * qty, 2)
        lines.append(
            InvoiceLine(
                description=desc,
                device_key=price.device_key,
                quantity=qty,
                unit=price.unit,
                unit_price_usd=price.unit_price_usd,
                line_total_usd=line_total,
                yolo_classes=classes,
            )
        )

    subtotal = round(sum(ln.line_total_usd for ln in lines), 2)
    tax_rate = float(config.INVOICE_TAX_RATE) if tax_rate is None else float(tax_rate)
    tax = round(subtotal * tax_rate, 2)
    grand = round(subtotal + tax, 2)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "job_id": job_id,
        "currency": "USD",
        "generated_at": now,
        "pricing_csv": str(pricing_csv_path()),
        "tax_rate": tax_rate,
        "subtotal_usd": subtotal,
        "tax_usd": tax,
        "grand_total_usd": grand,
        "line_count": len(lines),
        "lines": [asdict(ln) for ln in lines],
    }
