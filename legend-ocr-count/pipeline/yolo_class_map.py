"""Explicit YOLO annotation name → legend description mapping.

Training used short annotation tags; legends use full equipment names.
Only classes present in this map are matched. Unmapped detections are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _norm_key(text: str) -> str:
    t = (text or "").upper()
    # Fix common training typos so keys still resolve
    t = t.replace("REACEWAY", "RACEWAY")
    t = t.replace("ALARAM", "ALARM")
    t = t.replace("PANNEL", "PANEL")
    t = t.replace("CABINATE", "CABINET")
    t = t.replace("SINGEL", "SINGLE")
    t = t.replace("ACESS", "ACCESS")
    t = t.replace("WIRELESS ACESS", "WIRELESS ACCESS")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass(frozen=True)
class ClassMapEntry:
    """Maps a YOLO class to one or more legend phrases and a count weight.

    Any listed phrase can match — useful when different PDFs use different DTL refs
    (e.g. EXHAUST FAN, DTL P/6 vs EXHAUST FAN, DTL C/5).
    """

    legend_phrases: tuple[str, ...]
    count_weight: int = 1

    def __init__(self, *phrases: str, count_weight: int = 1):
        object.__setattr__(self, "legend_phrases", tuple(p for p in phrases if p))
        object.__setattr__(self, "count_weight", count_weight)

    @property
    def legend_phrase(self) -> str:
        """Primary phrase (back-compat)."""
        return self.legend_phrases[0] if self.legend_phrases else ""


# Annotation / YOLO class → legend description phrase (as on the symbol legend)
_CLASS_TO_LEGEND: dict[str, ClassMapEntry] = {
    "DATA RACK": ClassMapEntry("DATA RACK"),
    "GROUND BOX": ClassMapEntry("GROUND BOX"),
    "JUNCTION BOX": ClassMapEntry("JUNCTION BOX"),
    "CONDUIT STUB": ClassMapEntry("CONDUIT STUB"),
    # Paired stubs: each detection counts as 2 toward CONDUIT STUB
    "CONDUIT STUB 2": ClassMapEntry("CONDUIT STUB", count_weight=2),
    "SURFACE RACEWAY 2300": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM2300",
        "SURFACE RACEWAY WM 2300",
        "SURFACE REACEWAY WM2300",
        "SURFACE REACEWAY WM 2300",
        # Dual-channel raceway rows (detail WM2300BACD) are the same symbol
        "SURFACE RACEWAY (DUAL CHANNEL)",
    ),
    "SURFACE RACEWAY 5400": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5400",
        "SURFACE RACEWAY WM 5400",
        "SURFACE REACEWAY WM5400",
        "SURFACE REACEWAY WM 5400",
        "SURFACE RACEWAY (DUAL CHANNEL)",
    ),
    "SURFACE RACEWAY WM2300": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM2300",
        "SURFACE RACEWAY WM 2300",
        "SURFACE REACEWAY WM2300",
        "SURFACE REACEWAY WM 2300",
    ),
    "SURFACE RACEWAY WM 2300": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM2300",
        "SURFACE RACEWAY WM 2300",
        "SURFACE REACEWAY WM2300",
        "SURFACE REACEWAY WM 2300",
    ),
    "SURFACE RACEWAY WM5400": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5400",
        "SURFACE RACEWAY WM 5400",
        "SURFACE REACEWAY WM5400",
        "SURFACE REACEWAY WM 5400",
    ),
    "SURFACE RACEWAY WM 5400": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5400",
        "SURFACE RACEWAY WM 5400",
        "SURFACE REACEWAY WM5400",
        "SURFACE REACEWAY WM 5400",
    ),
    "SURFACE RACEWAY WM5500": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5500",
        "SURFACE RACEWAY WM 5500",
    ),
    "SURFACE RACEWAY WM 5500": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5500",
        "SURFACE RACEWAY WM 5500",
    ),
    # Training typo class names → same SURFACE RACEWAY / REACEWAY legend rows
    "SURFACE REACEWAY WM 5400": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5400",
        "SURFACE RACEWAY WM 5400",
        "SURFACE REACEWAY WM5400",
        "SURFACE REACEWAY WM 5400",
    ),
    "SURFACE REACEWAY WM5400": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM5400",
        "SURFACE RACEWAY WM 5400",
        "SURFACE REACEWAY WM5400",
        "SURFACE REACEWAY WM 5400",
    ),
    "SURFACE REACEWAY WM2300": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM2300",
        "SURFACE RACEWAY WM 2300",
        "SURFACE REACEWAY WM2300",
        "SURFACE REACEWAY WM 2300",
    ),
    "SURFACE REACEWAY WM 2300": ClassMapEntry(
        "SURFACE RACEWAY",
        "SURFACE REACEWAY",
        "SURFACE RACEWAY WM2300",
        "SURFACE RACEWAY WM 2300",
        "SURFACE REACEWAY WM2300",
        "SURFACE REACEWAY WM 2300",
    ),
    "J HOOK": ClassMapEntry("J-HOOK (SINGLE/STACKED)", "J-HOOK", "J HOOK"),
    "DATA POLE": ClassMapEntry("DATA POLE"),
    "DATA PERMANENT LINK": ClassMapEntry(
        "DATA PERMANENT LINK, CAT6A JACK / CABLE",
        "DATA PERMANENT LINK CAT6A JACK / CABLE",
        "DATA PERMANENT LINK",
        "CAT6A JACK / CABLE",
    ),
    "INTERIOR WIRELESS ACCESS POINT": ClassMapEntry("INTERIOR WIRELESS ACCESS POINT"),
    "SINGLE NETWORK CAMERA": ClassMapEntry("NETWORK CAMERA"),
    "SINGEL NETWORK CAMERA": ClassMapEntry("NETWORK CAMERA"),
    "MOUNT": ClassMapEntry("MOUNT"),
    "NETWORK CAMERA": ClassMapEntry(
        "MULTI-SENSOR EXTERIOR NETWORK CAMERA (4-SENSOR)",
        # Other sets use the 2-sensor variant of the same camera symbol
        "MULTI-SENSOR EXTERIOR NETWORK CAMERA (2-SENSOR)",
        "MULTI-SENSOR EXTERIOR NETWORK CAMERA",
    ),
    "NETWORK VIDEO RECORDER": ClassMapEntry("NETWORK VIDEO RECORDER"),
    "SIGNAL TERMINAL CABINET": ClassMapEntry("SIGNAL TERMINAL CABINET"),
    "SIGNAL TERMINAL CABINATE": ClassMapEntry("SIGNAL TERMINAL CABINET"),
    "GROUND BUS BAR": ClassMapEntry("GROUND BUS BAR"),
    "MODULE COMBO BOX": ClassMapEntry(
        "CAT6A DATA DROP LOCATION (QTY = 1) - IP CLOCK/SPEAKER/IP MODULE COMBO BOX BOGEN",
        "CAT6A DATA DROP LOCATION (QTY = 1) - IP CLOCK/SPEAKER/IP MODULE COMBO BOX",
        "IP CLOCK/SPEAKER/IP MODULE COMBO BOX BOGEN",
        "IP MODULE COMBO BOX BOGEN",
        "MODULE COMBO BOX BOGEN",
        "MODULE COMBO BOX",
        # Some sets spec ATLAS instead of BOGEN for the same combo box symbol
        "CAT6A DATA DROP LOCATION (QTY = 1) - IP CLOCK/SPEAKER/IP MODULE COMBO BOX ATLAS",
        "CAT6A DATA DROP LOCATION (QTY = 1) IP CLOCK/SPEAKER/IP MODULE COMBO BOX ATLAS",
        "IP CLOCK/SPEAKER/IP MODULE COMBO BOX ATLAS",
        "IP MODULE COMBO BOX ATLAS",
        "MODULE COMBO BOX ATLAS",
    ),
    "MODULE BOGEN": ClassMapEntry(
        "MODULE BOGEN",
        "EXTERIOR INTERCOM SPEAKER/IP MODULE BOGEN",
        "CAT6A DATA DROP LOCATION - EXTERIOR INTERCOM SPEAKER/IP MODULE BOGEN",
        # Same symbol without the vendor suffix / dash
        "EXTERIOR INTERCOM SPEAKER/IP MODULE",
        "CAT6A DATA DROP LOCATION EXTERIOR INTERCOM SPEAKER/IP MODULE",
    ),
    "IACP": ClassMapEntry("INTRUSION ALARM CONTROL PANEL"),
    "FIRE ALARM CONTROL PANEL": ClassMapEntry("FIRE ALARM CONTROL PANEL"),
    "FIRE ALARAM CONTROL PANNEL": ClassMapEntry("FIRE ALARM CONTROL PANEL"),
    "MINIMUM POINT OF ENTRY": ClassMapEntry("MINIMUM POINT OF ENTRY"),
    "INTERCOM CONSOLE": ClassMapEntry("INTERCOM CONSOLE"),
    "LADDER": ClassMapEntry("LADDER RACK"),
    "EXTERIOR WIRELESS ACCESS POINT HIGH DENSITY": ClassMapEntry(
        "INTERIOR WIRELESS ACCESS POINT, HIGH DENSITY"
    ),
    "EXTERIOR WIRELESS ACESS POINT HIGH DENSITY": ClassMapEntry(
        "INTERIOR WIRELESS ACCESS POINT, HIGH DENSITY"
    ),
    "APWP": ClassMapEntry(
        "CAT6A DATA DROP LOCATION (QTY = 2) - EXTERIOR WIRELESS ACCESS POINT"
    ),
    "WALL MOUNT": ClassMapEntry("PHONE HANDSET EQUIPMENT (WALL MOUNT)"),
    "INTERIOR SPEAKER": ClassMapEntry(
        "CAT6A DATA DROP LOCATION - INTERIOR INTERCOM SPEAKER/IP MODULE"
    ),
    "LAY IN SPEAKERIP MODULE": ClassMapEntry(
        "CAT6A DATA DROP LOCATION - LAY IN SPEAKER/IP MODULE"
    ),
    "LAYER INTERCOM SPEAKER 25V": ClassMapEntry("LAY IN INTERCOM SPEAKER (25V)"),
    "TEL": ClassMapEntry("PHONE HANDSET EQUIPMENT"),
    "WIRE GUARD": ClassMapEntry("CAT6A DATA DROP LOCATION (QTY = 1) - VAPE SENSOR"),
    "NEMA4": ClassMapEntry("NEMA4 IDF ENCLOSURE", "NEMA4"),
    "ROOF DRAIN": ClassMapEntry(
        "ROOF DRAIN",
        "ROOF DRAIN, DTL F/5",
        "ROOF DRAIN, DTL H/5",
        "ROOF DRAIN, DTL L/6",
        "ROOF DRAIN, DTL N/6",
        "ROOF DRAIN, DTL Q/6",
        "ROOF DRAIN OVERFLOW, DTL M/6",
        "ROOF DRAIN OVERFLOW",
    ),
    # Current roof model class names (symbol_detector_best.pt)
    # Phrases collected from all drawing-zoom-split/storage/technical_symbols PDFs.
    "CURB": ClassMapEntry(
        "CURB DETAIL, DTL N/6",
        "CURB DETAIL, DTL C/4",
        "CURB DETAIL, DTL F/5",
        "CURB DETAIL, DTL L/6",
        "CURB, DTL C/4",
        "CURB DETAIL",
        "CURB",
        "VENT CURB, DTL F/5",
        "VENT CURB",
    ),
    "PIPE HOUSING": ClassMapEntry(
        "PIPE HOUSING, DTL G/5",
        "PIPE HOUSING, DTL M/6",
        "PIPE HOUSING, DTL W/8",
        "PIPE HOUSING",
        "HEAT STACK, DTL G/5",
        "HEAT STACK, DTL E/5",
        "HEAT STACK, DTL P/6",
        "HEAT STACK",
        "HEAT DETAIL",
    ),
    "HATCH DETAIL": ClassMapEntry(
        "HATCH DETAIL, DTL Q/7",
        "HATCH DETAIL",
        "ROOF HATCH, DTL E/5",
        "ROOF HATCH",
        "ROOF ACCESS HATCH",
    ),
    "HEAT DETAIL": ClassMapEntry(
        "HEAT STACK, DTL G/5",
        "HEAT STACK, DTL E/5",
        "HEAT STACK, DTL P/6",
        "HEAT STACK",
        "HEAT DETAIL",
        "PIPE HOUSING, DTL G/5",
        "PIPE HOUSING, DTL M/6",
        "PIPE HOUSING, DTL W/8",
        "PIPE HOUSING",
    ),
    "PIPE PENETRATION": ClassMapEntry(
        "PIPE PENETRATION, DTL F/4",
        "PIPE PENETRATION, DTL E/4",
        "PIPE PENETRATION, DTL K/5",
        "PIPE PENETRATION, DTL, K/5",
        "PIPE PENETRATION, DTL X/8",
        "PIPE/TUBE PENETRATION, DTL S/7",
        "PIPE/TUBE PENETRATION",
        "PIPE PENETRATION",
        "PIPE PENETRATIO",
    ),
    "SKYLIGHT": ClassMapEntry(
        "SKYLIGHT, DTL U/8",
        "SKYLIGHT, DTL D/4",
        "SKYLIGHT",
        "SKY LIGHT",
    ),
    # Different technical-symbol PDFs use different detail refs for the same symbol
    "EXHAUST FAN": ClassMapEntry(
        "EXHAUST FAN",
        "EXHAUST FAN, DTL P/6",
        "EXHAUST FAN, DTL C/5",
        "EXHAUST FAN, DTL N/6",
        "EXHAUST FAN, DTL N",
        "EXHAUST FAN OR GRILLE",
    ),
    "PLUMBING STACK": ClassMapEntry(
        "PLUMBING STACK",
        "PLUMBING STACK, DTL D/5",
        "PLUMBING STACK, DTL L/6",
        "PLUMBING STACK, DTL V/7",
    ),
    "HEAT STACK": ClassMapEntry(
        "HEAT STACK, DTL G/5",
        "HEAT STACK, DTL E/5",
        "HEAT STACK, DTL P/6",
        "HEAT STACK",
        "HEAT DETAIL",
        "PIPE HOUSING, DTL G/5",
        "PIPE HOUSING, DTL M/6",
        "PIPE HOUSING, DTL W/8",
        "PIPE HOUSING",
    ),
    "VENT INTAKE": ClassMapEntry(
        "VENT INTAKE",
        "VENT INTAKE, DTL Q/6",
        "PASSIVE AIR VENT, DTL U/7",
        "PASSIVE AIR VENT",
    ),
    "ROOF HATCH": ClassMapEntry(
        "ROOF HATCH",
        "ROOF HATCH, DTL E/5",
        "HATCH DETAIL, DTL Q/7",
        "HATCH DETAIL",
        "ROOF ACCESS HATCH",
    ),
}

# Normalized lookup (handles spacing / typos)
_NORM_MAP: dict[str, ClassMapEntry] = {
    _norm_key(k): v for k, v in _CLASS_TO_LEGEND.items()
}


def resolve_class_map(class_name: str) -> ClassMapEntry | None:
    """Return mapped legend phrase + weight for a YOLO class, or None if unmapped."""
    raw = (class_name or "").strip()
    if not raw:
        return None
    if raw in _CLASS_TO_LEGEND:
        return _CLASS_TO_LEGEND[raw]
    key = _norm_key(raw)
    if key in _NORM_MAP:
        return _NORM_MAP[key]
    # SURFACE RACEWAY WM#### / with spaces → SURFACE RACEWAY
    m = re.search(r"SURFACE\s+RACEWAY.*\b(2300|5400|5500)\b", key)
    if m:
        return ClassMapEntry("SURFACE RACEWAY")
    if "SURFACE" in key and "RACEWAY" in key:
        return ClassMapEntry("SURFACE RACEWAY")
    return None


def mapped_class_names() -> list[str]:
    return sorted(_CLASS_TO_LEGEND.keys())
