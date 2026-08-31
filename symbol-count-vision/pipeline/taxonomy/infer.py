"""Infer shape_class / ContextRule / count_semantics from a legend row.

This is the migrated look-alike table that used to live in:
  - suppress_hourglass_on_triangles / suppress_triangles_near_tags / templates
  - architect-OCR, symbol-locate, and count-verify prompt prose
  - reclassify_j_hook_clusters (rule text; clustering still runs in the resolver)

Do not duplicate these mappings in prompts or NMS suppressors.
"""

from __future__ import annotations

import re

from pipeline.cv.tag_match import count_key
from pipeline.cv.template_match import LINEAR_GLYPH_RE
from pipeline.extraction.symbol_table_extractor import SymbolEntry
from pipeline.taxonomy.schema import ContextRule, CountSemantics, ShapeClass

_JHOOK_RE = re.compile(r"J-?HOOK", re.I)
_STUB_RE = re.compile(r"CONDUIT\s+STUB", re.I)
_POLE_RE = re.compile(r"DATA\s+POLE", re.I)
_CAMERA_RE = re.compile(r"NETWORK\s+CAMERA|\bCAMERA\b", re.I)
_AP_RE = re.compile(r"\bACCESS\s+POINT\b|\bWIRELESS\b|\bWAP\b", re.I)
_RACEWAY_RE = re.compile(r"SURFACE\s+RACEWAY", re.I)
_HEX_RE = re.compile(r"\bHEXAGON\b", re.I)
_SHORT_TAG_RE = re.compile(r"^(?:#|[A-Z]{1,4}\d{0,3})$")


def _blob(entry: SymbolEntry, key: str) -> str:
    return " ".join(
        part
        for part in (
            key,
            entry.symbol or "",
            entry.description or "",
            entry.part_number or "",
        )
        if part
    )


def infer_taxonomy_fields(
    entry: SymbolEntry,
) -> tuple[ShapeClass, ContextRule, CountSemantics]:
    """Return (shape_class, context_rule, count_semantics) for one legend row."""
    key = count_key(entry) or ""
    blob = _blob(entry, key)
    tag = (entry.symbol or "").strip()

    if _JHOOK_RE.search(blob):
        return (
            "jhook_run",
            ContextRule(
                requires_collinear_min=2,
                collinear_of_tag="J",
                excludes_if_matches=["letter_tag"],
            ),
            "cluster_as_one",
        )

    if _STUB_RE.search(blob):
        return (
            "conduit_stub",
            ContextRule(middle_bar_longer=True, excludes_if_matches=["letter_tag"]),
            "discrete",
        )

    if _POLE_RE.search(blob):
        return (
            "bowtie",
            ContextRule(
                requires_enclosing_square=True,
                excludes_if_matches=["filled_triangle"],
            ),
            "discrete",
        )

    if _CAMERA_RE.search(blob) and "MOUNT" not in blob.upper():
        return (
            "hourglass",
            ContextRule(
                requires_opposing_shape="triangle",
                excludes_if_matches=["filled_triangle"],
            ),
            "cluster_as_one",
        )

    if tag.upper() == "AP" or _AP_RE.search(blob):
        return (
            "filled_triangle",
            ContextRule(requires_enclosing_shape="circle"),
            "discrete",
        )

    if tag == "#" or (
        "PERMANENT LINK" in blob.upper() and "CAMERA" not in blob.upper()
    ):
        return (
            "filled_triangle",
            ContextRule(excludes_if_matches=["hourglass", "bowtie", "letter_tag"]),
            "qty_expand",
        )

    if _RACEWAY_RE.search(blob) or (
        LINEAR_GLYPH_RE.search(blob) and not _STUB_RE.search(blob)
    ):
        part = (entry.part_number or "").strip()
        if _RACEWAY_RE.search(blob) and part:
            return "rect_digit_label", ContextRule(), "discrete"
        return "other", ContextRule(), "linear_suppressed"

    if _HEX_RE.search(blob):
        return "hexagon", ContextRule(), "discrete"

    if tag and _SHORT_TAG_RE.match(tag) and tag != "#":
        return "letter_tag", ContextRule(), "discrete"

    return "other", ContextRule(), "discrete"
