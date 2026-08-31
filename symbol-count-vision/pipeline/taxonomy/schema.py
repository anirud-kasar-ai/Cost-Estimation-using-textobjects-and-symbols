"""Single-source-of-truth taxonomy for legend-gated symbol counting.

Look-alike / context rules live HERE (and in the JSON this module serializes).
CV context resolution and any remaining LLM prompts must import this module
rather than restating the rules in prose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

ShapeClass = Literal[
    "filled_triangle",
    "hourglass",
    "bowtie",
    "letter_tag",
    "rect_digit_label",
    "hexagon",
    "conduit_stub",
    "jhook_run",
    "other",
]

CountSemantics = Literal[
    "discrete",
    "qty_expand",
    "linear_suppressed",
    "cluster_as_one",
]

# Candidate Step-2 guesses → taxonomy shape_class.
SHAPE_GUESS_TO_CLASS: dict[str, ShapeClass] = {
    "triangle_like": "filled_triangle",
    "hourglass_like": "hourglass",
    "bowtie_like": "bowtie",
    "letter_tag": "letter_tag",
    "digit_label": "rect_digit_label",
    "hexagon_like": "hexagon",
    "conduit_stub_like": "conduit_stub",
    "jhook_like": "jhook_run",
    "other": "other",
}


@dataclass
class ContextRule:
    """Geometric context that distinguishes look-alike marks.

    Extra fields beyond the four required by the architecture brief exist so
    J-HOOK clustering and CONDUIT STUB vs letter-E can live on the taxonomy
    entry rather than as free-standing if-trees.
    """

    requires_enclosing_shape: Optional[str] = None
    requires_opposing_shape: Optional[str] = None
    requires_enclosing_square: bool = False
    excludes_if_matches: list[str] = field(default_factory=list)
    requires_collinear_min: Optional[int] = None
    collinear_of_tag: Optional[str] = None
    middle_bar_longer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_enclosing_shape": self.requires_enclosing_shape,
            "requires_opposing_shape": self.requires_opposing_shape,
            "requires_enclosing_square": bool(self.requires_enclosing_square),
            "excludes_if_matches": list(self.excludes_if_matches),
            "requires_collinear_min": self.requires_collinear_min,
            "collinear_of_tag": self.collinear_of_tag,
            "middle_bar_longer": bool(self.middle_bar_longer),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextRule:
        data = data or {}
        excludes = data.get("excludes_if_matches") or []
        return cls(
            requires_enclosing_shape=data.get("requires_enclosing_shape") or None,
            requires_opposing_shape=data.get("requires_opposing_shape") or None,
            requires_enclosing_square=bool(data.get("requires_enclosing_square")),
            excludes_if_matches=[str(x) for x in excludes],
            requires_collinear_min=(
                int(data["requires_collinear_min"])
                if data.get("requires_collinear_min") is not None
                else None
            ),
            collinear_of_tag=data.get("collinear_of_tag") or None,
            middle_bar_longer=bool(data.get("middle_bar_longer")),
        )


@dataclass
class SymbolTaxonomyEntry:
    key: str
    description: str
    glyph_crop_path: str
    shape_class: ShapeClass
    context_rule: ContextRule
    count_semantics: CountSemantics
    tag: str | None = None
    part_number: str | None = None
    # In-memory legend artwork; never serialized to JSON.
    glyph_png: bytes | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "glyph_crop_path": self.glyph_crop_path,
            "shape_class": self.shape_class,
            "context_rule": self.context_rule.to_dict(),
            "count_semantics": self.count_semantics,
            "tag": self.tag,
            "part_number": self.part_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymbolTaxonomyEntry:
        path = str(data.get("glyph_crop_path") or "")
        png: bytes | None = None
        if path:
            try:
                file_path = Path(path)
                if file_path.is_file():
                    png = file_path.read_bytes()
            except OSError:
                png = None
        return cls(
            key=str(data.get("key") or ""),
            description=str(data.get("description") or ""),
            glyph_crop_path=path,
            shape_class=data.get("shape_class") or "other",  # type: ignore[arg-type]
            context_rule=ContextRule.from_dict(data.get("context_rule") or {}),
            count_semantics=data.get("count_semantics") or "discrete",  # type: ignore[arg-type]
            tag=data.get("tag") or None,
            part_number=data.get("part_number") or None,
            glyph_png=png,
        )


@dataclass
class SymbolTaxonomy:
    entries: list[SymbolTaxonomyEntry] = field(default_factory=list)
    source: str = ""

    def keys(self) -> set[str]:
        return {e.key for e in self.entries if e.key}

    def by_key(self, key: str) -> SymbolTaxonomyEntry | None:
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None

    def by_shape_class(self, shape_class: str) -> list[SymbolTaxonomyEntry]:
        return [e for e in self.entries if e.shape_class == shape_class]

    def entries_for_guess(self, shape_class_guess: str) -> list[SymbolTaxonomyEntry]:
        mapped = SHAPE_GUESS_TO_CLASS.get(shape_class_guess, "other")
        return self.by_shape_class(mapped)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source,
            "entries": self.to_list(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SymbolTaxonomy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            SymbolTaxonomyEntry.from_dict(row)
            for row in (payload.get("entries") or [])
            if isinstance(row, dict)
        ]
        return cls(entries=entries, source=str(payload.get("source") or path))


def taxonomy_rules_prompt_block(taxonomy: SymbolTaxonomy | None) -> str:
    """Compact JSON of ContextRule rows for LLM prompts.

    Prompts must attach this block instead of restating look-alike prose.
    """
    if taxonomy is None or not taxonomy.entries:
        return "TAXONOMY_CONTEXT_RULES: []"
    rows = []
    for entry in taxonomy.entries:
        rows.append(
            {
                "key": entry.key,
                "shape_class": entry.shape_class,
                "count_semantics": entry.count_semantics,
                "context_rule": entry.context_rule.to_dict(),
                "description": entry.description,
            }
        )
    return (
        "TAXONOMY_CONTEXT_RULES (the ONLY look-alike / context rules — "
        "do not invent others):\n"
        + json.dumps(rows, ensure_ascii=False, indent=1)
    )


def taxonomy_as_dict(taxonomy: SymbolTaxonomy) -> dict[str, Any]:
    return {"source": taxonomy.source, "entries": taxonomy.to_list()}
