"""Apply taxonomy count_semantics and emit result.json-compatible payloads."""

from __future__ import annotations

from typing import Any

from pipeline.cv.nms import SymbolDetection
from pipeline.taxonomy.schema import SymbolTaxonomy


def apply_count_semantics(
    detections: list[SymbolDetection],
    taxonomy: SymbolTaxonomy,
) -> list[SymbolDetection]:
    """Mutate qty according to the taxonomy entry; drop linear-suppressed marks."""
    kept: list[SymbolDetection] = []
    for det in detections:
        entry = taxonomy.by_key(det.symbol)
        if entry is None:
            continue
        if entry.count_semantics == "linear_suppressed":
            continue
        if entry.count_semantics in {"discrete", "cluster_as_one"}:
            det.qty = 1
        elif entry.count_semantics == "qty_expand":
            det.qty = max(1, int(det.qty or 1))
        kept.append(det)
    return kept


def aggregate_counts(
    detections: list[SymbolDetection],
) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    mark_counts: dict[str, int] = {}
    for det in detections:
        mark_counts[det.symbol] = mark_counts.get(det.symbol, 0) + 1
        counts[det.symbol] = counts.get(det.symbol, 0) + max(1, int(det.qty))
    return counts, mark_counts


def detection_trace(det: SymbolDetection) -> dict[str, Any]:
    return {
        "symbol": det.symbol,
        "score": det.score,
        "source": det.source,
        "qty": max(1, int(det.qty)),
        "classify_source": det.classify_source,
        "box": {
            "x1": det.box.x1,
            "y1": det.box.y1,
            "x2": det.box.x2,
            "y2": det.box.y2,
        },
        **({"tile_file": det.tile_file} if det.tile_file else {}),
    }
