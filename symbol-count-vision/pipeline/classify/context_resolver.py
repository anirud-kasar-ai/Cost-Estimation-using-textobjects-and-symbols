"""Deterministic geometric context → legend key.

Look-alike decisions come only from SymbolTaxonomyEntry.context_rule.
"""

from __future__ import annotations

from pipeline.cv.nms import BoundingBox
from pipeline.detect.candidates import Candidate
from pipeline.taxonomy.schema import ContextRule, SymbolTaxonomy, SymbolTaxonomyEntry


def _rule_accepts(cand: Candidate, rule: ContextRule) -> bool:
    if rule.requires_enclosing_shape == "circle" and not cand.has_enclosing_circle:
        return False
    if rule.requires_opposing_shape == "triangle" and not cand.has_opposing_triangle:
        return False
    if rule.requires_enclosing_square and not cand.has_enclosing_square:
        return False
    if rule.middle_bar_longer and cand.shape_class_guess != "conduit_stub_like":
        return False
    if rule.requires_collinear_min and cand.shape_class_guess not in {
        "jhook_like",
        "letter_tag",
    }:
        # Cluster pass handles this; a lone letter_tag is not a hook.
        if cand.shape_class_guess != "jhook_like":
            return False
    return True


def _standalone_triangle(cand: Candidate) -> bool:
    return (
        cand.shape_class_guess == "triangle_like"
        and not cand.has_enclosing_circle
        and not cand.has_opposing_triangle
        and not cand.has_enclosing_square
    )


def _entry_fits_triangle(cand: Candidate, entry: SymbolTaxonomyEntry) -> bool:
    rule = entry.context_rule
    if cand.has_enclosing_circle:
        return rule.requires_enclosing_shape == "circle"
    if cand.has_opposing_triangle:
        return rule.requires_opposing_shape == "triangle"
    if cand.has_enclosing_square:
        return bool(rule.requires_enclosing_square)
    # Bare triangle: only keys that do not require extra context.
    return (
        not rule.requires_enclosing_shape
        and not rule.requires_opposing_shape
        and not rule.requires_enclosing_square
    )


def _match_ocr_hint(cand: Candidate, taxonomy: SymbolTaxonomy) -> list[str]:
    hint = (cand.extras or {}).get("legend_hint")
    if hint and taxonomy.by_key(str(hint)):
        return [str(hint)]
    alias = (cand.ocr_text or "").strip().upper()
    if not alias:
        return []
    hits: list[str] = []
    for entry in taxonomy.entries:
        tag = (entry.tag or "").strip().upper()
        part = (entry.part_number or "").strip().upper()
        if tag and tag == alias:
            hits.append(entry.key)
            continue
        if part and (part == alias or part.endswith(alias)):
            hits.append(entry.key)
            continue
        if entry.key.upper() == alias:
            hits.append(entry.key)
    return hits


def _mark_resolved(cand: Candidate, key: str, *, source: str = "context_rule") -> None:
    cand.resolved_key = key
    cand.status = "resolved"
    cand.candidate_keys = [key]
    cand.classify_source = source


def _mark_ambiguous(cand: Candidate, keys: list[str]) -> None:
    uniq = []
    for k in keys:
        if k and k not in uniq:
            uniq.append(k)
    cand.candidate_keys = uniq[:3]
    if len(cand.candidate_keys) == 1:
        _mark_resolved(cand, cand.candidate_keys[0])
        return
    cand.resolved_key = None
    cand.status = "ambiguous"
    cand.classify_source = "ambiguous"


def _center(c: Candidate) -> tuple[float, float]:
    return (c.cx, c.cy)


def _x_gap(a: Candidate, b: Candidate) -> float:
    if a.bbox.x2 < b.bbox.x1:
        return b.bbox.x1 - a.bbox.x2
    if b.bbox.x2 < a.bbox.x1:
        return a.bbox.x1 - b.bbox.x2
    return 0.0


def _cluster_jhooks(
    candidates: list[Candidate],
    taxonomy: SymbolTaxonomy,
) -> list[Candidate]:
    """Row of collinear J's → one J-HOOK (ContextRule.requires_collinear_min)."""
    hook_entries = [
        e
        for e in taxonomy.entries
        if e.context_rule.requires_collinear_min
        and (e.context_rule.collinear_of_tag or "J").upper() == "J"
    ]
    if not hook_entries:
        return candidates
    hook = hook_entries[0]
    tag = (hook.context_rule.collinear_of_tag or "J").upper()
    min_n = int(hook.context_rule.requires_collinear_min or 2)

    j_hits = [
        c
        for c in candidates
        if c.shape_class_guess == "letter_tag"
        and (c.ocr_text or "").strip().upper() == tag
    ]
    others = [c for c in candidates if c not in j_hits]
    if len(j_hits) < min_n:
        return candidates

    ordered = sorted(j_hits, key=lambda d: (_center(d)[1], _center(d)[0]))
    used: set[int] = set()
    out = list(others)
    for i, anchor in enumerate(ordered):
        if i in used:
            continue
        _acy = _center(anchor)[1]
        ah = max(4.0, anchor.bbox.y2 - anchor.bbox.y1)
        cluster = [anchor]
        used.add(i)
        grew = True
        while grew:
            grew = False
            for j, cand in enumerate(ordered):
                if j in used:
                    continue
                cy = _center(cand)[1]
                if abs(cy - _acy) > max(14.0, ah * 1.25):
                    continue
                max_gap = max(80.0, max(d.bbox.x2 - d.bbox.x1 for d in cluster) * 6.0)
                if any(_x_gap(m, cand) <= max_gap for m in cluster):
                    cluster.append(cand)
                    used.add(j)
                    grew = True
        if len(cluster) >= min_n:
            merged = Candidate(
                tile_id=anchor.tile_id,
                bbox=BoundingBox(
                    x1=min(d.bbox.x1 for d in cluster),
                    y1=min(d.bbox.y1 for d in cluster),
                    x2=max(d.bbox.x2 for d in cluster),
                    y2=max(d.bbox.y2 for d in cluster),
                ),
                shape_class_guess="jhook_like",
                score=max(float(d.score) for d in cluster),
                source="jhook",
            )
            _mark_resolved(merged, hook.key)
            out.append(merged)
        else:
            out.append(anchor)
    return out


def resolve_candidates(
    candidates: list[Candidate],
    taxonomy: SymbolTaxonomy,
) -> list[Candidate]:
    """Apply ContextRule. Output: resolved (high conf) or ambiguous (2–3 keys)."""
    valid = taxonomy.keys()
    clustered = _cluster_jhooks(candidates, taxonomy)
    resolved: list[Candidate] = []

    for cand in clustered:
        if cand.status == "resolved" and cand.resolved_key in valid:
            resolved.append(cand)
            continue

        # Triangle members of an hourglass/bowtie are not standalone drops.
        if cand.shape_class_guess == "triangle_like" and (
            cand.has_opposing_triangle or cand.has_enclosing_square
        ):
            # Counted via the parent hourglass_like / bowtie_like candidate.
            continue

        matches: list[SymbolTaxonomyEntry] = []
        if cand.shape_class_guess == "triangle_like":
            for entry in taxonomy.by_shape_class("filled_triangle"):
                if _entry_fits_triangle(cand, entry):
                    matches.append(entry)
        else:
            for entry in taxonomy.entries_for_guess(cand.shape_class_guess):
                if _rule_accepts(cand, entry.context_rule):
                    matches.append(entry)

        ocr_keys = _match_ocr_hint(cand, taxonomy)
        if ocr_keys and cand.shape_class_guess in {"letter_tag", "digit_label"}:
            ocr_entries = [taxonomy.by_key(k) for k in ocr_keys]
            matches = [e for e in ocr_entries if e is not None]

        keys = [e.key for e in matches if e.key in valid]
        if cand.shape_class_guess == "triangle_like" and _standalone_triangle(cand):
            # Prefer qty_expand '#' when the triangle is truly standalone.
            hash_keys = [
                e.key
                for e in matches
                if e.count_semantics == "qty_expand" or e.key == "#"
            ]
            if len(hash_keys) == 1:
                keys = hash_keys

        if len(keys) == 1:
            _mark_resolved(cand, keys[0])
        elif len(keys) >= 2:
            _mark_ambiguous(cand, keys)
        elif ocr_keys:
            _mark_ambiguous(cand, [k for k in ocr_keys if k in valid] or ocr_keys)
        else:
            # Fail toward CV shape guess: keep as ambiguous against same-class keys.
            class_keys = [e.key for e in taxonomy.entries_for_guess(cand.shape_class_guess)]
            if class_keys:
                _mark_ambiguous(cand, class_keys)
            else:
                cand.status = "ambiguous"
                cand.candidate_keys = []
                cand.classify_source = "unmatched"
        resolved.append(cand)

    return resolved
