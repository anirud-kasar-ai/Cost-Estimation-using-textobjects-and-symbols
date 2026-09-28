"""Match YOLO class names to OCR legend entries via explicit annotation map."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.legend_ocr import LegendEntry
from pipeline.yolo_class_map import resolve_class_map


def _norm(text: str) -> str:
    t = (text or "").upper()
    t = t.replace("REACEWAY", "RACEWAY")  # common training typo
    t = t.replace("ALARAM", "ALARM")
    t = t.replace("PANNEL", "PANEL")
    t = t.replace("CABINATE", "CABINET")
    t = t.replace("SINGEL", "SINGLE")
    t = t.replace("ACESS", "ACCESS")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokens(text: str) -> set[str]:
    stop = {
        "THE",
        "OF",
        "AND",
        "OR",
        "A",
        "AN",
        "WITH",
        "FOR",
        "TO",
        "IN",
        "ON",
        "WM",
        "DTL",
        "QTY",
        "CAT6A",
        "CAT6",
        "LOCATION",
        "DROP",
    }
    return {tok for tok in _norm(text).split() if len(tok) > 1 and tok not in stop}


@dataclass(frozen=True)
class MatchResult:
    legend_number: int
    score: float
    reason: str
    count_weight: int = 1


def _score_phrase_to_entry(phrase: str, entry: LegendEntry) -> MatchResult | None:
    """Score how well a mapped legend phrase matches a legend row.

    Compares against description, detail, and raw_line. Detail-only matches are
    rejected so shared DTL refs cannot bind unrelated rows.
    """
    cn = _norm(phrase)
    desc = _norm(entry.description)
    full = _norm(f"{entry.description} {entry.detail_ref or ''}".strip())
    raw = _norm(getattr(entry, "raw_line", "") or "")
    if not cn or not desc:
        return None

    # Exact / substring on description, description+detail, or original OCR line
    for target in (desc, full, raw):
        if not target:
            continue
        if cn == target:
            return MatchResult(entry.number, 1.0, "map_exact")
        if cn in target or target in cn:
            dt = _tokens(entry.description) or _tokens(raw)
            ct = _tokens(phrase)
            if dt and ct and (dt & ct):
                return MatchResult(entry.number, 0.95, "map_substring")

    ct = _tokens(phrase)
    dt = _tokens(entry.description) or _tokens(raw)
    if not ct or not dt:
        return None
    inter = ct & dt
    if not inter:
        return None

    phrase_cov = len(inter) / max(1, len(ct))
    legend_cov = len(inter) / max(1, len(dt))
    jacc = len(inter) / max(1, len(ct | dt))

    if legend_cov >= 0.75 and len(inter) >= 1:
        return MatchResult(entry.number, min(0.94, 0.55 + 0.4 * legend_cov), "map_legend_cov")
    if phrase_cov >= 0.7 and len(inter) >= 1:
        return MatchResult(entry.number, min(0.93, 0.55 + 0.4 * phrase_cov), "map_tokens")
    if jacc >= 0.45 and len(inter) >= 2:
        return MatchResult(entry.number, min(0.88, 0.5 + 0.35 * jacc), "map_tokens")
    if len(inter) >= 3 and (phrase_cov >= 0.4 or legend_cov >= 0.5):
        return MatchResult(entry.number, 0.8, "map_key_tokens")
    return None


def match_class_to_legend(
    class_name: str,
    legend: list[LegendEntry],
    *,
    min_score: float = 0.55,
) -> MatchResult | None:
    """Match YOLO class → legend number using the annotation map only.

    Unmapped classes return None. Tries every alternate legend phrase, plus the
    raw YOLO class name as a short fallback (helps when the PDF wording differs).
    """
    mapped = resolve_class_map(class_name)
    if mapped is None:
        return None

    phrases: list[str] = list(mapped.legend_phrases)
    if class_name and class_name not in phrases:
        phrases.append(class_name)
    for p in list(mapped.legend_phrases):
        core = re.split(r"[,–—\-]", p, maxsplit=1)[0].strip()
        if core and core not in phrases:
            phrases.append(core)

    # Prefer matching SURFACE RACEWAY WM5400 → legend row whose detail is WM5400
    class_part = None
    m_part = re.search(r"\bWM\s*(2300|5400|5500)\b", _norm(class_name))
    if m_part:
        class_part = "WM" + m_part.group(1)

    best: MatchResult | None = None
    for entry in legend:
        for phrase in phrases:
            m = _score_phrase_to_entry(phrase, entry)
            if m is None:
                continue
            score = m.score
            detail_n = _norm(entry.detail_ref or "").replace(" ", "")
            if class_part:
                if class_part in detail_n:
                    score = min(1.0, score + 0.08)
                elif "WM2300" in detail_n or "WM5400" in detail_n or "WM5500" in detail_n:
                    # Same description, wrong raceway part — demote
                    score = max(0.0, score - 0.1)
            weighted = MatchResult(
                legend_number=m.legend_number,
                score=score,
                reason=m.reason,
                count_weight=mapped.count_weight,
            )
            if best is None or weighted.score > best.score:
                best = weighted
    if best is None or best.score < min_score:
        return None
    return best


def attach_legend_numbers(
    class_name: str,
    legend: list[LegendEntry],
    *,
    min_score: float = 0.55,
) -> int | None:
    m = match_class_to_legend(class_name, legend, min_score=min_score)
    return None if m is None else m.legend_number


def attach_legend_match(
    class_name: str,
    legend: list[LegendEntry],
    *,
    min_score: float = 0.55,
) -> MatchResult | None:
    """Full match including count_weight (e.g. CONDUIT STUB 2 → weight 2)."""
    return match_class_to_legend(class_name, legend, min_score=min_score)
