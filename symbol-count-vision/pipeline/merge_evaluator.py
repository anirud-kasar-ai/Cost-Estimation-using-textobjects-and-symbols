"""Post-merge detection evaluator: glyph re-check + hybrid vision judge."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import config
from pipeline.cv.nms import BoundingBox, SymbolDetection
from pipeline.cv.tag_match import count_key
from pipeline.cv.template_match import (
    _build_glyph_label_map,
    _resolve_glyph_label,
    score_glyph_on_crop,
)
from pipeline.extraction.symbol_table_extractor import SymbolEntry, SymbolTableInfo

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedDetection:
    detection: SymbolDetection
    verdict: str  # accepted | rejected
    glyph_score: float | None = None
    legend_match: bool = False
    judged: bool = False
    judge_present: bool | None = None
    judge_symbol_match: bool | None = None
    judge_confidence: float | None = None
    reason: str = ""
    mapped_from: str | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        det = self.detection
        item: dict[str, Any] = {
            "symbol": det.symbol,
            "score": det.score,
            "source": det.source,
            "qty": max(1, int(det.qty)),
            "box": {
                "x1": det.box.x1,
                "y1": det.box.y1,
                "x2": det.box.x2,
                "y2": det.box.y2,
            },
            "evaluation": {
                "verdict": self.verdict,
                "glyph_score": self.glyph_score,
                "legend_match": self.legend_match,
                "judged": self.judged,
                "judge_present": self.judge_present,
                "judge_symbol_match": self.judge_symbol_match,
                "judge_confidence": self.judge_confidence,
                "reason": self.reason,
                "mapped_from": self.mapped_from,
            },
        }
        if det.tile_file:
            item["tile_file"] = det.tile_file
        return item


def _crop_detection(
    roi_image: Image.Image,
    box: BoundingBox,
    *,
    pad_px: int = 40,
) -> Image.Image:
    w, h = roi_image.size
    bw = max(1.0, box.x2 - box.x1)
    bh = max(1.0, box.y2 - box.y1)
    pad = max(pad_px, int(max(bw, bh) * 0.5))
    x1 = max(0, int(box.x1) - pad)
    y1 = max(0, int(box.y1) - pad)
    x2 = min(w, int(box.x2) + pad)
    y2 = min(h, int(box.y2) + pad)
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return roi_image.crop((x1, y1, x2, y2))


def _legend_entry_for_key(legend: SymbolTableInfo, symbol_key: str) -> SymbolEntry | None:
    for entry in legend.entries:
        if count_key(entry) == symbol_key:
            return entry
    return None


def _glyph_image_for_symbol(
    legend: SymbolTableInfo,
    glyph_dir: Path | None,
    symbol_key: str,
) -> Image.Image | None:
    entry = _legend_entry_for_key(legend, symbol_key)
    if entry and entry.symbol_image_png:
        try:
            return Image.open(BytesIO(entry.symbol_image_png)).convert("RGB")
        except OSError:
            pass
    if not glyph_dir or not glyph_dir.is_dir():
        return None
    entry_by_key = _build_glyph_label_map(legend)
    for path in sorted(glyph_dir.glob("*.png")):
        if _resolve_glyph_label(path.stem, entry_by_key) == symbol_key:
            try:
                return Image.open(path).convert("RGB")
            except OSError:
                continue
    return None


_SOURCE_TO_GUESS = {
    "triangle": "triangle_like",
    "callout_drop": "triangle_like",
    "hourglass": "hourglass_like",
    "bowtie": "bowtie_like",
    "stub": "conduit_stub_like",
    "jhook": "jhook_like",
}


def _source_shape_class(src: str) -> str | None:
    from pipeline.taxonomy.schema import SHAPE_GUESS_TO_CLASS

    guess = _SOURCE_TO_GUESS.get((src or "").strip().lower())
    if not guess:
        return None
    return SHAPE_GUESS_TO_CLASS.get(guess)


def _lookalike_keys(
    det: SymbolDetection,
    taxonomy: Any,
) -> list[str]:
    """Assigned key plus same-shape legend rows (not the full gallery)."""
    from pipeline.cv.template_match import LINEAR_GLYPH_RE

    keys: list[str] = []
    assigned = (det.symbol or "").strip()
    if assigned:
        keys.append(assigned)
    if taxonomy is None:
        return keys
    src = (det.source or "").strip().lower()
    mapped = _SOURCE_TO_GUESS.get(src)
    guesses = [mapped] if mapped else []
    seen = set(keys)
    wanted = _source_shape_class(src)
    for guess in guesses:
        for entry in taxonomy.entries_for_guess(guess):
            if not entry.key or entry.key in seen:
                continue
            if LINEAR_GLYPH_RE.search(entry.key):
                continue
            if wanted and entry.shape_class != wanted:
                continue
            seen.add(entry.key)
            keys.append(entry.key)
            if len(keys) >= 8:
                return keys
    return keys


def _rank_keys_on_crop(
    crop: Image.Image,
    keys: list[str],
    *,
    legend: SymbolTableInfo,
    glyph_dir: Path | None,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for key in keys:
        glyph = _glyph_image_for_symbol(legend, glyph_dir, key)
        if glyph is None:
            continue
        score = float(score_glyph_on_crop(crop, glyph))
        if score >= 0.02:
            ranked.append((key, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _remap_detection(
    det: SymbolDetection,
    crop: Image.Image,
    *,
    legend: SymbolTableInfo,
    glyph_dir: Path | None,
    taxonomy: Any,
) -> tuple[SymbolDetection, float | None, str | None]:
    """Confirm or relabel ``det`` using its look-alike glyphs only."""
    src = (det.source or "").strip().lower()
    if src in {"ocr", "vision_ocr"}:
        glyph = _glyph_image_for_symbol(legend, glyph_dir, det.symbol)
        if glyph is None:
            return det, None, None
        raw = float(score_glyph_on_crop(crop, glyph))
        score = None if raw < 0.02 else raw
        return det, score, None

    keys = _lookalike_keys(det, taxonomy)
    ranking = _rank_keys_on_crop(crop, keys, legend=legend, glyph_dir=glyph_dir)
    assigned = (det.symbol or "").strip()
    wanted = _source_shape_class(src)
    if wanted and taxonomy is not None:
        ranking = [
            (key, score)
            for key, score in ranking
            if key == assigned
            or (
                (entry := taxonomy.by_key(key)) is not None
                and entry.shape_class == wanted
            )
        ]
    assigned_score = 0.0
    for key, score in ranking:
        if key == assigned:
            assigned_score = score
            break
    if not ranking:
        return det, None, None

    best_key, best_score = ranking[0]
    margin = float(config.GLYPH_IDENT_MARGIN)
    mapped_from: str | None = None
    mapped = det
    if (
        best_key != assigned
        and best_score >= assigned_score + margin
        and best_score >= float(config.GLYPH_IDENT_MIN)
    ):
        qty = max(1, int(det.qty or 1)) if best_key == "#" else 1
        mapped = SymbolDetection(
            symbol=best_key,
            box=det.box,
            score=float(det.score),
            source=det.source,
            qty=qty,
            tile_file=det.tile_file,
            classify_source="glyph_validate",
        )
        mapped_from = assigned
        assigned_score = best_score
    glyph_score = assigned_score if assigned_score >= 0.02 else (
        best_score if best_score >= 0.02 else None
    )
    return mapped, glyph_score, mapped_from


def _should_judge(
    det: SymbolDetection,
    *,
    glyph_score: float | None,
    judge_sources: set[str],
) -> bool:
    """Only spend Gemini on camera/pole (or configured) sources.

    OCR / triangles used to enter this path whenever the glyph template
    failed to lock (score 0.0). That burned the judge quota on 2300 labels
    and then 429-rejected the actual cameras.
    """
    src = (det.source or "").strip().lower()
    if src not in judge_sources:
        return False
    if glyph_score is None:
        return True
    if glyph_score < config.MERGE_EVAL_GLYPH_MIN:
        return True
    return False


def evaluate_merged_detections(
    *,
    roi_image: Image.Image,
    legend: SymbolTableInfo,
    glyph_dir: Path | None,
    detections: list[SymbolDetection],
) -> tuple[list[EvaluatedDetection], dict[str, Any]]:
    """Validate merged detections on ``full_wing`` crops vs legend glyphs.

    Each box is scored against its assigned glyph and same-shape look-alikes.
    A clear winner remaps the key; a poor match can reject the box.
    """
    if not config.MERGE_EVAL_ENABLED:
        accepted = [
            EvaluatedDetection(
                detection=d,
                verdict="accepted",
                reason="evaluation disabled",
                evaluation={"verdict": "accepted", "reason": "evaluation disabled"},
            )
            for d in detections
        ]
        summary = {
            "enabled": False,
            "pre_merge": len(detections),
            "accepted": len(detections),
            "rejected": 0,
            "judged": 0,
            "rejected_by_reason": {},
        }
        return accepted, summary

    from pipeline.taxonomy.builder import taxonomy_from_legend

    taxonomy = taxonomy_from_legend(legend, glyph_dir)
    judge_sources = config.merge_eval_judge_sources()
    judged_calls = 0
    max_judge = max(0, int(config.MERGE_EVAL_MAX_JUDGE_CALLS))
    results: list[EvaluatedDetection] = []
    rejected_by_reason: dict[str, int] = {}
    remapped_n = 0

    def bump(reason: str) -> None:
        rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1

    for det in detections:
        src = (det.source or "").strip().lower()
        crop = _crop_detection(roi_image, det.box)
        mapped, glyph_score, mapped_from = _remap_detection(
            det,
            crop,
            legend=legend,
            glyph_dir=glyph_dir,
            taxonomy=taxonomy,
        )
        if mapped_from:
            remapped_n += 1
            det = mapped
            src = (det.source or "").strip().lower()
        legend_match = bool(
            glyph_score is not None and glyph_score >= config.MERGE_EVAL_GLYPH_MIN
        )
        glyph = _glyph_image_for_symbol(legend, glyph_dir, det.symbol)

        # Geometry keep only when the assigned glyph could not lock — never
        # skip look-alike scoring when artwork exists.
        if (
            glyph is None
            and glyph_score is None
            and src in {"triangle", "hourglass", "bowtie"}
            and float(det.score) >= config.MERGE_EVAL_GEOMETRY_AUTO_ACCEPT
        ):
            results.append(
                EvaluatedDetection(
                    detection=det,
                    verdict="accepted",
                    reason=f"geometry auto-accept ({src}, score={det.score:.2f})",
                    glyph_score=None,
                    legend_match=False,
                    judged=False,
                    mapped_from=mapped_from,
                    evaluation={
                        "verdict": "accepted",
                        "reason": f"geometry auto-accept ({src}, score={det.score:.2f})",
                        "mapped_from": mapped_from,
                    },
                )
            )
            continue

        verdict = "accepted"
        reason = "default accept"
        judged = False
        judge_present: bool | None = None
        judge_symbol_match: bool | None = None
        judge_confidence: float | None = None

        handled = False
        if (
            src == "hourglass"
            and glyph_score is not None
            and glyph_score >= config.MERGE_EVAL_GLYPH_MIN
            and float(det.score) >= config.MERGE_EVAL_JUDGE_MIN_SCORE
        ):
            reason = f"hourglass glyph match {glyph_score:.2f}"
            handled = True
        elif (
            src == "hourglass"
            and glyph_score is not None
            and glyph_score < config.MERGE_EVAL_GLYPH_REJECT
            and not config.llm_enabled()
        ):
            verdict = "rejected"
            reason = f"hourglass low glyph {glyph_score:.2f}"
            bump("hourglass_low_glyph")
            handled = True

        if not handled:
            short_tag = bool(re.match(r"^[A-Z]{2,3}$", (det.symbol or "").strip()))
            if (
                src == "template"
                and short_tag
                and float(det.score) >= config.CV_SHORT_TAG_TEMPLATE_THRESHOLD
            ):
                reason = f"short-tag template accept ({det.symbol}, score={det.score:.2f})"
            elif (
                glyph_score is not None
                and glyph_score >= config.MERGE_EVAL_GLYPH_MIN
                and float(det.score) >= config.MERGE_EVAL_JUDGE_MIN_SCORE
            ):
                reason = f"glyph match {glyph_score:.2f} >= {config.MERGE_EVAL_GLYPH_MIN}"
            elif (
                glyph_score is not None
                and glyph_score < config.MERGE_EVAL_GLYPH_REJECT
                and float(det.score) < config.MERGE_EVAL_JUDGE_MIN_SCORE
                and src in {"ocr", "template"}
            ):
                verdict = "rejected"
                reason = f"low glyph {glyph_score:.2f} and low CV score {det.score:.2f}"
                bump("low_glyph_cv")
            elif _should_judge(det, glyph_score=glyph_score, judge_sources=judge_sources) or (
                src == "hourglass"
                and glyph_score is not None
                and glyph_score < config.MERGE_EVAL_GLYPH_REJECT
            ):
                if judged_calls < max_judge and config.llm_enabled():
                    from pipeline.llm_verify import judge_detection_crop

                    entry = _legend_entry_for_key(legend, det.symbol)
                    judge = judge_detection_crop(crop, entry, det.symbol, glyph)
                    judged = True
                    judged_calls += 1
                    judge_present = bool(judge.get("present"))
                    judge_symbol_match = bool(judge.get("symbol_matches"))
                    judge_confidence = judge.get("confidence")
                    if judge.get("ok") and judge_present and judge_symbol_match:
                        verdict = "accepted"
                        reason = str(judge.get("reason") or "vision judge accepted")
                    elif judge.get("ok"):
                        verdict = "rejected"
                        reason = str(judge.get("reason") or "vision judge rejected")
                        bump("vision_judge")
                    else:
                        err = str(judge.get("error") or "")
                        if "429" in err:
                            max_judge = judged_calls
                        geometry = src in {"triangle", "hourglass", "bowtie"}
                        if geometry or legend_match:
                            verdict = "accepted"
                            reason = (
                                f"judge skipped ({judge.get('error')}); kept on CV"
                            )
                        else:
                            verdict = "rejected"
                            reason = f"judge skipped ({judge.get('error')}); low confidence"
                            bump("judge_skipped_reject")
                elif src in {"hourglass", "bowtie"}:
                    if float(det.score) >= config.MERGE_EVAL_GEOMETRY_AUTO_ACCEPT:
                        verdict = "accepted"
                        reason = f"{src} CV keep (judge unavailable)"
                    elif legend_match:
                        verdict = "accepted"
                        reason = f"{src} glyph match without judge ({glyph_score:.2f})"
                    else:
                        verdict = "rejected"
                        reason = f"{src} uncertain; judge cap reached or LLM disabled"
                        bump("hourglass_no_judge")
                elif legend_match:
                    verdict = "accepted"
                    reason = f"glyph match without judge ({glyph_score:.2f})"
                elif (
                    src in {"ocr", "template", "stub"}
                    and float(det.score) < config.MERGE_EVAL_JUDGE_MIN_SCORE
                    and not legend_match
                ):
                    verdict = "rejected"
                    reason = f"low CV score {det.score:.2f} without glyph lock"
                    bump("low_cv_no_glyph")
                elif float(det.score) >= 0.75 and src in {
                    "triangle",
                    "ocr",
                    "bowtie",
                    "stub",
                    "jhook",
                    "callout_drop",
                }:
                    verdict = "accepted"
                    reason = f"high CV score {det.score:.2f} (judge cap/disabled)"
                else:
                    verdict = "rejected"
                    reason = "uncertain; judge cap reached or LLM disabled"
                    bump("uncertain_no_judge")
            else:
                if (
                    src in {"ocr", "template", "stub"}
                    and float(det.score) < config.MERGE_EVAL_JUDGE_MIN_SCORE
                    and not legend_match
                ):
                    verdict = "rejected"
                    reason = f"low CV score {det.score:.2f} without glyph lock"
                    bump("low_cv_no_glyph")
                else:
                    reason = (
                        f"passed CV/glyph thresholds "
                        f"(score={det.score:.2f}, glyph={glyph_score})"
                    )

        if mapped_from:
            reason = f"mapped {mapped_from} → {det.symbol}; {reason}"
        ev = EvaluatedDetection(
            detection=det,
            verdict=verdict,
            glyph_score=glyph_score,
            legend_match=legend_match,
            judged=judged,
            judge_present=judge_present,
            judge_symbol_match=judge_symbol_match,
            judge_confidence=judge_confidence,
            reason=reason,
            mapped_from=mapped_from,
        )
        ev.evaluation = dict(ev.to_dict()["evaluation"])
        results.append(ev)

    accepted_n = sum(1 for r in results if r.verdict == "accepted")
    rejected_n = len(results) - accepted_n
    summary = {
        "enabled": True,
        "pre_merge": len(detections),
        "post_merge": len(detections),
        "accepted": accepted_n,
        "rejected": rejected_n,
        "judged": judged_calls,
        "remapped": remapped_n,
        "rejected_by_reason": rejected_by_reason,
        "judge_cap": max_judge,
    }
    return results, summary


__all__ = ["EvaluatedDetection", "evaluate_merged_detections"]
