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


def _should_judge(
    det: SymbolDetection,
    *,
    glyph_score: float | None,
    judge_sources: set[str],
) -> bool:
    src = (det.source or "").strip().lower()
    if src in judge_sources:
        return True
    if glyph_score is None:
        return True
    if det.score < config.MERGE_EVAL_JUDGE_MIN_SCORE:
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
    """Validate merged detections on ``full_wing`` crops vs legend glyphs."""
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

    judge_sources = config.merge_eval_judge_sources()
    judged_calls = 0
    max_judge = max(0, int(config.MERGE_EVAL_MAX_JUDGE_CALLS))
    results: list[EvaluatedDetection] = []
    rejected_by_reason: dict[str, int] = {}

    def bump(reason: str) -> None:
        rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1

    for det in detections:
        crop = _crop_detection(roi_image, det.box)
        glyph = _glyph_image_for_symbol(legend, glyph_dir, det.symbol)
        glyph_score: float | None = None
        legend_match = False
        if glyph is not None:
            glyph_score = float(score_glyph_on_crop(crop, glyph))
            legend_match = glyph_score >= config.MERGE_EVAL_GLYPH_MIN

        src = (det.source or "").strip().lower()
        verdict = "accepted"
        reason = "default accept"
        judged = False
        judge_present: bool | None = None
        judge_symbol_match: bool | None = None
        judge_confidence: float | None = None

        handled = False
        if (
            src == "triangle"
            and float(det.score) >= config.MERGE_EVAL_GEOMETRY_AUTO_ACCEPT
        ):
            reason = f"geometry auto-accept ({src}, score={det.score:.2f})"
            handled = True
        elif (
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
                        if legend_match or float(det.score) >= 0.85:
                            verdict = "accepted"
                            reason = f"judge skipped ({judge.get('error')}); kept on CV/glyph"
                        else:
                            verdict = "rejected"
                            reason = f"judge skipped ({judge.get('error')}); low confidence"
                            bump("judge_skipped_reject")
                elif src == "hourglass":
                    verdict = "rejected"
                    reason = "hourglass uncertain; judge cap reached or LLM disabled"
                    bump("hourglass_no_judge")
                elif legend_match:
                    verdict = "accepted"
                    reason = f"glyph match without judge ({glyph_score:.2f})"
                elif float(det.score) >= 0.85 and src in {"triangle", "ocr"}:
                    verdict = "accepted"
                    reason = f"high CV score {det.score:.2f} (judge cap/disabled)"
                else:
                    verdict = "rejected"
                    reason = "uncertain; judge cap reached or LLM disabled"
                    bump("uncertain_no_judge")
            else:
                reason = f"passed CV/glyph thresholds (score={det.score:.2f}, glyph={glyph_score})"

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
        "rejected_by_reason": rejected_by_reason,
        "judge_cap": max_judge,
    }
    return results, summary


__all__ = ["EvaluatedDetection", "evaluate_merged_detections"]
