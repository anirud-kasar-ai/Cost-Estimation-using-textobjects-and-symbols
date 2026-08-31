from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env", override=True)

for _sibling in (
    ROOT.parent / "dual-pathway-drawing-split" / ".env",
    ROOT.parent / "drawing-zoom-split" / ".env",
):
    if not _sibling.is_file():
        continue
    from dotenv import dotenv_values

    for _key, _val in dotenv_values(_sibling).items():
        if not _key or _val is None:
            continue
        if not (os.getenv(_key) or "").strip():
            os.environ[_key] = str(_val)
    break


STORAGE_DIR = ROOT / "storage" / "jobs"
RESULTS_DIR = ROOT / "storage"
GLYPHS_DIR = "07_glyphs"


def ensure_storage_dirs() -> None:
    for path in (STORAGE_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _float_list(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name)
    if not raw:
        return default
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out or default


CV_OCR_MIN_CONF = _float("CV_OCR_MIN_CONF", 60.0)
CV_OCR_MIN_HEIGHT = _float("CV_OCR_MIN_HEIGHT", 8.0)
CV_OCR_MAX_HEIGHT = _float("CV_OCR_MAX_HEIGHT", 40.0)
CV_TEMPLATE_THRESHOLD = _float("CV_TEMPLATE_THRESHOLD", 0.72)
CV_SHORT_TAG_TEMPLATE_THRESHOLD = _float("CV_SHORT_TAG_TEMPLATE_THRESHOLD", 0.58)
CV_TEMPLATE_SCALES = _float_list(
    "CV_TEMPLATE_SCALES", [0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.25, 1.5, 2.0]
)
CV_TEMPLATE_MIN_SIZE = _int("CV_TEMPLATE_MIN_SIZE", 12)
CV_TITLE_BAND_PCT = _float("CV_TITLE_BAND_PCT", 0.12)
# Tighter default: CV contour boxes are exact, so 0.3 IoU is enough to merge
# the same mark across overlapping tiles without swallowing neighbours.
SYMBOL_COUNT_NMS_IOU = _float("SYMBOL_COUNT_NMS_IOU", 0.3)
SYMBOL_COUNT_SYNC_JOBS = _bool("SYMBOL_COUNT_SYNC_JOBS", False)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

# Vision classify + count-verify: Gemini only (no Groq / Llama).
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Pin Flash 3.7 (higher quota than Pro). 404 only → 3.5-flash.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip()
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash"
).strip()
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
).strip().rstrip("/")
SYMBOL_COUNT_USE_LLM = _bool("SYMBOL_COUNT_USE_LLM", True)
SYMBOL_COUNT_USE_VISION_OCR = _bool("SYMBOL_COUNT_USE_VISION_OCR", True)
VISION_OCR_EXPECTED_CONTEXT = (
    os.getenv(
        "VISION_OCR_EXPECTED_CONTEXT",
        "Low-Voltage Callouts, Equipment Tags, Raceway Dimensions, Room Labels",
    )
    or "Low-Voltage Callouts, Equipment Tags, Raceway Dimensions, Room Labels"
).strip()
LLM_MAX_IMAGE_DIM = _int("LLM_MAX_IMAGE_DIM", 1280)
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 30.0)

# Post-merge evaluation (folder jobs): glyph re-check + vision judge on flagged hits.
MERGE_EVAL_ENABLED = _bool("MERGE_EVAL_ENABLED", True)
MERGE_EVAL_GLYPH_MIN = _float("MERGE_EVAL_GLYPH_MIN", 0.55)
MERGE_EVAL_GLYPH_REJECT = _float("MERGE_EVAL_GLYPH_REJECT", 0.45)
MERGE_EVAL_JUDGE_MIN_SCORE = _float("MERGE_EVAL_JUDGE_MIN_SCORE", 0.65)
MERGE_EVAL_JUDGE_SOURCES = (
    os.getenv("MERGE_EVAL_JUDGE_SOURCES", "hourglass,bowtie")
    or "hourglass,bowtie"
).strip().lower()
# Pro judges cameras/poles after merge. 0 = CV-only (faster, more FPs).
MERGE_EVAL_MAX_JUDGE_CALLS = _int("MERGE_EVAL_MAX_JUDGE_CALLS", 16)
MERGE_EVAL_GEOMETRY_AUTO_ACCEPT = _float("MERGE_EVAL_GEOMETRY_AUTO_ACCEPT", 0.80)

# Folder zoom jobs: warn when zooms_manifest.json is missing.
SYMBOL_COUNT_REQUIRE_MANIFEST_WARN = _bool("SYMBOL_COUNT_REQUIRE_MANIFEST_WARN", True)
# Post-merge count-verify must not invent boxes; default off (CV counts win).
FOLDER_VISION_COUNT_VERIFY = _bool("FOLDER_VISION_COUNT_VERIFY", False)
# Per-tile vision locate is removed. Ambiguous crops use vision_classify instead.
FOLDER_USE_VISION_OCR = _bool("FOLDER_USE_VISION_OCR", False)
VISION_CLASSIFY_MARGIN_PX = _int("VISION_CLASSIFY_MARGIN_PX", 8)
VISION_OCR_FALLBACK_CONF = _float("VISION_OCR_FALLBACK_CONF", 60.0)
# Ambiguous look-alikes only (camera vs pole vs #). 0 = CV labels only.
VISION_CLASSIFY_MAX_PER_TILE = _int("VISION_CLASSIFY_MAX_PER_TILE", 4)
# Parallel OpenCV/OCR across zoom tiles. Gemini calls stay serialized.
TILE_DETECT_WORKERS = _int("TILE_DETECT_WORKERS", 4)
# Upright OCR only by default. "0,90,270" restores rotated-tag search (3× slower).
CV_OCR_ROTATIONS = os.getenv("CV_OCR_ROTATIONS", "0").strip() or "0"
# Candidate-key glyphs on classify; a few discrete glyphs on count-verify.
VISION_DETECT_MAX_REFS = _int("VISION_DETECT_MAX_REFS", 4)
# Max vision-detect box side as a fraction of the shorter image side; larger
# boxes are shrunk around their center so they cannot suppress neighbours.
VISION_DETECT_MAX_BOX_FRAC = _float("VISION_DETECT_MAX_BOX_FRAC", 0.15)
CALLOUT_DROP_RADIUS_PX = _float("CALLOUT_DROP_RADIUS_PX", 80.0)
GLYPH_IDENT_MIN = _float("GLYPH_IDENT_MIN", 0.55)
GLYPH_IDENT_MARGIN = _float("GLYPH_IDENT_MARGIN", 0.08)
FOLDER_VISION_429_RETRIES = _int("FOLDER_VISION_429_RETRIES", 0)
FOLDER_VISION_429_BACKOFF_S = _float("FOLDER_VISION_429_BACKOFF_S", 0.0)
# After a 429, skip remaining Gemini calls for this many seconds (CV keeps going).
GEMINI_QUOTA_SKIP_S = _float("GEMINI_QUOTA_SKIP_S", 900.0)


def ocr_rotations() -> tuple[int, ...]:
    allowed = {0, 90, 180, 270}
    out: list[int] = []
    for part in CV_OCR_ROTATIONS.split(","):
        part = part.strip()
        if not part.isdigit():
            continue
        val = int(part)
        if val in allowed and val not in out:
            out.append(val)
    return tuple(out) or (0,)


def merge_eval_judge_sources() -> set[str]:
    return {s.strip() for s in MERGE_EVAL_JUDGE_SOURCES.split(",") if s.strip()}


def llm_provider() -> str:
    return "gemini"


def llm_model() -> str:
    return GEMINI_MODEL


def llm_enabled() -> bool:
    return bool(SYMBOL_COUNT_USE_LLM) and bool(GEMINI_API_KEY)


def vision_ocr_enabled() -> bool:
    return SYMBOL_COUNT_USE_VISION_OCR and llm_enabled()

ensure_storage_dirs()
