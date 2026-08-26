from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env", override=True)

_sibling_env = ROOT.parent / "vision-extraction" / ".env"
if _sibling_env.is_file():
    from dotenv import dotenv_values

    for _key, _val in dotenv_values(_sibling_env).items():
        if not _key or _val is None:
            continue
        if not (os.getenv(_key) or "").strip():
            os.environ[_key] = str(_val)


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
SYMBOL_COUNT_NMS_IOU = _float("SYMBOL_COUNT_NMS_IOU", 0.5)
SYMBOL_COUNT_SYNC_JOBS = _bool("SYMBOL_COUNT_SYNC_JOBS", False)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

# Advanced mode: vision LLM verifies/refines the CV counts.
# LLM_PROVIDER: gemini | groq  (gemini preferred for free-tier vision testing)
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "groq") or "groq").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
).strip().rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b").strip()
GROQ_API_URL = os.getenv(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
).strip()
SYMBOL_COUNT_USE_LLM = _bool("SYMBOL_COUNT_USE_LLM", True)
SYMBOL_COUNT_USE_VISION_OCR = _bool("SYMBOL_COUNT_USE_VISION_OCR", True)
VISION_OCR_EXPECTED_CONTEXT = (
    os.getenv(
        "VISION_OCR_EXPECTED_CONTEXT",
        "Low-Voltage Callouts, Equipment Tags, Raceway Dimensions, Room Labels",
    )
    or "Low-Voltage Callouts, Equipment Tags, Raceway Dimensions, Room Labels"
).strip()
LLM_MAX_IMAGE_DIM = _int("LLM_MAX_IMAGE_DIM", 2048)
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 120.0)

# Post-merge evaluation (folder jobs): glyph re-check + vision judge on flagged hits.
MERGE_EVAL_ENABLED = _bool("MERGE_EVAL_ENABLED", True)
MERGE_EVAL_GLYPH_MIN = _float("MERGE_EVAL_GLYPH_MIN", 0.55)
MERGE_EVAL_GLYPH_REJECT = _float("MERGE_EVAL_GLYPH_REJECT", 0.45)
MERGE_EVAL_JUDGE_MIN_SCORE = _float("MERGE_EVAL_JUDGE_MIN_SCORE", 0.65)
MERGE_EVAL_JUDGE_SOURCES = (
    os.getenv("MERGE_EVAL_JUDGE_SOURCES", "ocr,template,vision_ocr") or "ocr,template,vision_ocr"
).strip().lower()
MERGE_EVAL_MAX_JUDGE_CALLS = _int("MERGE_EVAL_MAX_JUDGE_CALLS", 80)
MERGE_EVAL_GEOMETRY_AUTO_ACCEPT = _float("MERGE_EVAL_GEOMETRY_AUTO_ACCEPT", 0.80)

# Folder zoom jobs: warn when zooms_manifest.json is missing; one post-merge LLM verify.
SYMBOL_COUNT_REQUIRE_MANIFEST_WARN = _bool("SYMBOL_COUNT_REQUIRE_MANIFEST_WARN", True)
FOLDER_VISION_COUNT_VERIFY = _bool("FOLDER_VISION_COUNT_VERIFY", True)
# Per-tile Gemini vision OCR + symbol locate on folder zoom jobs (costly; accurate).
FOLDER_USE_VISION_OCR = _bool("FOLDER_USE_VISION_OCR", True)
CALLOUT_DROP_RADIUS_PX = _float("CALLOUT_DROP_RADIUS_PX", 80.0)
FOLDER_VISION_429_RETRIES = _int("FOLDER_VISION_429_RETRIES", 1)
FOLDER_VISION_429_BACKOFF_S = _float("FOLDER_VISION_429_BACKOFF_S", 4.0)


def merge_eval_judge_sources() -> set[str]:
    return {s.strip() for s in MERGE_EVAL_JUDGE_SOURCES.split(",") if s.strip()}


def llm_provider() -> str:
    if LLM_PROVIDER in {"gemini", "groq"}:
        return LLM_PROVIDER
    return "groq"


def llm_model() -> str:
    return GEMINI_MODEL if llm_provider() == "gemini" else GROQ_MODEL


def llm_enabled() -> bool:
    if not SYMBOL_COUNT_USE_LLM:
        return False
    if llm_provider() == "gemini":
        return bool(GEMINI_API_KEY)
    return bool(GROQ_API_KEY)


def vision_ocr_enabled() -> bool:
    return SYMBOL_COUNT_USE_VISION_OCR and llm_enabled()

ensure_storage_dirs()
