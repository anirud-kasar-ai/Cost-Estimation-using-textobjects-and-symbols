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
CV_TEMPLATE_SCALES = _float_list(
    "CV_TEMPLATE_SCALES", [0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
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
LLM_MAX_IMAGE_DIM = _int("LLM_MAX_IMAGE_DIM", 2048)
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 120.0)


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

ensure_storage_dirs()
