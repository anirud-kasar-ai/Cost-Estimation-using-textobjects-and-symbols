"""Environment settings for drawing-zoom-split."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env", override=True)

STORAGE_DIR = ROOT / "storage" / "jobs"
REQUIREMENTS_DIR = ROOT / "storage" / "requirements"
TECHNICAL_SYMBOLS_DIR = ROOT / "storage" / "technical_symbols"
EXTRACTION_CACHE_DIR = ROOT / "storage" / "extraction_cache"


def ensure_storage_dirs() -> None:
    """Create output directories used by the pipeline and extraction stages."""
    for path in (STORAGE_DIR, REQUIREMENTS_DIR, TECHNICAL_SYMBOLS_DIR, EXTRACTION_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_storage_dir(path: Path) -> None:
    """Override default output root (used by batch CLI)."""
    global STORAGE_DIR
    STORAGE_DIR = Path(path)


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


HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
HF_MODEL_ID = (
    os.getenv("HF_MODEL_ID") or "meta-llama/Llama-3.2-11B-Vision-Instruct"
).strip()
HF_INFERENCE_ENDPOINT = (os.getenv("HF_INFERENCE_ENDPOINT") or "").strip()
VISION_LLM_RUNTIME = (os.getenv("VISION_LLM_RUNTIME") or "groq").strip().lower()
VISION_REQUIRE_LLM = _bool("VISION_REQUIRE_LLM", True)
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL_ID = (os.getenv("GROQ_MODEL_ID") or "qwen/qwen3.6-27b").strip()
HF_LOAD_4BIT = _bool("HF_LOAD_4BIT", True)
VISION_DEVICE = (os.getenv("VISION_DEVICE") or "auto").strip().lower()
VISION_ALLOW_CPU_MODEL = _bool("VISION_ALLOW_CPU_MODEL", False)
VISION_PDF_DPI = _int("VISION_PDF_DPI", 200)
VISION_LLAMA_MAX_SIDE = _int("VISION_LLAMA_MAX_SIDE", 1280)

# Smaller tiles = more zoom crops per wing (closer view of each plan area).
ROI_ZOOM_TILE_SIZE = _int("ROI_ZOOM_TILE_SIZE", 588)
ROI_ZOOM_OVERLAP_PCT = _float("ROI_ZOOM_OVERLAP_PCT", 0.10)

EXTRACTION_CACHE_ENABLED = _bool("EXTRACTION_CACHE_ENABLED", True)
# PDF pixmap crops for every sheet-note callout; slow on large sets.
SHEET_NOTES_GLYPH_CROP = _bool("SHEET_NOTES_GLYPH_CROP", False)
