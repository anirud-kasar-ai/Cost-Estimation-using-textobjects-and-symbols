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
    for path in (
        STORAGE_DIR,
        REQUIREMENTS_DIR,
        TECHNICAL_SYMBOLS_DIR,
        EXTRACTION_CACHE_DIR,
        PRICING_DIR,
        ROOT / "model",
    ):
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
# 0.25 => ~147px overlap at 588px tiles: >=1.5x typical symbol size (~63-80px)
# so a symbol at a tile boundary is always whole in at least one tile.
ROI_ZOOM_OVERLAP_PCT = _float("ROI_ZOOM_OVERLAP_PCT", 0.25)

EXTRACTION_CACHE_ENABLED = _bool("EXTRACTION_CACHE_ENABLED", True)
# Extract sheet notes (notes PDF + per-diagram sidecars). Off by default —
# not needed for the symbol count / invoice flow.
SHEET_NOTES_ENABLED = _bool("SHEET_NOTES_ENABLED", False)
# PDF pixmap crops for every sheet-note callout; slow on large sets.
SHEET_NOTES_GLYPH_CROP = _bool("SHEET_NOTES_GLYPH_CROP", False)

# YOLO symbol detection (on ROI zoom tiles after legend extract + wing crop)
# Always use whatever weights live in drawing-zoom-split/model/ — any *.pt file
# dropped there is picked up automatically (newest one if there are several).
_MODEL_DIR = ROOT / "model"
YOLO_MODEL_DIR = _MODEL_DIR
_SIBLING_YOLO = ROOT.parent / "legend-ocr-count" / "model" / "symbol_detector_best.pt"


def _local_yolo_weights():
    if not _MODEL_DIR.is_dir():
        return None
    candidates = sorted(
        _MODEL_DIR.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


_yolo_env = (os.getenv("YOLO_MODEL_PATH") or "").strip()
_local_yolo = _local_yolo_weights()
if _yolo_env:
    YOLO_MODEL_PATH = Path(_yolo_env)
elif _local_yolo is not None:
    YOLO_MODEL_PATH = _local_yolo
elif _SIBLING_YOLO.is_file():
    # Only if the local model/ folder is empty
    YOLO_MODEL_PATH = _SIBLING_YOLO
else:
    YOLO_MODEL_PATH = _MODEL_DIR / "symbol_detector_best.pt"

YOLO_ENABLED = _bool("YOLO_ENABLED", True)
YOLO_CONF = _float("YOLO_CONF", 0.35)
YOLO_IOU = _float("YOLO_IOU", 0.45)
YOLO_IMGSZ = _int("YOLO_IMGSZ", 1280)
YOLO_KEEP_OBJECT_CLASS = _bool("YOLO_KEEP_OBJECT_CLASS", False)
YOLO_MATCH_MIN_SCORE = _float("YOLO_MATCH_MIN_SCORE", 0.55)
YOLO_NMS_IOU = _float("YOLO_NMS_IOU", 0.35)
YOLO_CROSS_NMS_IOU = _float("YOLO_CROSS_NMS_IOU", 0.50)
# Keep below the smallest symbol width: touching cluster members have centers
# one symbol-width apart, and a larger floor merges them into one count.
YOLO_NMS_CENTER_PX = _float("YOLO_NMS_CENTER_PX", 8.0)
# Second "rescue" pass: re-run each tile at lower conf (+ optional TTA) to
# recover symbols the primary threshold drops. Rescue boxes overlapping a
# primary detection are removed by NMS, so nothing is counted twice.
YOLO_RESCUE_ENABLED = _bool("YOLO_RESCUE_ENABLED", True)
# 0.25 chosen by eval sweep (scripts/tune_thresholds.py): keeps ~half of the
# rescue recall gain at roughly a third of its false-positive cost; the text
# mask + legend gate remove most of the remainder in production.
YOLO_RESCUE_CONF = _float("YOLO_RESCUE_CONF", 0.25)
YOLO_RESCUE_TTA = _bool("YOLO_RESCUE_TTA", True)
# Per-class minimum-confidence overrides (JSON {"CLASS NAME": 0.5, ...}),
# derived from eval sweeps (scripts/tune_thresholds.py at repo root). Classes
# not listed keep the pass floors. Raises the bar for classes the model is
# noisy on, cutting false detections without touching reliable classes.
YOLO_CLASS_CONF_PATH = Path(
    os.getenv("YOLO_CLASS_CONF_PATH", str(_MODEL_DIR / "class_conf_overrides.json"))
)
TILE_OVERLAP_PCT = _float("TILE_OVERLAP_PCT", ROI_ZOOM_OVERLAP_PCT)
# Text mask: drop letter-sized detections centered on PDF text words, so
# labels like "RESTORATION" are never counted as symbols. Words shorter than
# TEXT_MASK_MIN_WORD_CHARS (e.g. the "R"/"W" reference boxes) are ignored.
TEXT_MASK_ENABLED = _bool("TEXT_MASK_ENABLED", True)
TEXT_MASK_MIN_WORD_CHARS = _int("TEXT_MASK_MIN_WORD_CHARS", 2)
TEXT_MASK_PAD_PX = _float("TEXT_MASK_PAD_PX", 2.0)
# YOLO boxes on text are often 2-3x taller than the letters themselves, so
# allow up to 3x the text-line height before a hit stops counting as
# "letter-sized". Real symbols rarely have their center ON a text word.
TEXT_MASK_HEIGHT_RATIO = _float("TEXT_MASK_HEIGHT_RATIO", 3.0)
# Geometric text-line detector (for SHX stroke-font labels with no PDF text
# layer): a run of TEXT_MASK_CV_MIN_GLYPHS letter-sized ink blobs packed
# edge-to-edge is treated as a text line.
TEXT_MASK_CV_ENABLED = _bool("TEXT_MASK_CV_ENABLED", True)
TEXT_MASK_CV_MIN_GLYPH_H = _int("TEXT_MASK_CV_MIN_GLYPH_H", 5)
TEXT_MASK_CV_MAX_GLYPH_H = _int("TEXT_MASK_CV_MAX_GLYPH_H", 26)
TEXT_MASK_CV_MIN_GLYPHS = _int("TEXT_MASK_CV_MIN_GLYPHS", 4)
# Dense-text region mask: legend tables / title blocks / note columns stack
# many words tightly; every detection centered there is dropped (legend glyph
# pictures are real symbol shapes, so no threshold catches them).
TEXT_DENSE_MASK_ENABLED = _bool("TEXT_DENSE_MASK_ENABLED", True)
TEXT_DENSE_MIN_WORDS = _int("TEXT_DENSE_MIN_WORDS", 10)
TEXT_DENSE_RADIUS_PX = _float("TEXT_DENSE_RADIUS_PX", 140.0)
TEXT_DENSE_PAD_PX = _float("TEXT_DENSE_PAD_PX", 12.0)
# Two-tile consensus: a rescue-band hit whose location is covered by 2+ zoom
# tiles must be seen by at least 2 of them (real symbols re-detect across the
# 25% overlap; one-off smudges don't).
YOLO_RESCUE_CONSENSUS = _bool("YOLO_RESCUE_CONSENSUS", True)
# Full-wing pass for LARGE symbols: a symbol bigger than the tile overlap
# (~147px) can be cut at a boundary in every tile that sees it, so the tile
# pass never sees the whole shape. One extra pass over full_wing.jpg catches
# them; only confident boxes at/above the size floor are added.
YOLO_FULLWING_PASS = _bool("YOLO_FULLWING_PASS", True)
YOLO_FULLWING_MIN_PX = _float("YOLO_FULLWING_MIN_PX", 120.0)
# Second-stage verification classifier: crops of low-confidence detections are
# re-judged by a small true/false classifier (scripts/train_verifier.py at the
# repo root). No-op when the weights file is absent.
VERIFY_ENABLED = _bool("VERIFY_ENABLED", True)
VERIFY_MODEL_PATH = Path(
    os.getenv("VERIFY_MODEL_PATH", str(_MODEL_DIR / "verifier" / "detection_verifier.pt"))
)
# Only detections below this confidence get a second opinion; confident hits
# are trusted as-is.
VERIFY_MAX_CONF = _float("VERIFY_MAX_CONF", 0.60)
# Drop a checked detection when P(false_symbol) is at least this value.
VERIFY_DROP_PROB = _float("VERIFY_DROP_PROB", 0.80)
VERIFY_CROP_MARGIN = _float("VERIFY_CROP_MARGIN", 0.40)
VERIFY_IMGSZ = _int("VERIFY_IMGSZ", 64)

# Pricing / invoice
PRICING_DIR = ROOT / "pricing"
PRICING_CSV_PATH = Path(
    os.getenv("PRICING_CSV_PATH", str(PRICING_DIR / "device_prices.csv"))
)
INVOICE_TAX_RATE = _float("INVOICE_TAX_RATE", 0.045)
# Synthetic daily price drift is confusing in demos ("why did the estimate
# change?"), so it is OFF unless explicitly enabled. Manual edits always stick.
PRICE_DRIFT_ENABLED = _bool("PRICE_DRIFT_ENABLED", False)

# Company ("From") block printed on the estimate PDF — set real details in .env
COMPANY_NAME = (os.getenv("COMPANY_NAME") or "Aziro").strip()
COMPANY_ADDRESS = (os.getenv("COMPANY_ADDRESS") or "").strip()
COMPANY_CONTACT = (os.getenv("COMPANY_CONTACT") or "").strip()
COMPANY_TAX_ID = (os.getenv("COMPANY_TAX_ID") or "").strip()
# How long a generated cost estimate stays valid (shown on the PDF)
ESTIMATE_VALID_DAYS = _int("ESTIMATE_VALID_DAYS", 30)
SYMBOL_COUNT_ENABLED = _bool("SYMBOL_COUNT_ENABLED", True)
