from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(ROOT / "storage" / "jobs")))
TESSERACT_CMD = (os.getenv("TESSERACT_CMD") or "").strip() or None

# Legend OCR
LEGEND_OCR_MIN_CONF = float(os.getenv("LEGEND_OCR_MIN_CONF", "40"))
LEGEND_PDF_DPI = int(os.getenv("LEGEND_PDF_DPI", "200"))

# Circled callout detection (numbers INSIDE hollow circles only)
# Set CALLOUT_OCR_ENABLED=true to turn callout OCR back on.
CALLOUT_OCR_ENABLED = os.getenv("CALLOUT_OCR_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CALLOUT_OCR_MIN_CONF = float(os.getenv("CALLOUT_OCR_MIN_CONF", "30"))
CALLOUT_MIN_RADIUS_PX = int(os.getenv("CALLOUT_MIN_RADIUS_PX", "5"))
CALLOUT_MAX_RADIUS_PX = int(os.getenv("CALLOUT_MAX_RADIUS_PX", "28"))
CALLOUT_NMS_IOU = float(os.getenv("CALLOUT_NMS_IOU", "0.30"))
# Dedup nearby identical circled numbers
CALLOUT_DEDUP_DIST_PX = float(os.getenv("CALLOUT_DEDUP_DIST_PX", "22"))

# Sync jobs for tests / debugging
SYNC_JOBS = os.getenv("LEGEND_OCR_SYNC_JOBS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

ALLOWED_LEGEND_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# YOLO symbol detector
YOLO_MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", str(ROOT / "model" / "symbol_detector_best.pt"))
)
YOLO_ENABLED = os.getenv("YOLO_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_KEEP_OBJECT_CLASS = os.getenv("YOLO_KEEP_OBJECT_CLASS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
YOLO_MATCH_MIN_SCORE = float(os.getenv("YOLO_MATCH_MIN_SCORE", "0.55"))
# Prefer YOLO instance counts when a legend row matches a class; else OCR callouts
YOLO_PRIMARY_COUNTS = os.getenv("YOLO_PRIMARY_COUNTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Second "rescue" pass: re-run the same image at lower conf (+ optional TTA)
# to recover symbols the 0.35 cutoff silently drops. Rescued boxes overlapping
# a primary detection are discarded by NMS, so nothing counts twice.
YOLO_RESCUE_ENABLED = os.getenv("YOLO_RESCUE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
YOLO_RESCUE_CONF = float(os.getenv("YOLO_RESCUE_CONF", "0.15"))
YOLO_RESCUE_TTA = os.getenv("YOLO_RESCUE_TTA", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Post-YOLO overlap de-dup (same tile + overlapping zoom tiles)
YOLO_NMS_IOU = float(os.getenv("YOLO_NMS_IOU", "0.35"))
YOLO_CROSS_NMS_IOU = float(os.getenv("YOLO_CROSS_NMS_IOU", "0.50"))
# Keep below the smallest symbol width: touching cluster members have centers
# one symbol-width apart, and a larger floor merges them into one count.
YOLO_NMS_CENTER_PX = float(os.getenv("YOLO_NMS_CENTER_PX", "8"))
TILE_OVERLAP_PCT = float(os.getenv("TILE_OVERLAP_PCT", "0.10"))
