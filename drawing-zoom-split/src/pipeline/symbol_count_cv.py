"""Count legend symbols per wing using the direct wing crop image (no LLM).

Floor-plan sheets (T-200–T-299) only. Each wing JPEG is counted independently:
PDF tag text mapped into wing pixel coordinates, plus OpenCV triangle
detection for ``#`` data-drop glyphs on the wing image.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import fitz

from pipeline import config
from pipeline.extraction.symbol_table_extractor import (
    SymbolTableInfo,
    load_symbol_table_json,
)
from pipeline.sheet_scan import _in_title_block
from pipeline.symbol_count_report import write_symbol_count_reports

logger = logging.getLogger(__name__)

METADATA_DIR = "05_metadata"
SHORT_TAG_MAX_SIZE = 30.0
SHORT_TAG_MIN_SIZE = 6.0
DUPLICATE_IOU = 0.75
FLOOR_PLAN_SHEET_RE = re.compile(r"^T-(\d+)", re.IGNORECASE)
TITLE_FONTS = ("romantic", "times")
WING_TITLE_BAND_PCT = 0.12
TRIANGLE_MIN_AREA = 200
TRIANGLE_MAX_AREA = 500
TRIANGLE_SKIP_RADIUS = 90.0
TRIANGLE_DEDUPE_RADIUS = 25.0


def _display_key(symbol: str | None, description: str) -> str:
    return str(symbol or description or "").strip()


def tagged_legend_keys(legend: SymbolTableInfo) -> list[tuple[str, str]]:
    """(TAG, display_key) for entries that have a countable tag."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in legend.entries:
        tag = (entry.symbol or "").strip().upper()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append((tag, _display_key(entry.symbol, entry.description)))
    out.sort(key=lambda item: (-len(item[0]), item[0]))
    return out


def match_span_to_tag(
    text: str,
    tags: list[tuple[str, str]],
    *,
    size: float | None = None,
    font: str | None = None,
) -> str | None:
    """Return display_key if this span is a legend tag instance, else None."""
    raw = (text or "").strip()
    if not raw:
        return None
    font_l = (font or "").lower()
    if any(name in font_l for name in TITLE_FONTS):
        return None
    if size is not None and (size < SHORT_TAG_MIN_SIZE or size > SHORT_TAG_MAX_SIZE):
        return None
    upper = raw.upper()

    for tag, display in tags:
        if tag == "#":
            if upper == "#" or re.fullmatch(r"#\d+", upper):
                return display
            continue
        matched = upper == tag or bool(
            re.fullmatch(re.escape(tag) + r"(?:-\d+|\d+)$", upper)
        )
        if matched:
            return display
    return None


def pdf_points_to_page_pixels(
    x: float,
    y: float,
    *,
    scale: float,
) -> tuple[float, float]:
    return x * scale, y * scale


def point_in_cv_box(
    px: float,
    py: float,
    box: dict[str, Any] | None,
) -> bool:
    if not isinstance(box, dict):
        return False
    try:
        x1, y1 = float(box["x1"]), float(box["y1"])
        x2, y2 = float(box["x2"]), float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return False
    return x1 <= px <= x2 and y1 <= py <= y2


def box_area(box: dict[str, Any]) -> float:
    return max(0.0, float(box["x2"]) - float(box["x1"])) * max(
        0.0, float(box["y2"]) - float(box["y1"])
    )


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def is_floor_plan_sheet(sheet_no: str | None) -> bool:
    """True for T-200–T-299 technology floor plans (not demo, site, or details)."""
    match = FLOOR_PLAN_SHEET_RE.match((sheet_no or "").strip())
    if not match:
        return False
    number = int(match.group(1))
    return 200 <= number <= 299


def wing_named_on_sheet(wing_name: str, title: str | None) -> bool:
    name = (wing_name or "").strip().upper()
    header = (title or "").strip().upper()
    if not name or not header:
        return False
    return re.search(rf"(?<![A-Z0-9]){re.escape(name)}(?![A-Z0-9])", header) is not None


def dedupe_overlapping_wings(
    wings: list[dict[str, Any]],
    *,
    iou_threshold: float = DUPLICATE_IOU,
) -> list[dict[str, Any]]:
    """Keep one crop per wing name on a page (the smallest box)."""
    del iou_threshold
    best: dict[str, dict[str, Any]] = {}
    for wing in sorted(wings, key=lambda item: box_area(item["cv_box_on_page"])):
        name = wing["wing_name"]
        if name not in best:
            best[name] = wing
    return list(best.values())


def prefer_labeled_wings(wings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use PDF title crops when present; vision copies often reuse the wrong box."""
    labeled = [wing for wing in wings if wing.get("source") == "pdf_text"]
    return labeled or wings


def select_wings_for_counting(wings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every wing name on the page; prefer pdf_text crop when names overlap."""
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for wing in wings:
        by_name[wing["wing_name"]].append(wing)
    selected: list[dict[str, Any]] = []
    for name in sorted(by_name):
        group = by_name[name]
        pdf_crops = [w for w in group if w.get("source") == "pdf_text"]
        pool = pdf_crops or group
        selected.append(min(pool, key=lambda item: box_area(item["cv_box_on_page"])))
    return selected


def expand_wing_bands(
    wings: list[dict[str, Any]],
    diagram: dict[str, Any] | None,
    *,
    pad_y2: float = 0.0,
) -> list[dict[str, Any]]:
    """Full-width stacked bands covering the diagram, including tags just below."""
    if not wings:
        return []
    if isinstance(diagram, dict):
        x1 = float(diagram["x1"])
        y1 = float(diagram["y1"])
        x2 = float(diagram["x2"])
        y2 = max(float(diagram["y2"]), float(pad_y2 or 0.0))
    else:
        x1 = min(float(w["cv_box_on_page"]["x1"]) for w in wings)
        y1 = min(float(w["cv_box_on_page"]["y1"]) for w in wings)
        x2 = max(float(w["cv_box_on_page"]["x2"]) for w in wings)
        y2 = max(
            max(float(w["cv_box_on_page"]["y2"]) for w in wings),
            float(pad_y2 or 0.0),
        )

    ordered = sorted(wings, key=lambda item: float(item["cv_box_on_page"]["y1"]))
    bands: list[dict[str, Any]] = []
    for index, wing in enumerate(ordered):
        box_y1 = float(wing["cv_box_on_page"]["y1"])
        box_y2 = float(wing["cv_box_on_page"]["y2"])
        if index == 0:
            box_y1 = min(box_y1, y1)
        if index == len(ordered) - 1:
            box_y2 = max(box_y2, y2)
        bands.append(
            {
                **wing,
                "cv_box_on_page": {
                    "x1": x1,
                    "y1": box_y1,
                    "x2": x2,
                    "y2": box_y2,
                },
            }
        )
    for index in range(len(bands) - 1):
        split = (
            float(bands[index]["cv_box_on_page"]["y2"])
            + float(bands[index + 1]["cv_box_on_page"]["y1"])
        ) / 2.0
        bands[index]["cv_box_on_page"]["y2"] = split
        bands[index + 1]["cv_box_on_page"]["y1"] = split
    return bands


def assign_hits_to_wings(
    spans: list[dict[str, Any]],
    wings: list[dict[str, Any]],
    tags: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    """Give each tag hit to the smallest containing wing box (no double counting)."""
    counts: dict[str, dict[str, int]] = {wing["column_key"]: {} for wing in wings}
    for span in spans:
        matched = match_span_to_tag(
            str(span["text"]),
            tags,
            size=span.get("size"),
            font=span.get("font"),
        )
        if not matched:
            continue
        containing = [
            wing
            for wing in wings
            if point_in_cv_box(span["px"], span["py"], wing["cv_box_on_page"])
        ]
        if not containing:
            continue
        winner = min(containing, key=lambda item: box_area(item["cv_box_on_page"]))
        bucket = counts[winner["column_key"]]
        bucket[matched] = bucket.get(matched, 0) + 1
    return counts


def count_spans_for_wing(
    spans: list[dict[str, Any]],
    wing_box: dict[str, Any],
    tags: list[tuple[str, str]],
    *,
    exclude_top_pct: float = 0.0,
) -> dict[str, int]:
    """Count legend tag spans whose page-pixel center lies inside wing_box."""
    counts: dict[str, int] = {}
    try:
        x1 = float(wing_box["x1"])
        y1 = float(wing_box["y1"])
        x2 = float(wing_box["x2"])
        y2 = float(wing_box["y2"])
    except (KeyError, TypeError, ValueError):
        return counts
    wing_h = max(1.0, y2 - y1)
    title_cutoff = y1 + wing_h * max(0.0, exclude_top_pct)

    for span in spans:
        px = float(span["px"])
        py = float(span["py"])
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            continue
        if py < title_cutoff:
            continue
        matched = match_span_to_tag(
            str(span["text"]),
            tags,
            size=span.get("size"),
            font=span.get("font"),
        )
        if not matched or matched == "#":
            continue
        counts[matched] = counts.get(matched, 0) + 1
    return counts


def _ap_skip_centers_on_wing(
    spans: list[dict[str, Any]],
    wing_box: dict[str, Any],
    tags: list[tuple[str, str]],
) -> list[tuple[float, float]]:
    """Wing-local pixel centers of AP/TAP labels (triangles beside these are not drops)."""
    try:
        x1 = float(wing_box["x1"])
        y1 = float(wing_box["y1"])
    except (KeyError, TypeError, ValueError):
        return []
    centers: list[tuple[float, float]] = []
    for span in spans:
        if not point_in_cv_box(span["px"], span["py"], wing_box):
            continue
        text = str(span["text"]).strip().upper()
        if text == "AP" or text.startswith("TAP"):
            matched = match_span_to_tag(
                text,
                tags,
                size=span.get("size"),
                font=span.get("font"),
            )
            if matched == "AP" or text.startswith("TAP"):
                centers.append((float(span["px"]) - x1, float(span["py"]) - y1))
    return centers


def count_drop_triangles(
    image_path: Path,
    *,
    skip_centers: list[tuple[float, float]] | None = None,
    exclude_top_pct: float = WING_TITLE_BAND_PCT,
) -> int:
    """Count solid upward triangles (# data drops) on a wing crop JPEG."""
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    h, _w = image.shape[:2]
    y_min = int(h * max(0.0, exclude_top_pct))
    _, binary = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)
    if y_min > 0:
        binary[:y_min, :] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skip = skip_centers or []
    kept: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < TRIANGLE_MIN_AREA or area > TRIANGLE_MAX_AREA:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.08 * peri, True)
        if len(approx) != 3:
            continue
        x, y, width, height = cv2.boundingRect(approx)
        ratio = width / float(max(height, 1))
        if ratio < 0.6 or ratio > 1.6:
            continue
        cx, cy = x + width / 2.0, y + height / 2.0
        if any(
            (cx - sx) ** 2 + (cy - sy) ** 2 <= TRIANGLE_SKIP_RADIUS**2
            for sx, sy in skip
        ):
            continue
        if any(
            (cx - kx) ** 2 + (cy - ky) ** 2 <= TRIANGLE_DEDUPE_RADIUS**2
            for kx, ky in kept
        ):
            continue
        kept.append((cx, cy))
    return len(kept)


def count_symbols_on_wing_image(
    job_root: Path,
    wing: dict[str, Any],
    page_spans: list[dict[str, Any]],
    tags: list[tuple[str, str]],
) -> dict[str, int]:
    """Count symbols for one wing using its direct crop image + scoped PDF text."""
    image_path = job_root / str(wing["image"])
    if not image_path.is_file():
        return {}

    wing_box = wing["cv_box_on_page"]
    counts = count_spans_for_wing(
        page_spans,
        wing_box,
        tags,
        exclude_top_pct=WING_TITLE_BAND_PCT,
    )

    skip = _ap_skip_centers_on_wing(page_spans, wing_box, tags)
    drops = count_drop_triangles(
        image_path,
        skip_centers=skip,
        exclude_top_pct=WING_TITLE_BAND_PCT,
    )
    if drops:
        hash_key = next((display for tag, display in tags if tag == "#"), "#")
        counts[hash_key] = drops
    return counts


def _extract_text_spans(pdf_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Page-index → spans with pixel centers on the rendered sheet."""
    scale = max(72, int(config.VISION_PDF_DPI)) / 72.0
    pages: dict[int, list[dict[str, Any]]] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pw = max(1.0, float(page.rect.width))
            ph = max(1.0, float(page.rect.height))
            spans: list[dict[str, Any]] = []
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text") or "").strip()
                        if not text:
                            continue
                        bbox = span.get("bbox") or [0, 0, 0, 0]
                        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
                        nx0, ny0, nx1, ny1 = x0 / pw, y0 / ph, x1 / pw, y1 / ph
                        if _in_title_block([nx0, ny0, nx1, ny1]) or ny0 > 0.86:
                            continue
                        cx, cy = pdf_points_to_page_pixels(
                            (x0 + x1) / 2.0,
                            (y0 + y1) / 2.0,
                            scale=scale,
                        )
                        try:
                            size = float(span.get("size") or 0.0)
                        except (TypeError, ValueError):
                            size = 0.0
                        spans.append(
                            {
                                "text": text,
                                "px": cx,
                                "py": cy,
                                "size": size,
                                "font": str(span.get("font") or ""),
                            }
                        )
            pages[page_idx] = spans
    finally:
        doc.close()
    return pages


def _load_wing_regions(job_root: Path) -> list[dict[str, Any]]:
    meta_dir = job_root / METADATA_DIR
    if not meta_dir.is_dir():
        return []

    regions: list[dict[str, Any]] = []
    for meta_path in sorted(meta_dir.glob("page_*.json")):
        if meta_path.name.endswith("_notes.json"):
            continue
        try:
            page_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        sheet_no = str(page_meta.get("sheet_no") or "")
        title = str(page_meta.get("title") or "")
        if not is_floor_plan_sheet(sheet_no):
            continue

        page_idx = int(page_meta.get("page", 0)) - 1
        page_key = str(page_meta.get("page_key") or meta_path.stem)
        diagram = page_meta.get("cv_box_on_page")
        for wing in page_meta.get("wings") or []:
            if not isinstance(wing, dict):
                continue
            wing_name = str(wing.get("name") or "")
            image_rel = str(wing.get("image") or "")
            box = wing.get("cv_box_on_page")
            if not wing_name or not image_rel or not isinstance(box, dict):
                continue
            if not wing_named_on_sheet(wing_name, title):
                continue
            instance_dir = Path(image_rel).stem
            regions.append(
                {
                    "page_idx": page_idx,
                    "page_key": page_key,
                    "sheet_no": sheet_no,
                    "wing_name": wing_name,
                    "instance_dir": instance_dir,
                    "source": str(wing.get("source") or ""),
                    "column_key": wing_name,
                    "cv_box_on_page": box,
                    "diagram_box": diagram if isinstance(diagram, dict) else None,
                    "image": image_rel,
                    "zooms_manifest": str(wing.get("zooms_manifest") or ""),
                }
            )
    return regions


def _merge_counts(parts: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for part in parts:
        for key, value in part.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def aggregate_results_by_wing_name(
    per_sheet: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One pivot column per wing name (sum counts across all T-200 sheets)."""
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sheet:
        by_name[str(row["wing_name"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for wing_name in sorted(by_name):
        rows = by_name[wing_name]
        merged = _merge_counts([r.get("counts") or {} for r in rows])
        total = sum(merged.values())
        aggregated.append(
            {
                "column_key": wing_name,
                "wing_name": wing_name,
                "page_key": "floor-plans",
                "instance_dir_name": wing_name,
                "sheets_processed": len(rows),
                "wing_images": [r.get("wing_image") for r in rows if r.get("wing_image")],
                "tiles_processed": len(rows),
                "raw_detection_count": total,
                "post_nms_count": total,
                "nms_removed": 0,
                "counts": merged,
                "detections": [],
            }
        )
    return aggregated


def run_symbol_count_cv(
    job_root: Path,
    *,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Count tagged symbols per named wing on T-200 floor plans."""
    job_root = Path(job_root)

    symbol_path = job_root / "symbol_table.json"
    legend = load_symbol_table_json(symbol_path)
    if legend is None or not legend.entries:
        return {
            "status": "skipped",
            "reason": "symbol_table.json missing or empty",
            "instances": 0,
        }

    source_dir = job_root / "01_source"
    pdfs = list(source_dir.glob("*.pdf"))
    if not pdfs:
        return {
            "status": "skipped",
            "reason": "no source PDF found",
            "instances": 0,
        }

    wings = _load_wing_regions(job_root)
    if not wings:
        return {
            "status": "skipped",
            "reason": "no T-200 floor-plan wings with cv_box_on_page",
            "instances": 0,
        }

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress("counting_symbols", msg)

    _progress("Extracting PDF text into page-pixel coordinates")
    text_pages = _extract_text_spans(pdfs[0])
    tags = tagged_legend_keys(legend)

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for wing in wings:
        by_page[wing["page_idx"]].append(wing)

    per_sheet: list[dict[str, Any]] = []
    for page_idx, page_wings in sorted(by_page.items()):
        unique = select_wings_for_counting(page_wings)
        page_spans = text_pages.get(page_idx, [])
        _progress(
            f"Counting symbols on page {page_idx + 1:03d} "
            f"({len(unique)} wing image(s))"
        )
        for wing in unique:
            image_rel = str(wing.get("image") or "")
            _progress(f"  {wing['wing_name']} ({Path(image_rel).name})")
            counts = count_symbols_on_wing_image(
                job_root,
                wing,
                page_spans,
                tags,
            )
            total = sum(counts.values())
            per_sheet.append(
                {
                    "column_key": wing["wing_name"],
                    "wing_name": wing["wing_name"],
                    "sheet_no": wing["sheet_no"],
                    "page_key": wing["page_key"],
                    "instance_dir_name": wing["instance_dir"],
                    "wing_image": image_rel,
                    "tiles_processed": 1,
                    "raw_detection_count": total,
                    "post_nms_count": total,
                    "nms_removed": 0,
                    "counts": counts,
                    "detections": [],
                }
            )

    instance_results = aggregate_results_by_wing_name(per_sheet)

    out_dir = job_root / config.SYMBOL_COUNT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = {
        "status": "done",
        "method": "wing_image_by_wing_name",
        "instances": len(instance_results),
        "sheets_scanned": len(per_sheet),
        "legend_entry_count": legend.entry_count,
        "tags_used": [tag for tag, _ in tags],
        "sheets": "T-200–T-299",
        "instance_results": instance_results,
        "per_sheet_results": per_sheet,
    }
    (out_dir / "symbol_count_detail.json").write_text(
        json.dumps(detail, indent=2),
        encoding="utf-8",
    )

    report_paths = write_symbol_count_reports(
        legend=legend,
        instance_results=instance_results,
        out_dir=out_dir,
        source_filename=job_root.name,
        project_title=None,
    )

    return {
        "status": "done",
        "instances": len(instance_results),
        "symbol_count_detail_path": f"{config.SYMBOL_COUNT_DIR}/symbol_count_detail.json",
        "symbol_count_report_csv": report_paths["csv"],
        "symbol_count_report_pdf": report_paths.get("pdf"),
        "symbol_count_report_json": report_paths.get("json"),
    }
