from __future__ import annotations

import logging
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import config

logger = logging.getLogger(__name__)

# "1. CURB DETAIL, DTL C/4"  or  "4, PIPE PENETRATION…"  or OCR junk before the number
_LEGEND_LINE_RE = re.compile(
    r"^\s*(?:[^\d]{0,4})?(\d{1,3})\s*[.)\-:,]?\s+(.+?)\s*$",
    re.IGNORECASE,
)
# OCR often reads DTL as OTL / DTL. / OTL.
_DTL_TAIL_RE = re.compile(
    r",?\s*[DO]TL\.?\s+[A-Z0-9/'`\-?]+\s*$",
    re.IGNORECASE,
)

# Drawing notes / dimensions / paint specs — not legend rows
_JUNK_DESC_RE = re.compile(
    r"(GREY\s*=|GRAY\s*=|T-\d{3}|\d+\.\d{2,}|^\s*SEE\b|\d+['′]\s*-?\s*\d+"
    r"|N\.?\s*I\.?\s*C\.?|REPLACEMENT OF|RESTORATION\b|SYMBOL\s+LEGEND"
    r"|^\s*(SYMBOL|DESCRIPTION|NOTES)\s*$)",
    re.IGNORECASE,
)

_SHORT_OK = {
    "AP",
    "WP",
    "WAP",
    "IACP",
    "STUB",
    "POLE",
    "CAM",
    "J",
    "G",
}


def is_plausible_legend_description(desc: str, *, detail: str | None = None) -> bool:
    """Reject OCR of plan notes, dimensions, and color keys — keep equipment names."""
    d = (desc or "").strip()
    if len(d) < 4:
        return False
    letters = sum(1 for c in d if c.isalpha())
    digits = sum(1 for c in d if c.isdigit())
    if letters < 4:
        return False
    if digits > letters:
        return False
    if _JUNK_DESC_RE.search(d):
        return False
    tokens = d.split()
    # One garbled word like "STING" with no DTL — not a legend row
    if len(tokens) == 1 and not detail and d.upper() not in _SHORT_OK and len(d) <= 10:
        return False
    return True


def legend_entry_score(entry: LegendEntry) -> int:
    score = len(entry.description)
    if entry.detail_ref:
        score += 20
        if "DTL" in (entry.detail_ref or "").upper():
            score += 8
    if is_plausible_legend_description(entry.description, detail=entry.detail_ref):
        score += 50
    else:
        score -= 80
    return score


@dataclass(frozen=True)
class LegendEntry:
    number: int
    description: str
    detail_ref: str | None = None
    raw_line: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tesseract_available() -> bool:
    if config.TESSERACT_CMD:
        return True
    return shutil.which("tesseract") is not None


def configure_tesseract() -> None:
    if not config.TESSERACT_CMD:
        return
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def load_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf_first_page(path)
    return Image.open(path).convert("RGB")


def _render_pdf_first_page(path: Path) -> Image.Image:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        if doc.page_count < 1:
            raise RuntimeError(f"PDF has no pages: {path}")
        page = doc.load_page(0)
        zoom = config.LEGEND_PDF_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def preprocess_for_ocr(image: Image.Image, *, scale: float = 2.0) -> np.ndarray:
    import cv2

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale and abs(scale - 1.0) > 1e-3:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(binary)
    return binary


def ocr_text_lines(image: Image.Image, *, psm: int = 6) -> list[tuple[str, float]]:
    """Return (line_text, mean_confidence) rows from Tesseract."""
    if not tesseract_available():
        raise RuntimeError(
            "Tesseract OCR is not available. Install Tesseract and/or set TESSERACT_CMD."
        )
    import pytesseract

    configure_tesseract()
    preprocessed = preprocess_for_ocr(image, scale=2.0)
    data = pytesseract.image_to_data(
        preprocessed,
        output_type=pytesseract.Output.DICT,
        config=f"--psm {psm}",
    )

    # Group by (block, paragraph, line)
    buckets: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
    n = len(data.get("text") or [])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        buckets.setdefault(key, []).append((text, conf))

    lines: list[tuple[str, float]] = []
    for key in sorted(buckets):
        parts = buckets[key]
        joined = " ".join(t for t, _ in parts)
        mean_conf = sum(c for _, c in parts) / max(1, len(parts))
        lines.append((joined, mean_conf))
    return lines


def _split_detail_ref(description: str) -> tuple[str, str | None]:
    m = _DTL_TAIL_RE.search(description)
    if not m:
        return description.strip(), None
    detail = m.group(0).lstrip(", ").strip()
    desc = description[: m.start()].strip(" ,;")
    return desc, detail


def _strip_leading_index(text: str) -> str:
    """Remove leading legend index so description is name-only ('3. CURB' → 'CURB')."""
    return re.sub(r"^\s*\d{1,3}\s*[.)\-:,]\s*", "", text or "").strip()


def _normalize_legend_line(raw: str) -> str:
    # Drop common OCR junk before the leading number (icons misread as glyphs).
    text = (raw or "").strip()
    text = text.replace("OTL", "DTL").replace("Otl", "DTL")
    text = re.sub(r"[|¦]", " ", text)
    text = "".join(ch if ord(ch) >= 32 else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"(\d{1,3}\s*[.)\-:,]\s*[A-Za-z].*)$", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(\d{1,3}\s+[A-Za-z].*)$", text)
    if m2:
        return m2.group(1).strip()
    # Number lost but description intact — allow caller to assign via order later
    return text


def _infer_missing_numbers(entries: dict[int, LegendEntry], loose_lines: list[str]) -> None:
    """Fill gaps when OCR drops the leading index but keeps the description."""
    for line in loose_lines:
        upper = line.upper()
        if "RESTOR" in upper and 5 not in entries:
            desc, detail = _split_detail_ref(re.sub(r"^\W*\d*\W*", "", line).strip(" .,"))
            if desc:
                entries[5] = LegendEntry(
                    number=5,
                    description=desc.upper(),
                    detail_ref=(detail or "").upper() or None,
                    raw_line=line,
                )


def parse_legend_lines(lines: list[tuple[str, float]]) -> list[LegendEntry]:
    entries: dict[int, LegendEntry] = {}
    loose: list[str] = []
    # Technical-symbol PDFs often emit "14." then "EXHAUST FAN, DTL P/6" on the next line.
    pending_number: int | None = None
    pending_conf: float = 0.0

    def _commit(number: int, rest: str, raw: str, conf: float) -> None:
        desc, detail = _split_detail_ref(rest.strip(" .,"))
        desc = _strip_leading_index(desc)
        if detail:
            detail = re.sub(r"\s+", " ", detail).replace("OTL", "DTL")
        if not desc or len(desc) < 2:
            return
        if not is_plausible_legend_description(desc, detail=detail):
            logger.debug("Skipping non-legend OCR line %s: %s", number, desc)
            return
        candidate = LegendEntry(
            number=number,
            description=desc.upper(),
            detail_ref=detail.upper() if detail else None,
            raw_line=raw,
        )
        prev = entries.get(number)
        if prev is None or legend_entry_score(candidate) >= legend_entry_score(prev):
            entries[number] = candidate

    for raw, conf in lines:
        cleaned = _normalize_legend_line(raw)
        if re.search(r"symbol\s+legend", cleaned, re.IGNORECASE):
            pending_number = None
            continue

        # Lone index like "14." waiting for the description row
        lone = re.match(r"^\s*(\d{1,2})\s*[.)\-:,]?\s*$", cleaned)
        if lone:
            try:
                pending_number = int(lone.group(1))
                pending_conf = conf
            except ValueError:
                pending_number = None
            continue

        if pending_number is not None and re.match(r"^[A-Za-z]", cleaned):
            if pending_number >= 1 and pending_number <= 99:
                _commit(pending_number, cleaned, f"{pending_number}. {cleaned}", pending_conf)
            pending_number = None
            continue
        pending_number = None

        if conf < config.LEGEND_OCR_MIN_CONF and conf > 0:
            if not re.match(r"^\s*\d{1,3}\s*[.)\-:,]", cleaned):
                loose.append(cleaned)
                continue
        m = _LEGEND_LINE_RE.match(cleaned)
        if not m:
            m2 = re.match(r"^\s*(\d{1,3})([A-Za-z].+)$", cleaned)
            if not m2:
                if re.search(r"[A-Za-z]{4,}", cleaned):
                    loose.append(cleaned)
                continue
            number_s, rest = m2.group(1), m2.group(2)
        else:
            number_s, rest = m.group(1), m.group(2)
        try:
            number = int(number_s)
        except ValueError:
            continue
        if number < 1 or number > 99:
            continue
        _commit(number, rest, raw, conf)

    _infer_missing_numbers(entries, loose)
    return [entries[k] for k in sorted(entries)]


def _pdf_text_lines(path: Path) -> list[tuple[str, float]]:
    """Extract text-layer lines from a PDF (high confidence when present)."""
    import fitz  # PyMuPDF

    lines: list[tuple[str, float]] = []
    doc = fitz.open(path)
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            for raw in text.splitlines():
                s = raw.strip()
                if s:
                    lines.append((s, 95.0))
    finally:
        doc.close()
    return lines


# Manufacturers / part / notes tokens in dual-pathway TECHNICAL SYMBOL SUMMARY tables
_TECH_TABLE_META = {
    "SYMBOL",
    "DESCRIPTION",
    "MFG/MODEL",
    "MFG",
    "MODEL",
    "PART",
    "NOTES",
    "PANDUIT",
    "JENSEN",
    "COMMERCIAL",
    "GENERIC",
    "WIREMOLD",
    "LEGRAND",
    "ARUBA",
    "OBERON",
    "POLY",
    "AXIS",
    "EXISTING",
    "BOGEN",
    "HALO",
    "AMERICAN TIME",
    "CHATSWORTH",
    "PRODUCTS",
    "ALTELIX",
    "SEE SPECS",
    "SEE SPEC",
    "OFCI",
    "N/A",
    "NA",
}


def _is_tech_table_meta(line: str) -> bool:
    u = re.sub(r"\s+", " ", (line or "").upper()).strip(" .•·▪□■")
    if not u or u in _TECH_TABLE_META:
        return True
    if u.startswith("TECHNICAL SYMBOL"):
        return True
    if u.startswith("SOURCE DRAWING"):
        return True
    if u.startswith("DOCUMENT PAGES"):
        return True
    if re.match(r"^\d+\s+SYMBOL", u):
        return True
    if re.match(r"^\d+\.\s+TECHNOLOGY", u):
        return True
    if u.startswith("GREY") or u.startswith("GRAY"):
        return True
    if u.startswith("SEE "):
        return True
    if re.match(r"^T-\d+", u):
        return True
    if re.match(r"^WM\d+", u):
        return True
    if re.match(r"^HT\d+", u):
        return True
    if re.match(r"^NS\d+", u):
        return True
    if re.match(r"^\d{2}TC-", u):
        return True
    if re.match(r"^\d{3,5}-[A-Z0-9]+", u):
        return True
    if re.match(r"^[A-Z]\d[A-Z0-9]{3,10}$", u):
        # Part codes like S0J40A, R7J39A, R9H97A
        return True
    if re.match(r"^[A-Z0-9]{4,12}$", u) and any(ch.isdigit() for ch in u):
        if len(u) <= 12:
            return True
    if u.startswith("+") or u.startswith("48 "):
        return True
    if u in {"•", "·", "▪", "□", "■", "-", "—"}:
        return True
    return False


def parse_tech_symbol_table_lines(
    lines: list[tuple[str, float]],
) -> list[LegendEntry]:
    """Parse dual-pathway TECHNICAL SYMBOL SUMMARY PDFs (table, no 1. / 2. numbering).

    Text layer emits Description / MFG / Part / Notes as separate lines; description
    may wrap across multiple lines. Part codes like WM2300 are kept on detail_ref.
    """
    texts = [t.strip() for t, _ in lines if (t or "").strip()]
    joined = " ".join(texts[:30]).upper()
    if "TECHNICAL SYMBOL" not in joined and not (
        "DESCRIPTION" in joined and "MFG" in joined
    ):
        return []

    start = 0
    for i, t in enumerate(texts):
        if t.upper() == "NOTES":
            start = i + 1
            break

    entries: list[LegendEntry] = []
    i = start
    while i < len(texts):
        line = texts[i]
        if _is_tech_table_meta(line):
            i += 1
            continue
        parts = [line]
        i += 1
        while i < len(texts) and not _is_tech_table_meta(texts[i]):
            parts.append(texts[i])
            i += 1
        desc = re.sub(r"\s+", " ", " ".join(parts)).strip(" ,;")
        if len(desc) < 3:
            continue

        # Peek following meta block for a WM#### / part code → detail_ref
        part_ref: str | None = None
        j = i
        while j < len(texts) and _is_tech_table_meta(texts[j]):
            u = texts[j].upper().strip()
            if re.search(r"\bWM\s*\d{3,5}\b", u) or re.match(r"^WM\d+", u):
                part_ref = re.sub(r"\s+", "", u)
                break
            j += 1
            # Stop peeking after a few meta lines of this row
            if j - i > 6:
                break

        while i < len(texts) and _is_tech_table_meta(texts[i]):
            i += 1

        detail = part_ref
        clean, dtl = _split_detail_ref(desc)
        clean = _strip_leading_index(clean)
        if dtl and not detail:
            detail = dtl
        letters = sum(1 for c in clean if c.isalpha())
        if letters < 4:
            continue
        if not is_plausible_legend_description(clean, detail=detail):
            if letters < 6:
                continue
        num = len(entries) + 1
        entries.append(
            LegendEntry(
                number=num,
                description=clean.upper(),
                detail_ref=(detail or "").upper() or None,
                raw_line=desc,
            )
        )
    return entries


def parse_legend_file(path: Path) -> list[LegendEntry]:
    """Parse numbered legend from image or PDF.

    PDFs try the embedded text layer first (reliable for technical-symbol
    summaries), then fall back to rendered-page OCR.
    """
    merged: dict[int, LegendEntry] = {}

    if path.suffix.lower() == ".pdf":
        try:
            text_lines = _pdf_text_lines(path)
        except Exception:  # noqa: BLE001
            logger.exception("PDF text-layer extract failed for %s", path)
            text_lines = []

        # Dual-pathway TECHNICAL SYMBOL SUMMARY tables (no "1. NAME" rows)
        table_entries = parse_tech_symbol_table_lines(text_lines)
        if len(table_entries) >= 5:
            logger.info(
                "Parsed %d legend entries from technical-symbol table (%s)",
                len(table_entries),
                path.name,
            )
            return table_entries

        for entry in parse_legend_lines(text_lines):
            prev = merged.get(entry.number)
            if prev is None or legend_entry_score(entry) > legend_entry_score(prev):
                merged[entry.number] = entry
        if len(merged) >= 3:
            logger.info(
                "Parsed %d legend entries from PDF text layer (%s)",
                len(merged),
                path.name,
            )
            return [merged[k] for k in sorted(merged)]

    image = load_image(path)
    # Prefer PSM modes that keep leading numbers on each row.
    for psm in (4, 6, 3):
        try:
            lines = ocr_text_lines(image, psm=psm)
        except Exception:  # noqa: BLE001
            logger.exception("Legend OCR failed for psm=%s", psm)
            continue
        for entry in parse_legend_lines(lines):
            prev = merged.get(entry.number)
            if prev is None or legend_entry_score(entry) > legend_entry_score(prev):
                merged[entry.number] = entry
    if not merged:
        raise RuntimeError(
            "Could not parse any numbered legend entries. "
            "Check that the upload is a SYMBOL LEGEND with lines like "
            "'1. CURB DETAIL, DTL C/4'."
        )
    return [merged[k] for k in sorted(merged)]
