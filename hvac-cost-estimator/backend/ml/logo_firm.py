"""Fill provider_company when the firm appears only as a logo/stamp graphic.

Two strategies (company filled only when still empty):

1. Local stamp directory keyed by office phone / address tokens.
2. OCR of a stamp/logo ROI on the cover page (real PaddleOCR only; mock skips).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from ml.requirement_extractor import RequirementInfo

logger = logging.getLogger(__name__)

# Known AOR stamps in the Real data Architect CD sets (logo-only on many sheets).
_STAMP_FIRMS: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset({"7154470", "hedding", "95126"}),
        "196 Architects",
    ),
)

_FIRM_OCR_RE = re.compile(
    r"("
    r"\d{2,4}\s*ARCHITECTS?"
    r"|[A-Z][A-Za-z0-9.&'\-]+\s+(?:Design|Architecture|Architects)"
    r"(?:\s+Inc\.?)?"
    r"|[A-Z][A-Za-z0-9.&'\-]+\s+(?:Inc\.?|LLC|LLP|Studio|Group)"
    r")",
    re.IGNORECASE,
)

_DIGITS_RE = re.compile(r"\D+")


def normalize_phone_digits(phone: str | None) -> str | None:
    if not phone:
        return None
    digits = _DIGITS_RE.sub("", phone)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) >= 7 else None


def lookup_firm_from_stamp(
    *,
    phone: str | None = None,
    address: str | None = None,
    fax: str | None = None,
) -> str | None:
    """Return a known firm name when phone/address match a stamp entry."""
    tokens: set[str] = set()
    for phone_like in (phone, fax):
        digits = normalize_phone_digits(phone_like)
        if digits:
            tokens.add(digits[-7:])  # local 7 or full 10; map uses 7154470
            if len(digits) >= 10:
                tokens.add(digits[-10:])
    if address:
        upper = address.upper()
        for token in ("HEDDING", "95126", "SAN JOSE"):
            if token in upper:
                tokens.add(token.lower().replace(" ", ""))
        # street token without spaces
        if "HEDDING" in upper:
            tokens.add("hedding")
        if "95126" in upper:
            tokens.add("95126")

    if not tokens:
        return None

    for required, firm in _STAMP_FIRMS:
        # Match when at least one required key hits (phone OR address token).
        # Prefer stronger match: phone local digits or Hedding.
        if required & tokens:
            # Require phone local OR hedding to avoid weak zip-only hits.
            if {"7154470", "hedding"} & tokens:
                return firm
    return None


def parse_firm_from_ocr_lines(lines: list[str]) -> str | None:
    """Pick a short firm-like string from OCR lines on a logo/stamp crop."""
    candidates: list[tuple[int, str]] = []
    for raw in lines:
        text = re.sub(r"\s+", " ", (raw or "").strip())
        if not text or len(text) > 60:
            continue
        upper = text.upper()
        if any(
            bad in upper
            for bad in (
                "SCHOOL DISTRICT",
                "PROJECT OWNER",
                "CONSTRUCTION DOCUMENT",
                "DOES NOT REPRESENT",
                "DISCLAIMS",
                "SUITE",
                "TEL ",
                "PHONE",
                "FAX",
            )
        ):
            continue

        match = _FIRM_OCR_RE.search(text)
        if match:
            firm = re.sub(r"\s+", " ", match.group(1)).strip()
            score = 5
            if re.search(r"\bARCHITECTS?\b", firm, re.I):
                score += 4
            if re.match(r"^\d{2,4}\s*ARCHITECTS?$", firm, re.I):
                score += 6
            if re.search(r"\bDESIGN\b", firm, re.I):
                score += 3
            candidates.append((score, firm))
            continue

        # Bare "196 ARCHITECTS" sometimes OCR'd as two lines.
        if re.fullmatch(r"\d{2,4}", text):
            candidates.append((2, text))
        elif upper in {"ARCHITECTS", "ARCHITECT"} and candidates:
            # Merge trailing ARCHITECTS onto a prior bare number candidate.
            prev_score, prev = candidates[-1]
            if re.fullmatch(r"\d{2,4}", prev):
                candidates[-1] = (prev_score + 8, f"{prev} Architects")

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    firm = candidates[0][1]
    # Title-case digit+Architects logos for readable output.
    if re.match(r"^\d{2,4}\s+ARCHITECTS?$", firm, re.I):
        parts = firm.split()
        return f"{parts[0]} Architects"
    return firm


def _render_stamp_crops(pdf_path: Path) -> list[Image.Image]:
    """Render cover-page regions likely to contain the AOR logo."""
    import fitz

    images: list[Image.Image] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("logo_firm: cannot open %s: %s", pdf_path, exc)
        return images

    try:
        page = doc[0]
        mat = fitz.Matrix(2.5, 2.5)
        clips: list[fitz.Rect] = []

        for needle in ("HEDDING", "715-4470", "TEL"):
            for rect in page.search_for(needle)[:2]:
                # Expand around stamp text; logo usually adjacent on Architect CDs.
                pad = fitz.Rect(
                    rect.x0 - 120,
                    rect.y0 - 220,
                    rect.x1 + 280,
                    rect.y1 + 80,
                )
                clips.append(pad & page.rect)

        # Fallback: lower-left of display page (rotated sheets put "196" logo there).
        r = page.rect
        clips.append(fitz.Rect(r.x0, r.y1 - r.height * 0.35, r.x0 + r.width * 0.28, r.y1))
        # Fallback: lower-right title-block strip.
        clips.append(fitz.Rect(r.x1 - r.width * 0.22, r.y1 - r.height * 0.45, r.x1, r.y1))

        seen: set[tuple[int, int, int, int]] = set()
        for clip in clips:
            if clip.is_empty or clip.width < 20 or clip.height < 20:
                continue
            key = (int(clip.x0), int(clip.y0), int(clip.x1), int(clip.y1))
            if key in seen:
                continue
            seen.add(key)
            try:
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                if pix.width < 40 or pix.height < 40:
                    continue
                images.append(
                    Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("logo_firm: crop failed %s: %s", key, exc)

        # Also try wide/short embedded images (logo banners).
        for img in page.get_images(full=True)[:6]:
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 80 or pix.height < 40:
                    continue
                # Prefer landscape logo-like aspect ratios.
                ratio = pix.width / max(pix.height, 1)
                if ratio < 1.2 and pix.width * pix.height > 400_000:
                    continue
                images.append(
                    Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("logo_firm: image xref %s failed: %s", xref, exc)
    finally:
        doc.close()

    return images


def _ocr_logo_images(images: list[Image.Image]) -> list[str]:
    """OCR logo crops. Returns [] in mock mode (never invents mock title-block firms)."""
    from config import get_settings
    from ml.ocr import ocr_logo_image

    settings = get_settings()
    lines: list[str] = []
    for image in images:
        for ocr_line in ocr_logo_image(settings, image):
            if ocr_line.text:
                lines.append(ocr_line.text)
    return lines


def fill_provider_company_from_logo(pdf_path: Path, info: RequirementInfo) -> None:
    """Set info.provider_company from stamp map and/or logo OCR when empty."""
    if info.provider_company:
        return

    firm = lookup_firm_from_stamp(
        phone=info.provider_phone,
        address=info.provider_address,
        fax=info.provider_fax,
    )
    if firm:
        info.provider_company = firm
        return

    try:
        crops = _render_stamp_crops(pdf_path)
        ocr_lines = _ocr_logo_images(crops)
        parsed = parse_firm_from_ocr_lines(ocr_lines)
        if parsed:
            info.provider_company = parsed
    except Exception as exc:  # noqa: BLE001
        logger.warning("logo_firm: OCR fill failed for %s: %s", pdf_path.name, exc)
