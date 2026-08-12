"""End-to-end inference pipeline orchestration.

``run_pipeline`` is the pure processing flow (PDF in, structured results out).
``process_project`` is the background-task entrypoint: it owns a DB session,
tracks project status (processing / done / failed), and persists results.

Flow:

    PDF -+-> requirement extract (text) -> ``<name> requirement.pdf``
         +-> technical symbol extract -> ``<name> technical symbol.pdf``
         +-> images -+-> ROI detect -> OCR -> metadata mapping
                     +-> symbol detect -> classify -> counts
                                  -> cost calculation -> SQLite
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from config import Settings, get_settings
from db.models import DetectionInstance, DeviceLine, Page, Project, ProjectStatus
from ml.base import ClassifiedDevice, OcrLine, PageResult
from ml.classifier import get_device_classifier
from ml.cost_calculator import CostingSummary, calculate_costs, load_cost_rates
from ml.location_tagger import tag_devices_with_rooms
from ml.metadata_mapper import ExtractedMetadata, map_metadata
from ml.ocr import get_ocr_engine
from ml.pdf_to_image import convert_pdf_to_images
from ml.pricing import build_pricing_lookup
from ml.requirement_extractor import RequirementInfo, extract_requirement_from_pdf
from ml.requirement_pdf import generate_requirement_pdf, requirement_pdf_filename
from ml.roi_detector import crop_roi, get_roi_detector
from ml.scale_calibration import (
    apply_linear_feet,
    calibrate_from_ocr,
    is_line_type_box,
)
from ml.symbol_detector import get_symbol_detector
from ml.symbol_table_extractor import SymbolTableInfo, extract_symbol_table_from_pdf
from ml.symbol_table_pdf import (
    generate_technical_symbol_pdf,
    technical_symbol_pdf_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline extracted from one PDF."""

    metadata: ExtractedMetadata
    pages: list[PageResult] = field(default_factory=list)
    costing: CostingSummary | None = None
    requirement: RequirementInfo | None = None
    requirement_pdf_path: str | None = None
    technical_symbol: SymbolTableInfo | None = None
    technical_symbol_pdf_path: str | None = None
    pages_truncated: bool = False

    @property
    def devices(self) -> list[ClassifiedDevice]:
        return [device for page in self.pages for device in page.devices]


def _extract_requirement_stage(
    pdf_path: Path,
    source_filename: str,
    settings: Settings,
) -> tuple[RequirementInfo | None, str | None]:
    """Stage 0: extract requirement metadata/scope and write the requirement PDF.

    Non-fatal — returns (None, None) on failure so the CV pipeline can continue.
    """
    try:
        info = extract_requirement_from_pdf(pdf_path)
        out_name = requirement_pdf_filename(source_filename)
        out_path = settings.requirements_dir / out_name
        generate_requirement_pdf(info, source_filename, out_path)
        logger.info(
            "Requirement PDF written to %s (provider=%s, scope_sections=%d)",
            out_path,
            info.provider_summary,
            len(info.scope_sections),
        )
        return info, str(out_path)
    except Exception:
        logger.exception("Requirement extraction failed for %s", pdf_path)
        return None, None


def _extract_technical_symbol_stage(
    pdf_path: Path,
    source_filename: str,
    settings: Settings,
) -> tuple[SymbolTableInfo | None, str | None]:
    """Stage 0b: extract TECHNOLOGY SYMBOL / SYMBOL LEGEND tables (full PDF).

    Non-fatal — returns (None, None) on failure. Still writes a PDF when the
    scan succeeds even if zero entries were found (document records the miss).
    """
    try:
        info = extract_symbol_table_from_pdf(pdf_path)
        out_name = technical_symbol_pdf_filename(source_filename)
        out_path = settings.technical_symbols_dir / out_name
        generate_technical_symbol_pdf(info, source_filename, out_path)
        logger.info(
            "Technical symbol PDF written to %s (entries=%d, legend_pages=%s)",
            out_path,
            info.entry_count,
            info.legend_pages,
        )
        return info, str(out_path)
    except Exception:
        logger.exception("Technical symbol extraction failed for %s", pdf_path)
        return None, None


def run_pipeline(
    pdf_path: Path,
    pages_dir: Path,
    settings: Settings,
    source_filename: str | None = None,
) -> PipelineResult:
    """Run the full extraction pipeline over a PDF. Raises on CV/pipeline failure."""
    filename = source_filename or pdf_path.name
    started = time.perf_counter()

    # Requirement + technical-symbol text extracts and page rasterization overlap.
    with ThreadPoolExecutor(max_workers=3) as pool:
        req_future = pool.submit(
            _extract_requirement_stage, pdf_path, filename, settings
        )
        sym_future = pool.submit(
            _extract_technical_symbol_stage, pdf_path, filename, settings
        )
        render_future = pool.submit(
            convert_pdf_to_images,
            pdf_path,
            pages_dir,
            settings.pdf_dpi,
            settings.max_pdf_pages,
            settings.poppler_path,
        )
        requirement, requirement_pdf_path = req_future.result()
        technical_symbol, technical_symbol_pdf_path = sym_future.result()
        image_paths = render_future.result()

    pages_truncated = False
    if requirement is not None and requirement.total_pages > settings.max_pdf_pages:
        pages_truncated = True
    elif (
        technical_symbol is not None
        and technical_symbol.total_pages > settings.max_pdf_pages
    ):
        pages_truncated = True
    elif requirement is None and technical_symbol is None:
        # Best-effort truncation flag without a full text pass.
        try:
            import fitz

            with fitz.open(str(pdf_path)) as doc:
                pages_truncated = doc.page_count > settings.max_pdf_pages
        except Exception:
            pages_truncated = False

    logger.info(
        "Stage ready for %s: requirement=%s, technical_symbol=%s, "
        "rendered_pages=%d (%.1fs)",
        filename,
        "ok" if requirement_pdf_path else "skipped",
        "ok" if technical_symbol_pdf_path else "skipped",
        len(image_paths),
        time.perf_counter() - started,
    )

    roi_detector = get_roi_detector(settings)
    ocr_engine = get_ocr_engine(settings)
    symbol_detector = get_symbol_detector(settings)
    classifier = get_device_classifier(settings)

    pages: list[PageResult] = []
    all_ocr_lines: list[OcrLine] = []
    crops_dir = pages_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for page_number, image_path in enumerate(image_paths, start=1):
        page = PageResult(page_number=page_number, image_path=str(image_path))
        with Image.open(image_path) as image:
            image.load()

            page.roi = roi_detector.detect(image)
            if page.roi is not None:
                roi_crop = crop_roi(image, page.roi)
                page.ocr_lines = ocr_engine.extract(roi_crop)
                all_ocr_lines.extend(page.ocr_lines)

            # Full-page OCR (when available) feeds room labels + scale bar.
            # Mock OCR has no boxes; location_tagger falls back to quadrant rooms.
            page_ocr = list(page.ocr_lines)
            try:
                page_ocr = ocr_engine.extract(image)
            except Exception:
                logger.debug("Full-page OCR skipped on page %d", page_number, exc_info=True)

            scale = calibrate_from_ocr(
                page_ocr,
                dpi=float(settings.pdf_dpi),
                use_mock_fallback=settings.use_mock_models,
            )

            width, height = image.size
            raw_devices: list[ClassifiedDevice] = []
            for det_index, detection in enumerate(symbol_detector.detect(image)):
                box = detection.box.clamp(width, height)
                crop = image.crop(box.to_int_tuple())
                if detection.label == "raceway" or is_line_type_box(box):
                    device_type, confidence = "raceway", detection.score
                else:
                    device_type, confidence = classifier.classify(crop)

                crop_name = f"p{page_number:03d}_d{det_index:04d}.png"
                crop_path = crops_dir / crop_name
                crop.save(crop_path)

                raw_devices.append(
                    ClassifiedDevice(
                        device_type=device_type,
                        confidence=confidence,
                        detection=detection,
                        page_number=page_number,
                        crop_path=str(crop_path),
                        quantity=1.0,
                        unit="EA",
                    )
                )

            tagged = tag_devices_with_rooms(
                raw_devices,
                page_ocr,
                page_width=float(width),
                page_height=float(height),
            )
            page.devices = apply_linear_feet(tagged, scale)
        pages.append(page)
        logger.info(
            "Page %d: %d OCR lines, %d device symbols (scale=%s %s)",
            page_number,
            len(page.ocr_lines),
            len(page.devices),
            scale.label,
            scale.source,
        )

    metadata = map_metadata(all_ocr_lines)

    # Prefer requirement-extract metadata when the mock OCR title-block is empty
    # or when real drawings carry richer cover-sheet fields.
    if requirement is not None:
        metadata = ExtractedMetadata(
            title=metadata.title or requirement.project_title,
            client=metadata.client or requirement.client,
            architect=metadata.architect or requirement.provider_company,
            engineer=metadata.engineer
            or (requirement.consultants[0].firm if requirement.consultants else None),
            project_address=metadata.project_address or requirement.site_address,
            due_date=metadata.due_date or requirement.date,
        )

    result = PipelineResult(
        metadata=metadata,
        pages=pages,
        requirement=requirement,
        requirement_pdf_path=requirement_pdf_path,
        technical_symbol=technical_symbol,
        technical_symbol_pdf_path=technical_symbol_pdf_path,
        pages_truncated=pages_truncated,
    )

    rates = load_cost_rates(settings.cost_rates_path)
    pricing_lookup = None
    try:
        from db.session import SessionLocal

        pricing_session = SessionLocal()
        try:
            pricing_lookup = build_pricing_lookup(
                pricing_session,
                currency=rates.currency,
                default_unit_cost=rates.default_unit_cost,
            )
            # Stamp mfg / part / category from the catalog onto each detection
            enriched: list[ClassifiedDevice] = []
            from dataclasses import replace

            for device in result.devices:
                item = pricing_lookup.by_device_type.get(device.device_type)
                if item is None:
                    enriched.append(device)
                    continue
                enriched.append(
                    replace(
                        device,
                        mfg=device.mfg or item.mfg,
                        part_number=device.part_number or item.part_number,
                        category=device.category or item.category,
                    )
                )
            # Write enriched devices back onto pages
            by_page: dict[int, list[ClassifiedDevice]] = {}
            for device in enriched:
                by_page.setdefault(device.page_number, []).append(device)
            for page in result.pages:
                page.devices = by_page.get(page.page_number, [])
        finally:
            pricing_session.close()
    except Exception:
        logger.exception("Pricing lookup unavailable; falling back to JSON rates")

    result.costing = calculate_costs(result.devices, rates, pricing=pricing_lookup)
    logger.info(
        "Pipeline finished for %s in %.1fs (%d pages, %d devices)",
        filename,
        time.perf_counter() - started,
        len(pages),
        len(result.devices),
    )
    return result


def process_project(project_id: str) -> None:
    """Background-task entrypoint: process an uploaded project end to end.

    Never raises — failures are recorded on the project row so the frontend
    can surface them.
    """
    from db.session import SessionLocal

    settings = get_settings()
    session = SessionLocal()
    try:
        project = session.get(Project, project_id)
        if project is None:
            logger.error("Project %s vanished before processing", project_id)
            return

        project.status = ProjectStatus.PROCESSING
        session.commit()

        try:
            pages_dir = settings.storage_dir / project.id / "pages"
            result = run_pipeline(
                Path(project.file_path),
                pages_dir,
                settings,
                source_filename=project.filename,
            )
            _persist_result(session, project, result)
            project.status = ProjectStatus.DONE
            project.error_message = None
        except Exception as exc:
            logger.exception("Pipeline failed for project %s", project_id)
            project.status = ProjectStatus.FAILED
            project.error_message = str(exc)
        session.commit()
    finally:
        session.close()


def _persist_result(session, project: Project, result: PipelineResult) -> None:  # type: ignore[no-untyped-def]
    metadata = result.metadata
    project.title = metadata.title
    project.client = metadata.client
    project.architect = metadata.architect
    project.engineer = metadata.engineer
    project.project_address = metadata.project_address
    project.due_date = metadata.due_date
    project.requirement_pdf_path = result.requirement_pdf_path
    project.requirement_provider = (
        result.requirement.provider_summary if result.requirement else None
    )
    project.technical_symbol_pdf_path = result.technical_symbol_pdf_path
    project.pages_truncated = result.pages_truncated

    project.pages.clear()
    for page in result.pages:
        project.pages.append(
            Page(page_number=page.page_number, image_path=page.image_path)
        )

    # Persist every detection (bbox + crop) before aggregating lines
    project.detections.clear()
    flat_devices = result.devices
    detection_rows: list[DetectionInstance] = []
    for device in flat_devices:
        box = device.detection.box
        row = DetectionInstance(
            page_number=device.page_number,
            device_type=device.device_type,
            confidence=device.confidence,
            x1=box.x1,
            y1=box.y1,
            x2=box.x2,
            y2=box.y2,
            crop_path=device.crop_path,
            room_label=device.room_label,
            quantity=device.quantity,
            unit=device.unit,
            mfg=device.mfg,
            part_number=device.part_number,
            category=device.category,
            needs_review=device.confidence < 0.85,
        )
        detection_rows.append(row)
        project.detections.append(row)

    session.flush()  # assign detection IDs

    project.device_lines.clear()
    assert result.costing is not None
    for line in result.costing.lines:
        sample_id = None
        if line.sample_device_index is not None and 0 <= line.sample_device_index < len(
            detection_rows
        ):
            sample_id = detection_rows[line.sample_device_index].id
        project.device_lines.append(
            DeviceLine(
                device_type=line.device_type,
                display_name=line.display_name,
                count=line.count,
                unit_cost=line.unit_cost,
                detected_count=line.count,
                default_unit_cost=line.unit_cost,
                needs_review=line.needs_review,
                category=line.category,
                unit=line.unit,
                mfg=line.mfg,
                part_number=line.part_number,
                locations=line.locations,
                sample_detection_id=sample_id,
            )
        )

    # Link detections back to their aggregated line when possible
    line_by_key = {
        (dl.category, dl.device_type, dl.mfg or "", dl.part_number or ""): dl
        for dl in project.device_lines
    }
    for det in project.detections:
        key = (det.category or "Uncategorized", det.device_type, det.mfg or "", det.part_number or "")
        match = line_by_key.get(key)
        if match is not None:
            det.device_line_id = match.id

