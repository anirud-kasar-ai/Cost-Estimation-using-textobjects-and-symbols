"""Tests for the text-extraction branch: ROI detection, OCR, metadata mapping."""

from __future__ import annotations

from PIL import Image

from ml.metadata_mapper import ExtractedMetadata, map_metadata, normalize_date
from ml.ocr import MockOcrEngine
from ml.roi_detector import MockRoiDetector, crop_roi


class TestMockRoiDetector:
    def test_returns_right_hand_strip(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        detection = MockRoiDetector().detect(image)

        assert detection is not None
        assert detection.label == "title_block"
        assert detection.box.x1 == 1000 * (1 - MockRoiDetector.ROI_WIDTH_FRACTION)
        assert detection.box.x2 == 1000
        assert (detection.box.y1, detection.box.y2) == (0, 700)

    def test_crop_roi_matches_box(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        detection = MockRoiDetector().detect(image)
        assert detection is not None

        crop = crop_roi(image, detection)
        assert crop.height == 700
        assert crop.width == 1000 - int(detection.box.x1)


class TestMockOcr:
    def test_emits_title_block_lines(self) -> None:
        lines = MockOcrEngine().extract(Image.new("RGB", (300, 800), "white"))
        texts = [line.text for line in lines]

        assert "CLIENT" in texts
        assert "CONSULTANT" in texts
        assert all(line.confidence > 0.9 for line in lines)


class TestMetadataMapper:
    def test_stacked_layout_from_mock_ocr(self) -> None:
        lines = MockOcrEngine().extract(Image.new("RGB", (300, 800), "white"))
        metadata = map_metadata(lines)

        assert metadata.title == "Riverside Office Tower - Level 3 HVAC Layout"
        assert metadata.client == "Meridian Property Group"
        assert metadata.architect == "Atelier North Architects"
        # "CONSULTANT" label maps to the engineer field
        assert metadata.engineer == "Vector Building Services Ltd."
        assert metadata.project_address == "128 Riverside Drive, Springfield"
        # "ISSUE DATE" maps to due_date, normalized to ISO
        assert metadata.due_date == "2026-08-15"

    def test_inline_colon_layout(self) -> None:
        metadata = map_metadata(
            [
                "Client: ACME Corp",
                "Architect: Studio 9",
                "Due Date: 15/08/2026",
            ]
        )
        assert metadata.client == "ACME Corp"
        assert metadata.architect == "Studio 9"
        assert metadata.due_date == "2026-08-15"

    def test_fuzzy_matches_ocr_noise_in_label(self) -> None:
        metadata = map_metadata(["CL1ENT", "Northwind Holdings"])
        assert metadata.client == "Northwind Holdings"

    def test_combined_architect_engineer_label(self) -> None:
        metadata = map_metadata(["ARCHITECT / ENGINEER", "Omni Design-Build LLP"])
        assert metadata.architect == "Omni Design-Build LLP"
        assert metadata.engineer == "Omni Design-Build LLP"

    def test_unrelated_lines_are_ignored(self) -> None:
        metadata = map_metadata(["DRAWING NO", "M-301", "SCALE", "1:100"])
        assert metadata == ExtractedMetadata()

    def test_first_match_wins(self) -> None:
        metadata = map_metadata(["CLIENT", "First Corp", "OWNER", "Second Corp"])
        assert metadata.client == "First Corp"


class TestNormalizeDate:
    def test_common_formats(self) -> None:
        assert normalize_date("2026-08-15") == "2026-08-15"
        assert normalize_date("15/08/2026") == "2026-08-15"
        assert normalize_date("15 Aug 2026") == "2026-08-15"
        assert normalize_date("August 15, 2026") == "2026-08-15"

    def test_unparseable_returns_raw(self) -> None:
        assert normalize_date("TBC") == "TBC"
