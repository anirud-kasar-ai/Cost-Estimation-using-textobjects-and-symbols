"""Basic batch discovery tests (no vision LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from batch import discover_pdfs  # noqa: E402
from pipeline.job import job_id_from_filename  # noqa: E402


REAL_DATA = Path(r"D:\Cost Estimation Using Text and Object\Real data")


def test_job_id_sanitizes_windows_chars():
    assert job_id_from_filename("1948BIDDrawings.pdf") == "1948BIDDrawings"
    assert job_id_from_filename("foo<bar>.pdf") == "foo_bar_"


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="Real data folder not present")
def test_discover_pdfs_finds_bid_sets():
    pdfs = discover_pdfs(REAL_DATA)
    assert len(pdfs) >= 3
    names = {p.name for p in pdfs}
    assert "1948BIDDrawings.pdf" in names
