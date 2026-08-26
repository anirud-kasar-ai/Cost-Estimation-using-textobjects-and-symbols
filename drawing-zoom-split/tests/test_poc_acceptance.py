"""POC acceptance gate — requires processed 1948BIDDrawings job on disk."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.poc_acceptance_test import run_acceptance  # noqa: E402


@pytest.mark.integration
def test_poc_acceptance_1948_bid_drawings():
    job_root = ROOT / "storage" / "jobs" / "1948BIDDrawings"
    if not (job_root / "job.json").is_file():
        pytest.skip("1948BIDDrawings job not present — run full pipeline first")
    report = run_acceptance("1948BIDDrawings")
    failures = [c for c in report.checks if not c.passed]
    assert not failures, "\n".join(
        f"{c.stage}/{c.name}: {c.detail}" for c in failures
    )
