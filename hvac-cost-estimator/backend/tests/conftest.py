"""Shared pytest fixtures.

Tests run against a temporary SQLite database and temporary storage dir, and
always use mock model backends — no trained weights or heavy ML dependencies
are required.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic single-page HVAC layout PDF."""
    from scripts.generate_sample_pdf import generate_sample_pdf

    path = tmp_path_factory.mktemp("pdfs") / "sample_layout.pdf"
    return generate_sample_pdf(path)


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh Settings bound to a temp storage dir + temp SQLite file."""
    from config import Settings, get_settings

    monkeypatch.setenv("HVAC_USE_MOCK_MODELS", "true")
    monkeypatch.setenv("HVAC_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv(
        "HVAC_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    )
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture()
def db_session(settings) -> Iterator["Session"]:  # noqa: F821 - forward ref
    """A session on a fresh temp database with all tables created."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from db.models import Base

    engine = create_engine(
        settings.database_url, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
