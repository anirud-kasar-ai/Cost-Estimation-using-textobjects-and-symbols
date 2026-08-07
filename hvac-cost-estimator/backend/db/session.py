"""Engine/session factory and FastAPI dependency for database access."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from db.models import Base


def _create_engine() -> Engine:
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        # SQLite: allow use across FastAPI's threadpool + background tasks.
        connect_args={"check_same_thread": False}
        if settings.database_url.startswith("sqlite")
        else {},
    )
    if settings.database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_fk(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine: Engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create all tables (idempotent). Called on app startup.

    Also adds any newly introduced SQLite columns for this POC (no Alembic).
    """
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "sqlite":
        _ensure_sqlite_columns()


def _ensure_sqlite_columns() -> None:
    """Best-effort ALTER TABLE for columns added after the initial POC schema."""
    additions = (
        ("projects", "requirement_pdf_path", "TEXT"),
        ("projects", "requirement_provider", "VARCHAR(500)"),
        ("projects", "pages_truncated", "BOOLEAN DEFAULT 0"),
    )
    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(projects)").fetchall()
        }
        for table, column, coltype in additions:
            if column not in existing:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"
                )


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
