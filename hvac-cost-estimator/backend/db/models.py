"""SQLAlchemy ORM models (SQLite)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class ProjectStatus(str, enum.Enum):
    """Lifecycle of an uploaded drawing through the pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Project(Base):
    """One uploaded HVAC layout PDF and its extracted results."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(Text)
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus, values_callable=lambda e: [m.value for m in e]),
        default=ProjectStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Title-block metadata extracted by the text branch
    title: Mapped[str | None] = mapped_column(String(255), default=None)
    client: Mapped[str | None] = mapped_column(String(255), default=None)
    architect: Mapped[str | None] = mapped_column(String(255), default=None)
    engineer: Mapped[str | None] = mapped_column(String(255), default=None)
    project_address: Mapped[str | None] = mapped_column(String(500), default=None)
    due_date: Mapped[str | None] = mapped_column(String(64), default=None)

    # Requirement extract (provider + scope PDF generated on upload)
    requirement_pdf_path: Mapped[str | None] = mapped_column(Text, default=None)
    requirement_provider: Mapped[str | None] = mapped_column(String(500), default=None)
    pages_truncated: Mapped[bool] = mapped_column(default=False)

    pages: Mapped[list[Page]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Page.page_number"
    )
    device_lines: Mapped[list[DeviceLine]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="DeviceLine.device_type"
    )


class Page(Base):
    """A rendered page image of the uploaded PDF."""

    __tablename__ = "pages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str] = mapped_column(Text)

    project: Mapped[Project] = relationship(back_populates="pages")


class DeviceLine(Base):
    """A costed line item: one detected device type with count and unit cost.

    ``count`` and ``unit_cost`` are editable from the dashboard; the original
    pipeline values are kept in ``detected_count`` / ``default_unit_cost`` so
    overrides remain visible and reversible.
    """

    __tablename__ = "device_lines"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))

    device_type: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(150))
    count: Mapped[int] = mapped_column(Integer)
    unit_cost: Mapped[float] = mapped_column(Float)
    detected_count: Mapped[int] = mapped_column(Integer)
    default_unit_cost: Mapped[float] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(default=False)

    project: Mapped[Project] = relationship(back_populates="device_lines")

    @property
    def line_total(self) -> float:
        return round(self.count * self.unit_cost, 2)
