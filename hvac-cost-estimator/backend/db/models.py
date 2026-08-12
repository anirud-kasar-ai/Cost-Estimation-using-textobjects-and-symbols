"""SQLAlchemy ORM models (SQLite)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
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

    # Technical symbol legend table PDF generated on upload
    technical_symbol_pdf_path: Mapped[str | None] = mapped_column(Text, default=None)

    pages: Mapped[list[Page]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Page.page_number"
    )
    device_lines: Mapped[list[DeviceLine]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="DeviceLine.category, DeviceLine.device_type",
    )
    detections: Mapped[list[DetectionInstance]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="DetectionInstance.page_number",
    )


class Page(Base):
    """A rendered page image of the uploaded PDF."""

    __tablename__ = "pages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str] = mapped_column(Text)

    project: Mapped[Project] = relationship(back_populates="pages")


class PricingItem(Base):
    """Editable unit-cost catalog keyed by manufacturer + part number."""

    __tablename__ = "pricing_items"
    __table_args__ = (UniqueConstraint("mfg", "part_number", name="uq_pricing_mfg_part"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    mfg: Mapped[str] = mapped_column(String(120))
    part_number: Mapped[str] = mapped_column(String(120))
    display_name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(120), default="Uncategorized")
    unit: Mapped[str] = mapped_column(String(16), default="EA")  # EA | LF
    unit_cost: Mapped[float] = mapped_column(Float)
    device_type: Mapped[str | None] = mapped_column(String(100), default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


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

    category: Mapped[str] = mapped_column(String(120), default="Uncategorized")
    unit: Mapped[str] = mapped_column(String(16), default="EA")
    mfg: Mapped[str | None] = mapped_column(String(120), default=None)
    part_number: Mapped[str | None] = mapped_column(String(120), default=None)
    locations: Mapped[str | None] = mapped_column(Text, default=None)
    sample_detection_id: Mapped[str | None] = mapped_column(String(32), default=None)

    project: Mapped[Project] = relationship(back_populates="device_lines")

    @property
    def line_total(self) -> float:
        return round(self.count * self.unit_cost, 2)


class DetectionInstance(Base):
    """One detection+classification result with bounding box and sheet crop."""

    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    device_line_id: Mapped[str | None] = mapped_column(String(32), default=None)

    page_number: Mapped[int] = mapped_column(Integer)
    device_type: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    crop_path: Mapped[str | None] = mapped_column(Text, default=None)
    room_label: Mapped[str | None] = mapped_column(String(120), default=None)
    quantity: Mapped[float] = mapped_column(Float, default=1.0)
    unit: Mapped[str] = mapped_column(String(16), default="EA")
    mfg: Mapped[str | None] = mapped_column(String(120), default=None)
    part_number: Mapped[str | None] = mapped_column(String(120), default=None)
    category: Mapped[str | None] = mapped_column(String(120), default=None)
    needs_review: Mapped[bool] = mapped_column(default=False)

    project: Mapped[Project] = relationship(back_populates="detections")
