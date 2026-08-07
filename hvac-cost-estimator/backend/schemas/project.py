"""Pydantic models for project endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from db.models import ProjectStatus


class ProjectMetadata(BaseModel):
    """Structured title-block metadata."""

    title: str | None = None
    client: str | None = None
    architect: str | None = None
    engineer: str | None = None
    project_address: str | None = None
    due_date: str | None = None


class DeviceLineOut(BaseModel):
    """One costed line item, including original pipeline values for override UX."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    device_type: str
    display_name: str
    count: int
    unit_cost: float
    detected_count: int
    default_unit_cost: float
    needs_review: bool
    line_total: float


class ProjectSummary(BaseModel):
    """List-view representation of a project."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: ProjectStatus
    error_message: str | None
    created_at: datetime


class ProjectDetail(ProjectSummary):
    """Full project view: metadata + costed device lines + totals."""

    metadata: ProjectMetadata
    device_lines: list[DeviceLineOut]
    grand_total: float
    currency: str
    page_count: int
    has_requirement_pdf: bool = False
    requirement_provider: str | None = None
    pages_truncated: bool = False


# Generous sanity caps for manual overrides; mirrored in the frontend
# (CostingReportTable.tsx). allow_inf_nan blocks 1e999/Infinity/NaN, which
# would otherwise slip past the ge/le comparisons and poison the totals.
MAX_LINE_COUNT = 100_000
MAX_UNIT_COST = 10_000_000


class DeviceLineUpdate(BaseModel):
    """Manual override of a line's count and/or unit cost."""

    count: int | None = Field(default=None, ge=0, le=MAX_LINE_COUNT)
    unit_cost: float | None = Field(
        default=None, ge=0, le=MAX_UNIT_COST, allow_inf_nan=False
    )


class UploadResponse(BaseModel):
    """Returned immediately after upload; processing continues in background."""

    project_id: str
    status: ProjectStatus
