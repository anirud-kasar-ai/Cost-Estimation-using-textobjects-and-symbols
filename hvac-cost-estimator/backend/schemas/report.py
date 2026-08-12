"""Pydantic models for the costing report endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.project import ProjectMetadata


class ReportLine(BaseModel):
    device_type: str
    display_name: str
    count: int
    unit_cost: float
    line_total: float
    needs_review: bool
    category: str = "Uncategorized"
    unit: str = "EA"
    mfg: str | None = None
    part_number: str | None = None
    locations: str | None = None


class CostingReport(BaseModel):
    """The consolidated report: metadata + device line items + grand total."""

    project_id: str
    filename: str
    metadata: ProjectMetadata
    lines: list[ReportLine]
    grand_total: float
    currency: str
