"""Pydantic schemas for the editable SQLite pricing catalog."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PricingItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    mfg: str
    part_number: str
    display_name: str
    category: str
    unit: str
    unit_cost: float
    device_type: str | None = None
    updated_at: datetime | None = None


class PricingItemCreate(BaseModel):
    mfg: str = Field(min_length=1, max_length=120)
    part_number: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=200)
    category: str = Field(default="Uncategorized", max_length=120)
    unit: str = Field(default="EA", pattern="^(EA|LF)$")
    unit_cost: float = Field(ge=0, le=10_000_000, allow_inf_nan=False)
    device_type: str | None = Field(default=None, max_length=100)


class PricingItemUpdate(BaseModel):
    mfg: str | None = Field(default=None, min_length=1, max_length=120)
    part_number: str | None = Field(default=None, min_length=1, max_length=120)
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    category: str | None = Field(default=None, max_length=120)
    unit: str | None = Field(default=None, pattern="^(EA|LF)$")
    unit_cost: float | None = Field(default=None, ge=0, le=10_000_000, allow_inf_nan=False)
    device_type: str | None = Field(default=None, max_length=100)
