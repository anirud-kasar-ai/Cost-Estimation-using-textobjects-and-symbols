"""Minimal shared types for extraction modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float = 1.0
