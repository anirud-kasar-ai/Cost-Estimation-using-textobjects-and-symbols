"""Application configuration.

All settings are loaded from environment variables (prefix ``HVAC_``) or a
``.env`` file, with local-friendly defaults. Relative paths are resolved
against the ``backend/`` directory so the app behaves the same regardless of
the current working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Runtime settings, overridable via environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="HVAC_",
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core
    use_mock_models: bool = True
    database_url: str = "sqlite:///./storage/hvac.db"
    storage_dir: Path = Path("./storage")

    # PDF conversion — keep modest for POC speed (mock models don't need 300 DPI).
    # 8 pages @ 120 DPI ≈ 3–4s render on large bid sets; 40 @ 300 DPI can exceed 2 min.
    pdf_dpi: int = 120
    poppler_path: Path | None = None

    # Upload validation. max_upload_mb unset/None = no size limit.
    max_upload_mb: int | None = None
    # Max pages to RENDER for CV / device costing. Requirement text still scans
    # the full PDF; pages beyond this are truncated (not rejected).
    max_pdf_pages: int = 8

    # Model weights (used only when use_mock_models is False)
    roi_model_path: Path = Path("./models/roi_detector.pth")
    roi_model_config: Path = Path("./models/roi_detector.yaml")
    symbol_model_path: Path = Path("./models/symbol_detector.pth")
    symbol_model_config: Path = Path("./models/symbol_detector.yaml")
    classifier_model_path: Path = Path("./models/device_classifier.onnx")

    # Cost data
    cost_rates_path: Path = Path("./data/cost_rates.json")

    # CORS origins for the Vite dev server
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator(
        "storage_dir",
        "roi_model_path",
        "roi_model_config",
        "symbol_model_path",
        "symbol_model_config",
        "classifier_model_path",
        "cost_rates_path",
        "poppler_path",
        mode="after",
    )
    @classmethod
    def _resolve_relative_to_backend(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value if value.is_absolute() else (BACKEND_DIR / value).resolve()

    @field_validator("database_url", mode="after")
    @classmethod
    def _resolve_sqlite_path(cls, value: str) -> str:
        """Anchor relative SQLite file paths to the backend directory."""
        prefix = "sqlite:///"
        if value.startswith(prefix):
            raw = value[len(prefix) :]
            path = Path(raw)
            if not path.is_absolute():
                path = (BACKEND_DIR / raw).resolve()
            return f"{prefix}{path.as_posix()}"
        return value

    @property
    def max_upload_bytes(self) -> int | None:
        """Upload size limit in bytes, or None when uploads are unlimited."""
        if self.max_upload_mb is None:
            return None
        return self.max_upload_mb * 1024 * 1024

    @property
    def requirements_dir(self) -> Path:
        """Folder for generated ``<name> requirement.pdf`` files."""
        return self.storage_dir / "requirements"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (import this, not a module-level instance)."""
    return Settings()
