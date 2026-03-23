"""
Application configuration via environment variables / .env file.
All settings have sensible defaults so the app works out of the box
with just an API key set.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # allow both field name and alias at construction time
    )

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0

    # API keys — pydantic-settings auto-maps anthropic_api_key → ANTHROPIC_API_KEY env var
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Paths
    pdf_path: str = "table_full.pdf"
    db_path: str = "fedex_rates.db"

    # Pipeline
    concurrency: int = Field(default=5, ge=1, le=20)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_wait_seconds: float = Field(default=2.0, ge=0.0)
    batch_size: int = Field(default=100, ge=1)

    # Page range (1-indexed, inclusive). None = use PDF total page count.
    page_start: int = Field(default=1, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> "Settings":
        if self.page_end is not None and self.page_end < self.page_start:
            raise ValueError(
                f"page_end ({self.page_end}) must be >= page_start ({self.page_start})"
            )
        return self


# Singleton loaded once at import time.
# Individual modules import this and use settings.xyz.
settings = Settings()
