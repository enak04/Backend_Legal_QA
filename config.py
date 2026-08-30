"""
Application configuration loaded from environment variables.

All secrets and service URLs are configured here — never hardcoded.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-driven application settings.

    Values are read from a ``.env`` file (if present) and can be
    overridden by actual environment variables.
    """

    # ── Legal_QA Service ──────────────────────────────────────
    legal_qa_base_url: str = "http://localhost:8000"
    legal_qa_timeout: int = 120  # seconds – ML inference is slow

    # ── Database ──────────────────────────────────────────────
    database_url: str = "sqlite:///./conversations.db"

    # ── CORS ──────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # ── Server ────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8080

    # ── OpenAI API ────────────────────────────────────────────
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
