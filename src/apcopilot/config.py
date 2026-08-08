from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

LLMMode = Literal["live", "replay", "record", "off"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APCOPILOT_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    anthropic_api_key: str | None = None

    llm_mode: LLMMode = "live"
    extract_model: str = "claude-haiku-4-5"
    extract_retry_model: str = "claude-sonnet-5"
    approval_model: str = "claude-sonnet-5"
    critic_model: str = "claude-sonnet-5"

    # Pinned rather than datetime.now(): the corpus is dated Jan 2026 with Feb-Mar
    # due dates, so a live clock would mark every invoice past-due and make the
    # eval drift over time.
    as_of_date: date = date(2026, 2, 1)

    data_dir: Path = REPO_ROOT / "data"
    seed_dir: Path = REPO_ROOT / "data" / "seed"
    invoice_dir: Path = REPO_ROOT / "data" / "invoices"
    var_dir: Path = REPO_ROOT / "var"
    fixture_dir: Path = REPO_ROOT / "tests" / "fixtures" / "llm"

    log_json: bool = True

    @property
    def db_path(self) -> Path:
        return self.var_dir / "app.db"

    @property
    def log_path(self) -> Path:
        return self.var_dir / "logs" / "run.jsonl"

    @property
    def policy_path(self) -> Path:
        return self.seed_dir / "policies.yaml"

    def resolved_api_key(self) -> str | None:
        import os

        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    def ensure_dirs(self) -> None:
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
