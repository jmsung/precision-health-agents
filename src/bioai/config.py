"""Configuration and settings."""

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings."""

    # Models
    agent_model: str = "claude-sonnet-4-20250514"
    synthesis_model: str = "claude-opus-4-20250514"
    judge_model: str = "claude-sonnet-4-20250514"
    ralph_model: str = "claude-opus-4-20250514"

    # API
    api_key: str = ""
    max_tokens: int = 4096

    # Paths
    prompts_dir: Path = Path("src/bioai/prompts")
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
