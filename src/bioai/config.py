"""Configuration and settings."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    model: str = "claude-sonnet-4-20250514"
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
