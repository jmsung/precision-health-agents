"""Tests for precision_health_agents.config — Settings model."""

from pathlib import Path

from precision_health_agents.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings(api_key="test-key")
        assert s.agent_model == "claude-sonnet-4-20250514"
        assert s.synthesis_model == "claude-opus-4-20250514"
        assert s.judge_model == "claude-sonnet-4-20250514"
        assert s.ralph_model == "claude-opus-4-20250514"
        assert s.max_tokens == 4096
        assert s.prompts_dir == Path("src/precision_health_agents/prompts")
        assert s.data_dir == Path("data")

    def test_custom_values(self):
        s = Settings(
            api_key="key",
            agent_model="claude-haiku-4-5-20251001",
            max_tokens=2048,
        )
        assert s.agent_model == "claude-haiku-4-5-20251001"
        assert s.max_tokens == 2048

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        s = Settings.from_env()
        assert s.api_key == "env-key-123"

    def test_from_env_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = Settings.from_env()
        assert s.api_key == ""

    def test_backward_compat_model_field(self):
        """Old code used Settings().model — verify agent_model works."""
        s = Settings(api_key="k")
        assert s.agent_model == "claude-sonnet-4-20250514"
