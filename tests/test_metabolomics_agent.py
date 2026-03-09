"""Tests for the MetabolomicsAgent (mocked Anthropic API)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from precision_health_agents.agents.metabolomics import MetabolomicsAgent
from precision_health_agents.config import Settings
from precision_health_agents.models import AgentStatus, RiskLevel


def _make_settings() -> Settings:
    return Settings(api_key="test-key")


def _mock_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _mock_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "call_1") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


_MOCK_TOOL_RESULT = {
    "metabolite_scores": {"triglycerides": 0.92, "leucine": 0.78, "HDL": -0.65},
    "elevated_metabolites": ["triglycerides", "leucine"],
    "insulin_resistance_score": 0.82,
    "metabolic_pattern": "lipid_dysregulation",
    "risk_level": "high",
    "subtype_refinement": {"subtype": "metabolic_insulin_resistant", "confidence": "high", "reasoning": "Lipid + BCAA pattern"},
    "diabetes_confirmed": {"confirmed": True, "confidence": "high", "reasoning": "Severe metabolic dysregulation"},
    "interpretation": "Significant lipid dysregulation with elevated BCAAs indicating insulin resistance.",
}


class TestMetabolomicsAgent:

    @patch("precision_health_agents.agents.metabolomics.analyze_metabolic_profile")
    def test_analyze_with_tool_call(self, mock_tool):
        mock_tool.return_value = _MOCK_TOOL_RESULT
        settings = _make_settings()
        agent = MetabolomicsAgent(settings=settings)

        metabolite_levels = {"triglycerides": 250.0, "leucine": 180.5, "HDL": 35.0}
        tool_response = _mock_tool_use_response(
            "analyze_metabolic_profile",
            {"metabolite_levels": metabolite_levels},
        )
        text_response = _mock_text_response(
            "The analysis shows significant lipid dysregulation."
        )

        with patch.object(agent._client.messages, "create", side_effect=[tool_response, text_response]):
            result = asyncio.run(agent.analyze(
                f"Analyze this metabolic profile: {metabolite_levels}"
            ))

        assert result.status == AgentStatus.SUCCESS
        assert result.agent == "metabolomics"
        assert result.findings is not None
        assert result.findings.metabolite_scores == {"triglycerides": 0.92, "leucine": 0.78, "HDL": -0.65}
        assert "triglycerides" in result.findings.elevated_metabolites
        assert result.findings.insulin_resistance_score == 0.82
        assert result.summary == "The analysis shows significant lipid dysregulation."

    def test_analyze_without_tool_call(self):
        settings = _make_settings()
        agent = MetabolomicsAgent(settings=settings)

        text_response = _mock_text_response(
            "I need metabolite data to analyze."
        )

        with patch.object(agent._client.messages, "create", return_value=text_response):
            result = asyncio.run(agent.analyze("What metabolites are relevant?"))

        assert result.status == AgentStatus.SUCCESS
        assert result.findings is None
        assert "metabolite data" in result.summary

    def test_analyze_with_context(self):
        settings = _make_settings()
        agent = MetabolomicsAgent(settings=settings)

        context = {
            "genomics": {"predicted_class": "DMT2", "confidence": 0.95},
            "doctor": {"prediction": "Diabetic", "probability": 0.82},
            "transcriptomics": {"dominant_pathway": "insulin_resistance", "risk_level": "moderate"},
        }

        text_response = _mock_text_response("Analysis with context.")

        with patch.object(agent._client.messages, "create", return_value=text_response) as mock_create:
            result = asyncio.run(agent.analyze(
                "Analyze metabolic profile.",
                context=context,
            ))

        call_kwargs = mock_create.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system", "")
        assert "DMT2" in system_prompt
        assert "Diabetic" in system_prompt
        assert "insulin_resistance" in system_prompt

    def test_analyze_handles_error(self):
        settings = _make_settings()
        agent = MetabolomicsAgent(settings=settings)

        with patch.object(agent._client.messages, "create", side_effect=Exception("API error")):
            result = asyncio.run(agent.analyze("test query"))

        assert result.status == AgentStatus.ERROR
        assert result.error == "API error"

    @patch("precision_health_agents.agents.metabolomics.analyze_metabolic_profile")
    def test_findings_have_correct_types(self, mock_tool):
        mock_tool.return_value = _MOCK_TOOL_RESULT
        settings = _make_settings()
        agent = MetabolomicsAgent(settings=settings)

        metabolite_levels = {"triglycerides": 250.0, "leucine": 180.5}
        tool_response = _mock_tool_use_response(
            "analyze_metabolic_profile",
            {"metabolite_levels": metabolite_levels},
        )
        text_response = _mock_text_response("Done.")

        with patch.object(agent._client.messages, "create", side_effect=[tool_response, text_response]):
            result = asyncio.run(agent.analyze(f"Analyze: {metabolite_levels}"))

        f = result.findings
        assert isinstance(f.metabolite_scores, dict)
        assert isinstance(f.elevated_metabolites, list)
        assert isinstance(f.insulin_resistance_score, float)
        assert isinstance(f.metabolic_pattern, str)
        assert isinstance(f.risk_level, RiskLevel)
        assert isinstance(f.subtype_refinement, dict)
        assert isinstance(f.interpretation, str)
