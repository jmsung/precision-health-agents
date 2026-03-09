"""Tests for the ProteomicsAgent (mocked Anthropic API)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from precision_health_agents.agents.proteomics import ProteomicsAgent
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
    "biomarker_scores": {"CRP": 0.85, "TNF_alpha": 0.72, "IL6": 0.65},
    "elevated_biomarkers": ["CRP", "TNF_alpha"],
    "biomarker_panel": "inflammatory",
    "risk_level": "high",
    "complication_evidence": [
        {"complication": "cardiovascular", "severity": "moderate", "evidence_proteins": "CRP, TNF_alpha"},
    ],
    "diabetes_confirmed": {"confirmed": True, "confidence": "high", "reasoning": "Elevated inflammatory markers"},
    "interpretation": "Elevated inflammatory protein biomarkers consistent with insulin resistance.",
}


class TestProteomicsAgent:

    @patch("precision_health_agents.agents.proteomics.analyze_protein_biomarkers")
    def test_analyze_with_tool_call(self, mock_tool):
        mock_tool.return_value = _MOCK_TOOL_RESULT
        settings = _make_settings()
        agent = ProteomicsAgent(settings=settings)

        protein_levels = {"CRP": 8.5, "TNF_alpha": 45.2, "IL6": 12.1}
        tool_response = _mock_tool_use_response(
            "analyze_protein_biomarkers",
            {"protein_levels": protein_levels},
        )
        text_response = _mock_text_response(
            "The analysis shows elevated inflammatory biomarkers."
        )

        with patch.object(agent._client.messages, "create", side_effect=[tool_response, text_response]):
            result = asyncio.run(agent.analyze(
                f"Analyze these protein biomarkers: {protein_levels}"
            ))

        assert result.status == AgentStatus.SUCCESS
        assert result.agent == "proteomics"
        assert result.findings is not None
        assert result.findings.biomarker_scores == {"CRP": 0.85, "TNF_alpha": 0.72, "IL6": 0.65}
        assert "CRP" in result.findings.elevated_biomarkers
        assert result.summary == "The analysis shows elevated inflammatory biomarkers."

    def test_analyze_without_tool_call(self):
        settings = _make_settings()
        agent = ProteomicsAgent(settings=settings)

        text_response = _mock_text_response(
            "I need protein biomarker data to analyze."
        )

        with patch.object(agent._client.messages, "create", return_value=text_response):
            result = asyncio.run(agent.analyze("What biomarkers are relevant?"))

        assert result.status == AgentStatus.SUCCESS
        assert result.findings is None
        assert "protein biomarker data" in result.summary

    def test_analyze_with_context(self):
        settings = _make_settings()
        agent = ProteomicsAgent(settings=settings)

        context = {
            "genomics": {"predicted_class": "DMT2", "confidence": 0.95},
            "doctor": {"prediction": "Diabetic", "probability": 0.82},
            "transcriptomics": {"dominant_pathway": "inflammation_immune", "risk_level": "high"},
        }

        text_response = _mock_text_response("Analysis with context.")

        with patch.object(agent._client.messages, "create", return_value=text_response) as mock_create:
            result = asyncio.run(agent.analyze(
                "Analyze protein biomarkers.",
                context=context,
            ))

        call_kwargs = mock_create.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system", "")
        assert "DMT2" in system_prompt
        assert "Diabetic" in system_prompt
        assert "inflammation_immune" in system_prompt

    def test_analyze_handles_error(self):
        settings = _make_settings()
        agent = ProteomicsAgent(settings=settings)

        with patch.object(agent._client.messages, "create", side_effect=Exception("API error")):
            result = asyncio.run(agent.analyze("test query"))

        assert result.status == AgentStatus.ERROR
        assert result.error == "API error"

    @patch("precision_health_agents.agents.proteomics.analyze_protein_biomarkers")
    def test_findings_have_correct_types(self, mock_tool):
        mock_tool.return_value = _MOCK_TOOL_RESULT
        settings = _make_settings()
        agent = ProteomicsAgent(settings=settings)

        protein_levels = {"CRP": 8.5, "TNF_alpha": 45.2}
        tool_response = _mock_tool_use_response(
            "analyze_protein_biomarkers",
            {"protein_levels": protein_levels},
        )
        text_response = _mock_text_response("Done.")

        with patch.object(agent._client.messages, "create", side_effect=[tool_response, text_response]):
            result = asyncio.run(agent.analyze(f"Analyze: {protein_levels}"))

        f = result.findings
        assert isinstance(f.biomarker_scores, dict)
        assert isinstance(f.elevated_biomarkers, list)
        assert isinstance(f.biomarker_panel, str)
        assert isinstance(f.risk_level, RiskLevel)
        assert isinstance(f.complication_evidence, list)
        assert isinstance(f.interpretation, str)
