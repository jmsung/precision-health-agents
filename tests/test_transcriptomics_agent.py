"""Tests for the TranscriptomicsAgent (mocked Anthropic API)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from bioai.agents.transcriptomics import TranscriptomicsAgent
from bioai.config import Settings
from bioai.models import AgentStatus, RiskLevel


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


class TestTranscriptomicsAgent:

    def test_analyze_with_tool_call(self):
        settings = _make_settings()
        agent = TranscriptomicsAgent(settings=settings)

        gene_expr = {"TNF": 500.0, "IL6": 300.0, "INS": 100.0}

        tool_response = _mock_tool_use_response(
            "analyze_gene_expression",
            {"gene_expression": gene_expr},
        )
        text_response = _mock_text_response(
            "The analysis shows elevated inflammatory markers."
        )

        with patch.object(agent._client.messages, "create", side_effect=[tool_response, text_response]):
            result = asyncio.run(agent.analyze(
                f"Analyze this gene expression profile: {gene_expr}"
            ))

        assert result.status == AgentStatus.SUCCESS
        assert result.agent == "transcriptomics"
        assert result.findings is not None
        assert result.findings.pathway_scores is not None
        assert "inflammation_immune" in result.findings.pathway_scores
        assert result.summary == "The analysis shows elevated inflammatory markers."

    def test_analyze_without_tool_call(self):
        settings = _make_settings()
        agent = TranscriptomicsAgent(settings=settings)

        text_response = _mock_text_response(
            "I need gene expression data to analyze."
        )

        with patch.object(agent._client.messages, "create", return_value=text_response):
            result = asyncio.run(agent.analyze("What pathways are relevant to diabetes?"))

        assert result.status == AgentStatus.SUCCESS
        assert result.findings is None
        assert "gene expression data" in result.summary

    def test_analyze_with_context(self):
        settings = _make_settings()
        agent = TranscriptomicsAgent(settings=settings)

        context = {
            "genomics": {"predicted_class": "DMT2", "confidence": 0.95},
            "doctor": {"prediction": "Diabetic", "probability": 0.82},
        }

        text_response = _mock_text_response("Analysis with context.")

        with patch.object(agent._client.messages, "create", return_value=text_response) as mock_create:
            result = asyncio.run(agent.analyze(
                "Analyze gene expression.",
                context=context,
            ))

        call_kwargs = mock_create.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system", "")
        assert "DMT2" in system_prompt
        assert "Diabetic" in system_prompt

    def test_analyze_handles_error(self):
        settings = _make_settings()
        agent = TranscriptomicsAgent(settings=settings)

        with patch.object(agent._client.messages, "create", side_effect=Exception("API error")):
            result = asyncio.run(agent.analyze("test query"))

        assert result.status == AgentStatus.ERROR
        assert result.error == "API error"

    def test_findings_have_correct_types(self):
        settings = _make_settings()
        agent = TranscriptomicsAgent(settings=settings)

        gene_expr = {"TNF": 500.0, "IL6": 300.0}
        tool_response = _mock_tool_use_response(
            "analyze_gene_expression",
            {"gene_expression": gene_expr},
        )
        text_response = _mock_text_response("Done.")

        with patch.object(agent._client.messages, "create", side_effect=[tool_response, text_response]):
            result = asyncio.run(agent.analyze(f"Analyze: {gene_expr}"))

        f = result.findings
        assert isinstance(f.pathway_scores, dict)
        assert isinstance(f.dominant_pathway, str)
        assert isinstance(f.active_pathways, list)
        assert isinstance(f.risk_level, RiskLevel)
        assert isinstance(f.dysregulated_genes, list)
        assert isinstance(f.interpretation, str)
