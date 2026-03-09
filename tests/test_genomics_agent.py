"""Tests for the GenomicsAgent — verifies it calls the DNA classifier tool."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from precision_health_agents.agents.genomics import GenomicsAgent
from precision_health_agents.models import AgentStatus, RiskLevel

_SAMPLE_SEQ = "ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
_QUERY = f"Analyze this DNA sequence for diabetes risk: {_SAMPLE_SEQ}"


def _make_tool_use_response(call_id: str, sequence: str):
    block = MagicMock()
    block.type = "tool_use"
    block.id = call_id
    block.name = "classify_dna"
    block.input = {"sequence": sequence}

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def _make_text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


@pytest.fixture
def agent():
    return GenomicsAgent()


@patch("precision_health_agents.agents.genomics.anthropic.Anthropic")
def test_agent_calls_dna_classifier_tool(mock_anthropic_cls, agent):
    """Agent should invoke classify_dna and return structured GenomicsFindings."""
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.side_effect = [
        _make_tool_use_response("call_1", _SAMPLE_SEQ),
        _make_text_response("The sequence indicates DMT2 risk with high confidence."),
    ]
    agent._client = mock_client

    result = asyncio.run(agent.analyze(_QUERY))

    assert result.status == AgentStatus.SUCCESS
    assert result.agent == "genomics"
    assert result.findings is not None
    assert result.findings.predicted_class in {"DMT1", "DMT2", "NONDM"}
    assert result.findings.risk_level in {RiskLevel.HIGH, RiskLevel.LOW}
    assert 0.0 <= result.findings.confidence <= 1.0
    assert result.findings.interpretation != ""


@patch("precision_health_agents.agents.genomics.anthropic.Anthropic")
def test_agent_returns_summary(mock_anthropic_cls, agent):
    """Agent should include a text summary in the result."""
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.side_effect = [
        _make_tool_use_response("call_1", _SAMPLE_SEQ),
        _make_text_response("High risk for Type 2 Diabetes based on genomic pattern."),
    ]
    agent._client = mock_client

    result = asyncio.run(agent.analyze(_QUERY))

    assert "High risk" in result.summary


@patch("precision_health_agents.agents.genomics.anthropic.Anthropic")
def test_agent_returns_error_on_failure(mock_anthropic_cls, agent):
    """Agent should return error status if the API call fails."""
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.side_effect = Exception("API timeout")
    agent._client = mock_client

    result = asyncio.run(agent.analyze(_QUERY))

    assert result.status == AgentStatus.ERROR
    assert "API timeout" in result.error
