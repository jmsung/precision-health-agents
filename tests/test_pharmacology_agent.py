"""Tests for the PharmacologyAgent (mocked Anthropic API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bioai.agents.pharmacology import PharmacologyAgent, _build_clinical_context
from bioai.models import AgentStatus, PharmacologyFindings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TRANS_CONTEXT = {
    "transcriptomics": {
        "status": "success",
        "findings": {
            "pathway_scores": {
                "inflammation_immune": 0.8,
                "beta_cell_stress": 0.3,
                "insulin_resistance": 0.6,
                "fibrosis_ecm": 0.1,
                "oxidative_mitochondrial": 0.4,
            },
            "dominant_pathway": "inflammation_immune",
            "active_pathways": ["inflammation_immune", "insulin_resistance"],
            "diabetes_confirmed": {"confirmed": True, "confidence": "high", "reasoning": "3-layer agreement"},
            "diabetes_subtype": {"subtype": "inflammation_dominant", "confidence": "high"},
            "complication_risks": [
                {"complication": "cardiovascular", "severity": "high", "evidence": "inflammation + oxidative"},
            ],
            "monitoring": {"level": "actionable"},
            "recommendation": "pharmacology",
            "interpretation": "Active inflammatory pathway suggests GLP-1 RA therapy.",
        },
    },
}

_FULL_CONTEXT = {
    **_TRANS_CONTEXT,
    "genomics": {
        "status": "success",
        "findings": {
            "predicted_class": "DMT2",
            "confidence": 0.92,
            "risk_level": "high",
        },
    },
    "doctor": {
        "status": "success",
        "findings": {
            "prediction": "Diabetic",
            "probability": 0.85,
            "risk_level": "high",
        },
    },
}


def _make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tool_1"):
    """Create a mock Claude response that requests a tool call."""
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    resp.content = [block]
    return resp


def _make_text_response(text: str):
    """Create a mock Claude response with a text reply."""
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# Context building tests
# ---------------------------------------------------------------------------

class TestBuildClinicalContext:
    def test_no_context_returns_default(self):
        result = _build_clinical_context(None)
        assert "No transcriptomics findings" in result

    def test_empty_context_returns_default(self):
        result = _build_clinical_context({})
        assert "No transcriptomics findings" in result

    def test_transcriptomics_context_includes_subtype(self):
        result = _build_clinical_context(_TRANS_CONTEXT)
        assert "inflammation_dominant" in result

    def test_full_context_includes_all_agents(self):
        result = _build_clinical_context(_FULL_CONTEXT)
        assert "Genomics" in result
        assert "Doctor" in result
        assert "inflammation_dominant" in result
        assert "cardiovascular" in result

    def test_context_includes_pathway_scores(self):
        result = _build_clinical_context(_TRANS_CONTEXT)
        assert "inflammation_immune" in result

    def test_context_includes_complication_risks(self):
        result = _build_clinical_context(_TRANS_CONTEXT)
        assert "cardiovascular" in result
        assert "high" in result


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestPharmacologyAgent:
    @patch("bioai.agents.pharmacology.anthropic.Anthropic")
    def test_chat_calls_tool_and_returns_plan(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # First call: Claude asks to call recommend_medications
        # Second call: Claude returns text plan
        mock_client.messages.create.side_effect = [
            _make_tool_use_response("recommend_medications", {
                "diabetes_subtype": "inflammation_dominant",
                "complication_risks": [{"complication": "cardiovascular", "severity": "high"}],
            }),
            _make_text_response(
                "Based on your molecular profile, I recommend Liraglutide as your primary medication. "
                "It targets inflammation while protecting your heart."
            ),
        ]

        agent = PharmacologyAgent(context=_FULL_CONTEXT)
        reply = agent.chat("What medications do you recommend for my diabetes?")

        assert "Liraglutide" in reply
        assert mock_client.messages.create.call_count == 2

    @patch("bioai.agents.pharmacology.anthropic.Anthropic")
    def test_findings_populated_after_tool_call(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_tool_use_response("recommend_medications", {
                "diabetes_subtype": "inflammation_dominant",
                "complication_risks": [{"complication": "cardiovascular", "severity": "high"}],
            }),
            _make_text_response("Here is your medication plan."),
        ]

        agent = PharmacologyAgent(context=_FULL_CONTEXT)
        agent.chat("Recommend medications.")

        findings = agent.findings
        assert findings is not None
        assert isinstance(findings, PharmacologyFindings)
        assert findings.diabetes_subtype == "inflammation_dominant"
        assert len(findings.primary_medications) + len(findings.supportive_medications) > 0

    @patch("bioai.agents.pharmacology.anthropic.Anthropic")
    def test_result_success_after_tool_call(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_tool_use_response("recommend_medications", {
                "diabetes_subtype": "inflammation_dominant",
            }),
            _make_text_response("Your personalized plan is ready."),
        ]

        agent = PharmacologyAgent(context=_FULL_CONTEXT)
        agent.chat("Go ahead.")
        result = agent.result()

        assert result.agent == "pharmacology"
        assert result.status == AgentStatus.SUCCESS
        assert result.findings is not None
        assert result.error is None

    @patch("bioai.agents.pharmacology.anthropic.Anthropic")
    def test_result_error_without_tool_call(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_text_response("I need more information before recommending."),
        ]

        agent = PharmacologyAgent(context=_FULL_CONTEXT)
        agent.chat("Hello")
        result = agent.result()

        assert result.status == AgentStatus.ERROR
        assert result.findings is None
        assert result.error is not None
