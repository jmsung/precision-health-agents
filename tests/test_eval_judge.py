"""Tests for LLM-as-judge evaluation."""

import asyncio
import json
from unittest.mock import MagicMock, patch

from precision_health_agents.eval.cases import EvalCase, ExpectedOutput
from precision_health_agents.eval.judge import JudgeScore, judge_agent
from precision_health_agents.models import (
    AgentResult,
    AgentStatus,
    GenomicsFindings,
    RiskLevel,
)


def _genomics_result() -> AgentResult:
    return AgentResult(
        agent="genomics",
        status=AgentStatus.SUCCESS,
        findings=GenomicsFindings(
            predicted_class="DMT2",
            confidence=0.85,
            probabilities={"DMT1": 0.05, "DMT2": 0.85, "NONDM": 0.10},
            risk_level=RiskLevel.HIGH,
            interpretation="Strong Type 2 pattern detected.",
        ),
        summary="DNA analysis indicates DMT2 with 85% confidence.",
    )


def _case() -> EvalCase:
    return EvalCase(
        id="case-1",
        name="Confirmed Diabetic",
        description="Clinical positive + DMT2 DNA → hospital.",
        expected=ExpectedOutput(
            dna_class="DMT2",
            clinical_prediction="Diabetic",
            decision="hospital",
        ),
    )


def _mock_judge_response(scores: dict) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(scores)

    response = MagicMock()
    response.content = [block]
    return response


@patch("precision_health_agents.eval.judge.anthropic.Anthropic")
def test_judge_returns_valid_scores(mock_anthropic_cls):
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_judge_response(
        {
            "relevance": 4,
            "completeness": 5,
            "accuracy": 4,
            "safety": 5,
            "explanation": "Good interpretation of DMT2 classification.",
        }
    )

    score = asyncio.run(judge_agent(_genomics_result(), _case()))

    assert isinstance(score, JudgeScore)
    assert 1 <= score.relevance <= 5
    assert 1 <= score.completeness <= 5
    assert 1 <= score.accuracy <= 5
    assert 1 <= score.safety <= 5
    assert len(score.explanation) > 0


@patch("precision_health_agents.eval.judge.anthropic.Anthropic")
def test_judge_sends_agent_output_in_prompt(mock_anthropic_cls):
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_judge_response(
        {
            "relevance": 3,
            "completeness": 3,
            "accuracy": 3,
            "safety": 5,
            "explanation": "Adequate.",
        }
    )

    asyncio.run(judge_agent(_genomics_result(), _case()))

    call_args = mock_client.messages.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    user_msg = messages[0]["content"]
    assert "DMT2" in user_msg
    assert "case-1" in user_msg


@patch("precision_health_agents.eval.judge.anthropic.Anthropic")
def test_judge_handles_api_error(mock_anthropic_cls):
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.side_effect = Exception("API error")

    score = asyncio.run(judge_agent(_genomics_result(), _case()))

    assert score.relevance == 0
    assert score.safety == 0
    assert "error" in score.explanation.lower()
