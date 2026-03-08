"""Tests for HealthTrainerAgent — two-tool flow with clinical context injection."""

from unittest.mock import MagicMock, patch

import pytest

from bioai.agents.health_trainer import HealthTrainerAgent, _build_clinical_context
from bioai.models import AgentStatus, HealthTrainerFindings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_use_response(tool_id: str, tool_name: str, tool_input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def _make_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


_FAKE_CLASSIFICATION = {
    "suggested_type": "Strength",
    "experience_level": "Beginner",
    "bmi": 24.5,
    "all_scores": {"Cardio": 1.6, "Strength": 1.8, "Flexibility": 1.2, "HIIT": 0.8},
    "reasoning": "DMT2 genetic risk. ADA recommends Strength + Cardio combo.",
}

_FAKE_EXERCISES = [
    {
        "Name": "Barbell Squat", "Type": "Strength", "BodyPart": "Legs",
        "Equipment": "Barbell", "Level": "Intermediate",
        "Description": "Squat with barbell", "Benefits": "Builds legs", "CaloriesPerMinute": 10,
    }
]

_DMT2_CONTEXT = {
    "genomics": {
        "status": "success",
        "findings": {"predicted_class": "DMT2", "confidence": 0.82, "risk_level": "high"},
    },
    "doctor": {
        "status": "success",
        "findings": {"prediction": "Diabetic", "probability": 0.71, "risk_level": "high"},
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent():
    with patch("bioai.agents.health_trainer.anthropic.Anthropic"):
        a = HealthTrainerAgent()
        a._client = MagicMock()
        return a


@pytest.fixture()
def agent_with_context():
    with patch("bioai.agents.health_trainer.anthropic.Anthropic"):
        a = HealthTrainerAgent(context=_DMT2_CONTEXT)
        a._client = MagicMock()
        return a


# ---------------------------------------------------------------------------
# Clinical context injection
# ---------------------------------------------------------------------------

def test_clinical_context_injected_into_system_prompt():
    with patch("bioai.agents.health_trainer.anthropic.Anthropic"):
        a = HealthTrainerAgent(context=_DMT2_CONTEXT)
    assert "DMT2" in a._system
    assert "0.71" in a._system or "71%" in a._system


def test_no_context_uses_fallback_message():
    with patch("bioai.agents.health_trainer.anthropic.Anthropic"):
        a = HealthTrainerAgent(context=None)
    assert "No prior clinical findings" in a._system


def test_build_clinical_context_with_full_context():
    block = _build_clinical_context(_DMT2_CONTEXT)
    assert "DMT2" in block
    assert "Diabetic" in block


def test_build_clinical_context_with_no_context():
    block = _build_clinical_context(None)
    assert "No prior clinical findings" in block


# ---------------------------------------------------------------------------
# Full two-tool conversation flow
# ---------------------------------------------------------------------------

def test_full_flow_calls_both_tools(agent):
    with (
        patch("bioai.agents.health_trainer.classify_workout_type") as mock_classify,
        patch("bioai.agents.health_trainer.recommend_exercises") as mock_recommend,
    ):
        mock_classify.return_value = _FAKE_CLASSIFICATION
        mock_recommend.return_value = {
            "exercises": _FAKE_EXERCISES, "total_found": 1, "filters_applied": {},
        }

        classify_input = {
            "age": 45, "gender": "Female", "weight_kg": 70, "height_cm": 165,
            "workout_frequency_per_week": 1, "session_duration_hours": 0.5,
            "diabetes_type": "DMT2", "diabetes_probability": 0.71,
        }
        recommend_input = {"exercise_type": "Strength", "difficulty": "Beginner"}

        agent._client.messages.create.side_effect = [
            _make_tool_use_response("c1", "classify_workout_type", classify_input),
            _make_tool_use_response("c2", "recommend_exercises", recommend_input),
            _make_text_response("Here is your personalised Strength plan for managing diabetes."),
        ]

        reply = agent.chat("I have been referred by my doctor.")

        mock_classify.assert_called_once_with(**classify_input)
        mock_recommend.assert_called_once_with(**recommend_input)
        assert agent._classification == _FAKE_CLASSIFICATION
        assert len(agent._all_exercises) == 1
        assert "plan" in reply.lower() or reply


# ---------------------------------------------------------------------------
# findings and result
# ---------------------------------------------------------------------------

def test_findings_none_before_any_tool(agent):
    agent._client.messages.create.return_value = _make_text_response("What is your age?")
    agent.chat("Hello.")
    assert agent.findings is None


def test_findings_none_after_only_classify(agent):
    with patch("bioai.agents.health_trainer.classify_workout_type") as mock_classify:
        mock_classify.return_value = _FAKE_CLASSIFICATION
        agent._client.messages.create.side_effect = [
            _make_tool_use_response("c1", "classify_workout_type", {
                "age": 30, "gender": "Male", "weight_kg": 80, "height_cm": 180,
                "workout_frequency_per_week": 0, "session_duration_hours": 0,
            }),
            _make_text_response("What equipment do you have?"),
        ]
        agent.chat("I'm 30, male, 80kg, 180cm, no exercise history.")
    # exercises not yet fetched
    assert agent.findings is None


def test_findings_populated_after_both_tools(agent):
    with (
        patch("bioai.agents.health_trainer.classify_workout_type") as mock_classify,
        patch("bioai.agents.health_trainer.recommend_exercises") as mock_recommend,
    ):
        mock_classify.return_value = _FAKE_CLASSIFICATION
        mock_recommend.return_value = {
            "exercises": _FAKE_EXERCISES, "total_found": 1, "filters_applied": {},
        }
        agent._client.messages.create.side_effect = [
            _make_tool_use_response("c1", "classify_workout_type", {
                "age": 45, "gender": "Male", "weight_kg": 90, "height_cm": 178,
                "workout_frequency_per_week": 2, "session_duration_hours": 0.75,
            }),
            _make_tool_use_response("c2", "recommend_exercises", {"exercise_type": "Strength"}),
            _make_text_response("Here is your plan."),
        ]
        agent.chat("Start.")

    f = agent.findings
    assert f is not None
    assert isinstance(f, HealthTrainerFindings)
    assert f.fitness_level == "beginner"
    assert len(f.recommended_exercises) == 1


def test_result_success_after_full_flow(agent):
    agent._classification = _FAKE_CLASSIFICATION
    agent._all_exercises = _FAKE_EXERCISES
    agent._plan_summary = "Your weekly plan."
    result = agent.result()
    assert result.status == AgentStatus.SUCCESS
    assert result.agent == "health_trainer"
    assert result.findings is not None


def test_result_error_when_incomplete(agent):
    result = agent.result()
    assert result.status == AgentStatus.ERROR
    assert result.error is not None
