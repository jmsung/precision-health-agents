"""Tests for DoctorAgent — mocked Anthropic API."""

from unittest.mock import MagicMock, patch

from bioai.agents.doctor import DoctorAgent
from bioai.models import AgentStatus, Recommendation, RiskLevel


# ---------------------------------------------------------------------------
# Helpers to build fake Anthropic responses
# ---------------------------------------------------------------------------

def _text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def _tool_use_response(tool_input: dict):
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "classify_diabetes"
    tool_block.id = "tool_abc123"
    tool_block.input = tool_input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _print_exchange(patient: str, doctor: str):
    print(f"\n  Patient : {patient}")
    print(f"  Doctor  : {doctor}")


def _print_findings(agent: DoctorAgent):
    f = agent.findings
    if f:
        print(f"\n  --- Findings ---")
        print(f"  Prediction   : {f.prediction}")
        print(f"  Probability  : {f.probability:.0%}")
        print(f"  Risk level   : {f.risk_level.value}")
        print(f"  Recommend    : {f.recommendation.value}")


_SAMPLE_INPUT = {
    "pregnancies": 1,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("bioai.agents.doctor.anthropic.Anthropic")
def test_chat_returns_text_reply(mock_anthropic):
    """Agent returns a string reply to a patient message."""
    print("\n[test_chat_returns_text_reply]")

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.return_value = _text_response("Hello, how can I help you today?")

    agent = DoctorAgent()
    patient_msg = "Hi doctor, I am here for a checkup."
    reply = agent.chat(patient_msg)

    _print_exchange(patient_msg, reply)

    assert isinstance(reply, str)
    assert len(reply) > 0


@patch("bioai.agents.doctor.anthropic.Anthropic")
@patch("bioai.agents.doctor.classify_diabetes")
def test_tool_called_and_findings_set(mock_classify, mock_anthropic):
    """Agent calls classify_diabetes and stores findings when tool_use is triggered."""
    print("\n[test_tool_called_and_findings_set]")

    mock_classify.return_value = {
        "prediction": "Diabetic",
        "probability": 0.78,
        "risk_level": "high",
    }

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _tool_use_response(_SAMPLE_INPUT),
        _text_response("Based on your results, I recommend seeing a specialist."),
    ]

    agent = DoctorAgent()
    patient_msg = "Here are all my details."
    reply = agent.chat(patient_msg)

    _print_exchange(patient_msg, reply)
    print(f"\n  [Tool called] classify_diabetes({_SAMPLE_INPUT})")
    _print_findings(agent)

    mock_classify.assert_called_once_with(**_SAMPLE_INPUT)
    assert agent.findings is not None
    assert agent.findings.prediction == "Diabetic"
    assert agent.findings.risk_level == RiskLevel.HIGH
    assert agent.findings.recommendation == Recommendation.HOSPITAL


@patch("bioai.agents.doctor.anthropic.Anthropic")
@patch("bioai.agents.doctor.classify_diabetes")
def test_low_risk_recommends_health_trainer(mock_classify, mock_anthropic):
    """Low-risk result maps to health_trainer recommendation."""
    print("\n[test_low_risk_recommends_health_trainer]")

    mock_classify.return_value = {
        "prediction": "Non-Diabetic",
        "probability": 0.15,
        "risk_level": "low",
    }

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _tool_use_response(_SAMPLE_INPUT),
        _text_response("Great news — you are low risk. I recommend a health trainer."),
    ]

    agent = DoctorAgent()
    patient_msg = "Here are my values."
    reply = agent.chat(patient_msg)

    _print_exchange(patient_msg, reply)
    _print_findings(agent)

    assert agent.findings.recommendation == Recommendation.HEALTH_TRAINER
    assert agent.findings.risk_level == RiskLevel.LOW


@patch("bioai.agents.doctor.anthropic.Anthropic")
@patch("bioai.agents.doctor.classify_diabetes")
def test_result_returns_agent_result(mock_classify, mock_anthropic):
    """result() returns a well-formed AgentResult after conversation."""
    print("\n[test_result_returns_agent_result]")

    mock_classify.return_value = {
        "prediction": "Diabetic",
        "probability": 0.82,
        "risk_level": "high",
    }

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _tool_use_response(_SAMPLE_INPUT),
        _text_response("Please visit a hospital."),
    ]

    agent = DoctorAgent()
    patient_msg = "All my data is ready."
    summary = agent.chat(patient_msg)
    ar = agent.result(summary=summary)

    _print_exchange(patient_msg, summary)
    _print_findings(agent)
    print(f"\n  AgentResult → agent={ar.agent!r}, status={ar.status.value}, error={ar.error}")

    assert ar.agent == "doctor"
    assert ar.status == AgentStatus.SUCCESS
    assert ar.findings is not None
    assert ar.error is None


@patch("bioai.agents.doctor.anthropic.Anthropic")
@patch("bioai.agents.doctor.classify_diabetes")
def test_full_conversation_collects_all_8_features(mock_classify, mock_anthropic):
    """Simulate a realistic multi-turn conversation where the agent gathers all 8
    clinical features across several exchanges before calling the tool.

    Turn 1: patient introduces themselves
    Turn 2: patient gives age + gender
    Turn 3: patient gives glucose + blood pressure
    Turn 4: patient gives weight/height (BMI) + pregnancies
    Turn 5: patient gives family history + insulin
    → Claude now has all 8 values → tool_use → final recommendation
    """
    print("\n[test_full_conversation_collects_all_8_features]")
    print("  --- Full intake conversation ---")

    mock_classify.return_value = {
        "prediction": "Diabetic",
        "probability": 0.74,
        "risk_level": "high",
    }

    collected_input = {
        "pregnancies": 2,
        "glucose": 160,
        "blood_pressure": 80,
        "skin_thickness": 28,
        "insulin": 0,
        "bmi": 31.2,
        "diabetes_pedigree_function": 0.5,
        "age": 42,
    }

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _text_response("Hello! I'm your doctor. Can you tell me your age and biological sex?"),
        _text_response("Thank you. Do you have any recent blood test results, such as your glucose level or blood pressure?"),
        _text_response("Got it. Could you share your height and weight so I can calculate your BMI? And how many times have you been pregnant?"),
        _text_response("Almost done. Does anyone in your immediate family have diabetes? And do you know your insulin level from any recent test?"),
        _tool_use_response(collected_input),
        _text_response(
            "Based on your results, your diabetes risk is elevated. "
            "I strongly recommend visiting a specialist at a hospital for further evaluation."
        ),
    ]

    agent = DoctorAgent()

    exchanges = [
        "Hi, I'd like a diabetes checkup.",
        "I'm 42 years old, female.",
        "My glucose was 160 mg/dL and blood pressure 80 mm Hg.",
        "I'm 165 cm and 85 kg, and I've been pregnant twice.",
        "My mother has type 2 diabetes, and I don't know my insulin level.",
    ]

    replies = []
    for msg in exchanges:
        reply = agent.chat(msg)
        replies.append(reply)
        _print_exchange(msg, reply)

    print(f"\n  [Tool called] classify_diabetes({collected_input})")
    _print_findings(agent)

    # Tool must have been called exactly once with all 8 features
    mock_classify.assert_called_once()
    call_kwargs = mock_classify.call_args[1]
    assert set(call_kwargs.keys()) == {
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree_function", "age",
    }

    # Conversation history:
    #   5 user messages + 4 text assistant replies
    #   + 1 tool_use assistant message + 1 tool_result user message
    #   + 1 final assistant text reply = 12
    assert len(agent._messages) == 12

    # Findings must be populated
    assert agent.findings is not None
    assert agent.findings.recommendation == Recommendation.HOSPITAL

    # All replies are non-empty strings
    for reply in replies:
        assert isinstance(reply, str)
        assert len(reply) > 0
