"""Integration test: Genomics Agent → Doctor Agent pipeline.

Simulates a patient who:
  1. Has their DNA tested by the Genomics Agent → DMT2 detected (high genetic risk)
  2. Because genetic risk is high, is referred to the Doctor Agent
  3. Has a multi-turn conversation with the Doctor Agent to gather clinical data
  4. Receives a confirmed diabetes diagnosis and hospital recommendation
"""

import asyncio
from unittest.mock import MagicMock, patch

from precision_health_agents.agents.doctor import DoctorAgent
from precision_health_agents.agents.genomics import GenomicsAgent
from precision_health_agents.models import AgentStatus, Recommendation, RiskLevel


# ---------------------------------------------------------------------------
# Helpers — Genomics Agent mock responses
# ---------------------------------------------------------------------------

def _genomics_tool_use(sequence: str):
    block = MagicMock()
    block.type = "tool_use"
    block.id = "genomics_tool_1"
    block.name = "classify_dna"
    block.input = {"sequence": sequence}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def _genomics_text(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Helpers — Doctor Agent mock responses
# ---------------------------------------------------------------------------

def _doctor_text(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def _doctor_tool_use(tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_diabetes"
    block.id = "doctor_tool_1"
    block.input = tool_input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Patient profile used across the test
# ---------------------------------------------------------------------------

_DNA_SEQUENCE = "ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

_CLINICAL_VALUES = {
    "pregnancies": 2,
    "glucose": 160,
    "blood_pressure": 82,
    "skin_thickness": 30,
    "insulin": 0,
    "bmi": 31.5,
    "diabetes_pedigree_function": 0.5,   # one parent has diabetes
    "age": 42,
}


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@patch("precision_health_agents.agents.genomics.anthropic.Anthropic")
@patch("precision_health_agents.agents.genomics.classify_dna")
@patch("precision_health_agents.agents.doctor.anthropic.Anthropic")
@patch("precision_health_agents.agents.doctor.classify_diabetes")
def test_dna_positive_then_doctor_confirms(
    mock_clinical_classify,
    mock_doctor_anthropic,
    mock_dna_classify,
    mock_genomics_anthropic,
):
    """Full pipeline: DMT2 DNA → high risk → Doctor intake → Diabetic → hospital."""

    print("\n" + "=" * 60)
    print("INTEGRATION TEST: DNA → Doctor Pipeline")
    print("=" * 60)

    # ── Step 1: Genomics Agent ─────────────────────────────────────
    print("\n[Step 1] Genomics Agent — DNA classification")

    mock_dna_classify.return_value = {
        "predicted_class": "DMT2",
        "probabilities": {"DMT1": 0.05, "DMT2": 0.88, "NONDM": 0.07},
        "confidence": 0.88,
    }

    mock_genomics_client = MagicMock()
    mock_genomics_anthropic.return_value = mock_genomics_client
    mock_genomics_client.messages.create.side_effect = [
        _genomics_tool_use(_DNA_SEQUENCE),
        _genomics_text(
            "The DNA sequence shows a strong Type 2 Diabetes (DMT2) pattern "
            "with 88% confidence. This patient has significant genetic predisposition "
            "to Type 2 Diabetes. Clinical evaluation is strongly recommended."
        ),
    ]

    genomics_agent = GenomicsAgent()
    genomics_agent._client = mock_genomics_client
    genomics_result = asyncio.run(genomics_agent.analyze(
        f"Analyze this DNA sequence for diabetes risk: {_DNA_SEQUENCE}"
    ))

    print(f"  DNA class   : {genomics_result.findings.predicted_class}")
    print(f"  Confidence  : {genomics_result.findings.confidence:.0%}")
    print(f"  Risk level  : {genomics_result.findings.risk_level.value}")
    print(f"  Summary     : {genomics_result.summary}")

    assert genomics_result.status == AgentStatus.SUCCESS
    assert genomics_result.findings.predicted_class == "DMT2"
    assert genomics_result.findings.risk_level == RiskLevel.HIGH

    # ── Routing decision ──────────────────────────────────────────
    genetic_risk = genomics_result.findings.risk_level
    print(f"\n[Routing] Genetic risk = {genetic_risk.value} → referring to Doctor Agent")
    assert genetic_risk == RiskLevel.HIGH  # only HIGH triggers doctor referral

    # ── Step 2: Doctor Agent ───────────────────────────────────────
    print("\n[Step 2] Doctor Agent — clinical intake conversation")

    mock_clinical_classify.return_value = {
        "prediction": "Diabetic",
        "probability": 0.76,
        "risk_level": "high",
    }

    mock_doctor_client = MagicMock()
    mock_doctor_client.messages.create.side_effect = [
        _doctor_text("Hello! I'm your doctor. Can you tell me your age?"),
        _doctor_text("Thank you. Do you have any recent blood glucose or blood pressure results?"),
        _doctor_text("Got it. Can you share your height and weight, and how many times you've been pregnant?"),
        _doctor_text("Almost there. Any family history of diabetes? And do you know your insulin level?"),
        _doctor_tool_use(_CLINICAL_VALUES),
        _doctor_text(
            "Based on your clinical measurements and your DNA results showing Type 2 "
            "Diabetes genetic markers, I strongly recommend visiting a hospital specialist. "
            "Your risk is confirmed at both the genetic and clinical level."
        ),
    ]

    doctor_agent = DoctorAgent()
    doctor_agent._client = mock_doctor_client

    conversation = [
        ("Hi doctor. My DNA test showed I might have Type 2 Diabetes risk.",
         "Hello! I'm your doctor. Can you tell me your age?"),
        ("I'm 42 years old, female.",
         "Thank you. Do you have any recent blood glucose or blood pressure results?"),
        ("My glucose was 160 mg/dL and blood pressure around 82.",
         "Got it. Can you share your height and weight, and how many times you've been pregnant?"),
        ("I'm 168 cm, 89 kg, and I've been pregnant twice.",
         "Almost there. Any family history of diabetes? And do you know your insulin level?"),
        ("My father has Type 2 Diabetes. I don't know my insulin level.",
         None),   # final turn triggers tool + recommendation
    ]

    print()
    for patient_msg, expected_reply in conversation:
        reply = doctor_agent.chat(patient_msg)
        print(f"  Patient : {patient_msg}")
        print(f"  Doctor  : {reply}")
        print()
        if expected_reply is not None:
            assert reply == expected_reply

    # ── Step 3: Combined result ────────────────────────────────────
    print("[Step 3] Combined DNA + Clinical result")

    clinical_findings = doctor_agent.findings
    assert clinical_findings is not None

    print(f"  DNA result        : {genomics_result.findings.predicted_class} "
          f"({genomics_result.findings.confidence:.0%} confidence)")
    print(f"  Clinical result   : {clinical_findings.prediction} "
          f"({clinical_findings.probability:.0%} probability)")
    print(f"  Risk level        : {clinical_findings.risk_level.value}")
    print(f"  Recommendation    : {clinical_findings.recommendation.value}")

    # Both DNA and clinical agree → confirmed diabetes → hospital
    assert clinical_findings.prediction == "Diabetic"
    assert clinical_findings.risk_level == RiskLevel.HIGH
    assert clinical_findings.recommendation == Recommendation.HOSPITAL

    # Confirm classify_diabetes was called with the correct 8 features
    mock_clinical_classify.assert_called_once()
    call_kwargs = mock_clinical_classify.call_args[1]
    assert set(call_kwargs.keys()) == {
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree_function", "age",
    }

    print("\n[Decision] DNA=DMT2 (high) + Clinical=Diabetic (high) → HOSPITAL")
    print("           Recommended drug class: Metformin / GLP-1 agonists (Type 2)")
    print("=" * 60)
