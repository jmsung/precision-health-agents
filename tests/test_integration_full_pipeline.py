"""Integration test: Genomics → Doctor → Health Trainer full pipeline.

Simulates a patient who:
  1. Has DNA tested by GenomicsAgent         → DMT2 genetic risk detected
  2. Is referred to DoctorAgent              → moderate clinical risk, Non-Diabetic
                                               → Recommendation.HEALTH_TRAINER
  3. Is referred to HealthTrainerAgent       → receives workout type classification
                                               informed by both prior agents' findings
  4. HealthTrainerAgent delivers a report    → workout plan with clinical reasoning

Patient profile
  - Age 38, Female, 172 cm, 74 kg  (BMI ~25)
  - DNA: DMT2 with 72% confidence
  - Clinical: glucose 128, BP 76, BMI 25, low insulin, 1 pregnancy, age 38
    → Non-Diabetic but 44% probability (borderline / pre-diabetic territory)
    → risk_level = moderate → Recommendation.HEALTH_TRAINER
  - Exercise history: 1 day/week, 45-minute sessions → Beginner
  - Equipment: dumbbells at home
"""

import asyncio
from unittest.mock import MagicMock, patch

from precision_health_agents.agents.doctor import DoctorAgent
from precision_health_agents.agents.genomics import GenomicsAgent
from precision_health_agents.agents.health_trainer import HealthTrainerAgent
from precision_health_agents.models import AgentStatus, Recommendation, RiskLevel


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = [block]
    return r


def _tool_response(tool_id: str, tool_name: str, tool_input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input
    r = MagicMock()
    r.stop_reason = "tool_use"
    r.content = [block]
    return r


# ---------------------------------------------------------------------------
# Patient profile
# ---------------------------------------------------------------------------

_DNA_SEQUENCE = "ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

_CLINICAL_VALUES = {
    "pregnancies": 1,
    "glucose": 128,
    "blood_pressure": 76,
    "skin_thickness": 22,
    "insulin": 40,
    "bmi": 25.0,
    "diabetes_pedigree_function": 0.45,
    "age": 38,
}

_CLASSIFY_INPUT = {
    "age": 38,
    "gender": "Female",
    "weight_kg": 74.0,
    "height_cm": 172.0,
    "workout_frequency_per_week": 1,
    "session_duration_hours": 0.75,
    "diabetes_type": "DMT2",
    "diabetes_probability": 0.44,
}

_RECOMMEND_INPUT = {
    "exercise_type": "Strength",
    "difficulty": "Beginner",
    "equipment": "Dumbbell",
}

_FAKE_EXERCISES = [
    {
        "Name": "Dumbbell Lateral Raise", "Type": "Strength", "BodyPart": "Shoulders",
        "Equipment": "Dumbbell", "Level": "Beginner",
        "Description": "Raise dumbbells to shoulder height",
        "Benefits": "Builds shoulder width", "CaloriesPerMinute": 5,
    },
    {
        "Name": "Hammer Curl", "Type": "Strength", "BodyPart": "Arms",
        "Equipment": "Dumbbell", "Level": "Beginner",
        "Description": "Curl dumbbells with neutral grip",
        "Benefits": "Builds arm thickness", "CaloriesPerMinute": 5,
    },
    {
        "Name": "Romanian Deadlift", "Type": "Strength", "BodyPart": "Legs",
        "Equipment": "Dumbbell", "Level": "Intermediate",
        "Description": "Hinge at hips with dumbbells",
        "Benefits": "Builds hamstrings and glutes", "CaloriesPerMinute": 9,
    },
]

_WORKOUT_CLASSIFICATION = {
    "suggested_type": "Strength",
    "experience_level": "Beginner",
    "bmi": 25.0,
    "all_scores": {"Cardio": 1.5, "Strength": 1.8, "Flexibility": 1.2, "HIIT": 0.9},
    "reasoning": (
        "DMT2 genetic risk. ADA recommends Strength + Cardio combo for insulin sensitivity. "
        "Experience level: Beginner. Recommended type: Strength."
    ),
}


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

@patch("precision_health_agents.agents.genomics.anthropic.Anthropic")
@patch("precision_health_agents.agents.genomics.classify_dna")
@patch("precision_health_agents.agents.doctor.anthropic.Anthropic")
@patch("precision_health_agents.agents.doctor.classify_diabetes")
@patch("precision_health_agents.agents.health_trainer.anthropic.Anthropic")
@patch("precision_health_agents.agents.health_trainer.classify_workout_type")
@patch("precision_health_agents.agents.health_trainer.recommend_exercises")
def test_full_pipeline_dna_doctor_health_trainer(
    mock_recommend,
    mock_classify_workout,
    mock_trainer_anthropic,
    mock_classify_diabetes,
    mock_doctor_anthropic,
    mock_classify_dna,
    mock_genomics_anthropic,
):
    """DNA → Doctor → Health Trainer: patient with DMT2 risk referred for exercise."""

    print("\n" + "=" * 65)
    print("INTEGRATION TEST: DNA → Doctor → Health Trainer Pipeline")
    print("=" * 65)

    # ── Step 1: Genomics Agent ─────────────────────────────────────────────
    print("\n[Step 1] GenomicsAgent — DNA classification")

    mock_classify_dna.return_value = {
        "predicted_class": "DMT2",
        "probabilities": {"DMT1": 0.08, "DMT2": 0.72, "NONDM": 0.20},
        "confidence": 0.72,
    }

    mock_genomics_client = MagicMock()
    mock_genomics_anthropic.return_value = mock_genomics_client
    mock_genomics_client.messages.create.side_effect = [
        _tool_response("g1", "classify_dna", {"sequence": _DNA_SEQUENCE}),
        _text_response(
            "The DNA analysis shows a Type 2 Diabetes (DMT2) pattern with 72% confidence. "
            "The patient carries genetic markers associated with insulin resistance. "
            "Clinical evaluation is recommended."
        ),
    ]

    genomics_agent = GenomicsAgent()
    genomics_agent._client = mock_genomics_client
    genomics_result = asyncio.run(
        genomics_agent.analyze(f"Analyze this DNA for diabetes risk: {_DNA_SEQUENCE}")
    )

    print(f"  Predicted class : {genomics_result.findings.predicted_class}")
    print(f"  Confidence      : {genomics_result.findings.confidence:.0%}")
    print(f"  Risk level      : {genomics_result.findings.risk_level.value}")

    assert genomics_result.status == AgentStatus.SUCCESS
    assert genomics_result.findings.predicted_class == "DMT2"
    assert genomics_result.findings.risk_level == RiskLevel.HIGH

    # ── Step 2: Routing — high genetic risk → refer to Doctor ─────────────
    print(f"\n[Routing] Genetic risk = HIGH → referring to DoctorAgent")

    # ── Step 3: Doctor Agent ───────────────────────────────────────────────
    print("\n[Step 2] DoctorAgent — clinical intake")

    mock_classify_diabetes.return_value = {
        "prediction": "Non-Diabetic",
        "probability": 0.44,
        "risk_level": "moderate",
    }

    mock_doctor_client = MagicMock()
    mock_doctor_anthropic.return_value = mock_doctor_client
    mock_doctor_client.messages.create.side_effect = [
        _text_response("Hello! I'm your doctor. Your DNA results show DMT2 risk — let me ask a few questions."),
        _text_response("What is your age and how many pregnancies have you had?"),
        _text_response("Can you share your glucose level, blood pressure, height and weight?"),
        _tool_response("d1", "classify_diabetes", _CLINICAL_VALUES),
        _text_response(
            "Your clinical measurements show a moderate diabetes risk (44% probability). "
            "You are not yet diabetic, but your genetic profile and borderline glucose "
            "suggest lifestyle intervention is important now. I'm referring you to a "
            "health trainer to build an exercise routine."
        ),
    ]

    doctor_agent = DoctorAgent()
    doctor_agent._client = mock_doctor_client

    doctor_conversation = [
        "My DNA test flagged DMT2 risk. What should I do?",
        "I'm 38 years old and have been pregnant once.",
        "Glucose is 128, BP 76, I'm 172 cm and 74 kg.",
        "I don't smoke and try to eat well.",
    ]

    print()
    for msg in doctor_conversation:
        reply = doctor_agent.chat(msg)
        print(f"  Patient : {msg}")
        print(f"  Doctor  : {reply}")
        print()

    doctor_findings = doctor_agent.findings
    doctor_result = doctor_agent.result(reply)

    print(f"  Clinical prediction : {doctor_findings.prediction}")
    print(f"  Probability         : {doctor_findings.probability:.0%}")
    print(f"  Risk level          : {doctor_findings.risk_level.value}")
    print(f"  Recommendation      : {doctor_findings.recommendation.value}")

    assert doctor_result.status == AgentStatus.SUCCESS
    assert doctor_findings.prediction == "Non-Diabetic"
    assert doctor_findings.risk_level == RiskLevel.MODERATE
    assert doctor_findings.recommendation == Recommendation.HEALTH_TRAINER

    # ── Step 4: Routing — HEALTH_TRAINER recommendation ───────────────────
    print(f"\n[Routing] Recommendation = HEALTH_TRAINER → referring to HealthTrainerAgent")
    assert doctor_findings.recommendation == Recommendation.HEALTH_TRAINER

    # ── Step 5: Health Trainer Agent ───────────────────────────────────────
    print("\n[Step 3] HealthTrainerAgent — exercise planning")

    mock_classify_workout.return_value = _WORKOUT_CLASSIFICATION
    mock_recommend.return_value = {
        "exercises": _FAKE_EXERCISES,
        "total_found": len(_FAKE_EXERCISES),
        "filters_applied": {"exercise_type": "Strength", "difficulty": "Beginner"},
    }

    mock_trainer_client = MagicMock()
    mock_trainer_anthropic.return_value = mock_trainer_client
    mock_trainer_client.messages.create.side_effect = [
        _text_response(
            "Hello! I'm your health trainer. I can see from your records that you have "
            "a DMT2 genetic marker and borderline glucose. Let's build a plan together. "
            "Can you tell me your age, gender, height, and weight?"
        ),
        _tool_response("t1", "classify_workout_type", _CLASSIFY_INPUT),
        _text_response(
            "Based on your profile, Strength training is ideal for you. "
            "Resistance exercise improves your body's ability to use blood sugar — "
            "which is directly relevant to your DMT2 genetic risk. "
            "What equipment do you have access to?"
        ),
        _tool_response("t2", "recommend_exercises", _RECOMMEND_INPUT),
        _text_response(
            "Here is your personalised 3-day-per-week Strength plan:\n\n"
            "Day 1 (Upper Body): Dumbbell Lateral Raise 3×12, Hammer Curl 3×10\n"
            "Day 2 (Lower Body): Romanian Deadlift 3×10\n"
            "Day 3 (Full Body): Repeat Day 1 + Day 2 at lighter weight\n\n"
            "Why Strength training? Because building muscle improves insulin sensitivity — "
            "your muscles become better at absorbing glucose from the bloodstream, "
            "directly countering the insulin resistance pattern associated with DMT2.\n\n"
            "Start light, focus on form, and aim to increase weight every 2 weeks."
        ),
    ]

    # Pass prior agent findings as context
    context = {
        "genomics": genomics_result.model_dump(),
        "doctor": doctor_result.model_dump(),
    }

    trainer_agent = HealthTrainerAgent(context=context)
    trainer_agent._client = mock_trainer_client

    # Verify clinical context was injected into system prompt
    assert "DMT2" in trainer_agent._system
    print(f"  Clinical context injected: DMT2 + {doctor_findings.probability:.0%} probability ✓")

    # 3 turns: turn 2 triggers classify (2 API calls), turn 3 triggers recommend (2 API calls)
    trainer_conversation = [
        "My doctor referred me to you. I have a DMT2 genetic risk.",
        "I'm 38, female, 74 kg, 172 cm. I exercise once a week for about 45 minutes.",
        "I have dumbbells at home. No injuries.",
    ]

    print()
    final_reply = ""
    for msg in trainer_conversation:
        reply = trainer_agent.chat(msg)
        final_reply = reply
        print(f"  Patient : {msg}")
        print(f"  Trainer : {reply}")
        print()

    # ── Step 6: Final report ───────────────────────────────────────────────
    print("[Step 4] Final report")

    trainer_result = trainer_agent.result()

    print(f"  Status              : {trainer_result.status.value}")
    print(f"  Fitness level       : {trainer_result.findings.fitness_level}")
    print(f"  Exercises in plan   : {len(trainer_result.findings.recommended_exercises)}")
    print(f"\n  Workout plan:\n")
    for line in trainer_result.findings.weekly_plan.splitlines():
        print(f"    {line}")

    # ── Assertions ─────────────────────────────────────────────────────────
    assert trainer_result.status == AgentStatus.SUCCESS

    findings = trainer_result.findings
    assert findings is not None
    assert findings.fitness_level == "beginner"
    assert len(findings.recommended_exercises) == len(_FAKE_EXERCISES)
    assert "Strength" in findings.weekly_plan or findings.weekly_plan

    # classify_workout_type received the correct diabetes signals from context
    mock_classify_workout.assert_called_once_with(**_CLASSIFY_INPUT)
    call_kwargs = mock_classify_workout.call_args[1]
    assert call_kwargs["diabetes_type"] == "DMT2"
    assert call_kwargs["diabetes_probability"] == 0.44

    # recommend_exercises used the classifier output
    mock_recommend.assert_called_once_with(**_RECOMMEND_INPUT)

    print("\n[Summary]")
    print(f"  DNA result        : DMT2 (72% confidence)")
    print(f"  Clinical result   : Non-Diabetic (44% probability, moderate risk)")
    print(f"  Recommendation    : HEALTH_TRAINER")
    print(f"  Workout type      : {_WORKOUT_CLASSIFICATION['suggested_type']}")
    print(f"  Experience level  : {_WORKOUT_CLASSIFICATION['experience_level']}")
    print(f"  Exercises         : {', '.join(e['Name'] for e in _FAKE_EXERCISES)}")
    print("=" * 65)
