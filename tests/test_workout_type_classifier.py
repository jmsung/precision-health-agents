"""Tests for workout_type_classifier — clinical rule logic."""

import pytest

from precision_health_agents.tools.workout_type_classifier import classify_workout_type


def _classify(**kwargs) -> dict:
    defaults = dict(
        age=35, gender="Male", weight_kg=75, height_cm=175,
        workout_frequency_per_week=2, session_duration_hours=1.0,
        diabetes_type="NONDM", diabetes_probability=0.0,
    )
    defaults.update(kwargs)
    return classify_workout_type(**defaults)


# ---------------------------------------------------------------------------
# Experience level
# ---------------------------------------------------------------------------

def test_no_exercise_history_is_beginner():
    r = _classify(workout_frequency_per_week=0, session_duration_hours=0)
    assert r["experience_level"] == "Beginner"


def test_light_exercise_is_beginner():
    r = _classify(workout_frequency_per_week=1, session_duration_hours=0.5)
    assert r["experience_level"] == "Beginner"


def test_moderate_exercise_is_intermediate():
    r = _classify(workout_frequency_per_week=3, session_duration_hours=1.0)
    assert r["experience_level"] == "Intermediate"


def test_heavy_exercise_is_expert():
    r = _classify(workout_frequency_per_week=5, session_duration_hours=1.5)
    assert r["experience_level"] == "Expert"


# ---------------------------------------------------------------------------
# Diabetes clinical rules
# ---------------------------------------------------------------------------

def test_dmt1_avoids_hiit():
    r = _classify(diabetes_type="DMT1", diabetes_probability=0.8)
    assert r["all_scores"]["HIIT"] < r["all_scores"]["Cardio"]
    assert r["suggested_type"] != "HIIT"


def test_dmt2_prefers_strength_or_cardio():
    r = _classify(diabetes_type="DMT2", diabetes_probability=0.75)
    assert r["suggested_type"] in ("Strength", "Cardio")
    # Strength and Cardio scores should be the highest two
    scores = r["all_scores"]
    top2 = sorted(scores, key=lambda t: scores[t], reverse=True)[:2]
    assert set(top2) == {"Strength", "Cardio"}


def test_high_clinical_probability_prefers_strength_or_cardio():
    # NONDM genetically but high clinical probability (pre-diabetic)
    r = _classify(diabetes_type="NONDM", diabetes_probability=0.65)
    assert r["suggested_type"] in ("Strength", "Cardio")


def test_low_risk_no_strong_clinical_override():
    r = _classify(diabetes_type="NONDM", diabetes_probability=0.1)
    # All scores should be close to base — no strong penalty
    scores = r["all_scores"]
    assert max(scores.values()) - min(scores.values()) < 1.5


# ---------------------------------------------------------------------------
# BMI rules
# ---------------------------------------------------------------------------

def test_obese_patient_avoids_hiit():
    # BMI ~40
    r = _classify(weight_kg=120, height_cm=173)
    assert r["bmi"] >= 30
    assert r["all_scores"]["HIIT"] < r["all_scores"]["Cardio"]


def test_bmi_is_computed_correctly():
    r = _classify(weight_kg=70, height_cm=175)
    expected = round(70 / (1.75 ** 2), 1)
    assert r["bmi"] == expected


# ---------------------------------------------------------------------------
# Age rules
# ---------------------------------------------------------------------------

def test_elderly_avoids_hiit():
    r = _classify(age=68)
    assert r["all_scores"]["HIIT"] < r["all_scores"]["Flexibility"]
    assert r["suggested_type"] != "HIIT"


def test_young_adult_hiit_not_penalised():
    r = _classify(age=25, workout_frequency_per_week=4, session_duration_hours=1.5)
    # Expert + young → HIIT bonus should apply
    assert r["all_scores"]["HIIT"] > 1.0


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_has_required_keys():
    r = _classify()
    for key in ("suggested_type", "experience_level", "bmi", "all_scores", "reasoning"):
        assert key in r


def test_suggested_type_is_valid():
    r = _classify()
    assert r["suggested_type"] in ("Cardio", "Strength", "Flexibility", "HIIT")


def test_reasoning_is_non_empty():
    r = _classify(diabetes_type="DMT2", diabetes_probability=0.7)
    assert len(r["reasoning"]) > 0
    assert "DMT2" in r["reasoning"] or "diabetes" in r["reasoning"].lower()
