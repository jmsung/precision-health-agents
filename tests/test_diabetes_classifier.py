"""Tests for diabetes_classifier tool."""

import pytest


def test_non_diabetic_low_risk():
    from precision_health_agents.tools.diabetes_classifier import classify_diabetes

    result = classify_diabetes(
        pregnancies=1,
        glucose=85,
        blood_pressure=66,
        skin_thickness=29,
        insulin=0,
        bmi=26.6,
        diabetes_pedigree_function=0.351,
        age=31,
    )
    assert result["prediction"] == "Non-Diabetic"
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ("low", "moderate", "high")


def test_diabetic_high_risk():
    from precision_health_agents.tools.diabetes_classifier import classify_diabetes

    result = classify_diabetes(
        pregnancies=8,
        glucose=183,
        blood_pressure=64,
        skin_thickness=0,
        insulin=0,
        bmi=23.3,
        diabetes_pedigree_function=0.672,
        age=32,
    )
    assert result["prediction"] == "Diabetic"
    assert result["probability"] > 0.5


def test_return_shape():
    from precision_health_agents.tools.diabetes_classifier import classify_diabetes

    result = classify_diabetes(6, 148, 72, 35, 0, 33.6, 0.627, 50)
    assert set(result.keys()) == {"prediction", "probability", "risk_level"}


def test_probability_range():
    from precision_health_agents.tools.diabetes_classifier import classify_diabetes

    result = classify_diabetes(0, 100, 70, 20, 50, 25.0, 0.3, 25)
    assert 0.0 <= result["probability"] <= 1.0


def test_risk_level_thresholds():
    """risk_level boundaries: <0.3 low, 0.3-0.6 moderate, >=0.6 high."""
    from precision_health_agents.tools.diabetes_classifier import classify_diabetes

    # Clearly healthy patient → low risk
    low = classify_diabetes(0, 80, 70, 15, 30, 20.0, 0.2, 22)
    # Clearly diabetic profile → high risk
    high = classify_diabetes(10, 200, 90, 40, 300, 40.0, 1.5, 55)

    assert low["risk_level"] == "low"
    assert high["risk_level"] == "high"
