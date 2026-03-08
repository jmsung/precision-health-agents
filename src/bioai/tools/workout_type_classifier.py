"""Workout type classifier — uses clinical rules to recommend exercise type and experience level.

Rules are derived from:
- ADA Standards of Medical Care in Diabetes 2023
- ACSM Guidelines for Exercise Testing and Prescription (11th ed.)
- WHO physical activity guidelines

When the gym members ML model is available it will replace the demographic scoring section
while the clinical override layer remains unchanged.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Experience level
# ---------------------------------------------------------------------------

def _experience_level(
    workout_frequency_per_week: int,
    session_duration_hours: float,
) -> str:
    """Infer experience level from self-reported exercise history.

    Thresholds calibrated to gym members dataset distribution:
      Beginner     : freq 0-2 days/wk  OR  session < 0.75 h
      Intermediate : freq 2-3 days/wk  AND session 0.75-1.5 h
      Expert       : freq >= 4 days/wk AND session > 1.0 h
    """
    freq = workout_frequency_per_week
    dur = session_duration_hours

    if freq == 0 or dur == 0:
        return "Beginner"
    if freq >= 4 and dur > 1.0:
        return "Expert"
    if freq >= 2 and dur >= 0.75:
        return "Intermediate"
    return "Beginner"


# ---------------------------------------------------------------------------
# Workout type scoring
# ---------------------------------------------------------------------------

_BASE_TYPES = ("Cardio", "Strength", "Flexibility", "HIIT")


def _score_types(
    age: int,
    bmi: float,
    experience: str,
    diabetes_type: str,
    diabetes_probability: float,
) -> dict[str, float]:
    scores: dict[str, float] = {t: 1.0 for t in _BASE_TYPES}

    # -- Diabetes clinical rules (ADA 2023) ----------------------------------

    if diabetes_type == "DMT1":
        # Moderate aerobic is safest; HIIT causes unpredictable glucose swings
        scores["Cardio"] += 0.5
        scores["Strength"] += 0.3
        scores["Flexibility"] += 0.2
        scores["HIIT"] -= 0.8  # hypoglycemia risk without close monitoring

    elif diabetes_type == "DMT2" or diabetes_probability >= 0.5:
        # ADA gold standard: resistance + aerobic combo for T2DM
        scores["Strength"] += 0.5   # improves insulin sensitivity via muscle glucose uptake
        scores["Cardio"] += 0.5     # reduces cardiovascular risk (leading T2DM complication)
        scores["HIIT"] += 0.1       # effective but caution at beginner level

    elif diabetes_probability >= 0.35:
        # Pre-diabetic / high-risk: lifestyle intervention prevents progression
        scores["Cardio"] += 0.2
        scores["Strength"] += 0.2
        scores["Flexibility"] += 0.1

    # -- BMI adjustments (joint load + cardiovascular capacity) ---------------

    if bmi >= 35:
        scores["Cardio"] += 0.3     # low-impact aerobic preferred
        scores["Flexibility"] += 0.2
        scores["HIIT"] -= 0.3       # high joint stress, injury risk
    elif bmi >= 30:
        scores["Cardio"] += 0.1
        scores["HIIT"] -= 0.1

    # -- Age adjustments ------------------------------------------------------

    if age >= 65:
        scores["Flexibility"] += 0.4    # preserves range of motion, reduces fall risk
        scores["Strength"] += 0.2       # prevents sarcopenia
        scores["Cardio"] += 0.1         # low-impact aerobic
        scores["HIIT"] -= 0.5           # cardiac stress, recovery capacity
    elif age >= 50:
        scores["Flexibility"] += 0.2
        scores["HIIT"] -= 0.2
    elif age < 30:
        scores["HIIT"] += 0.2           # higher recovery capacity

    # -- Experience adjustments -----------------------------------------------

    if experience == "Beginner":
        scores["HIIT"] -= 0.3           # technique and conditioning must come first
        scores["Cardio"] += 0.1
        scores["Flexibility"] += 0.1
    elif experience == "Expert":
        scores["HIIT"] += 0.3

    return scores


def _build_reasoning(
    top_type: str,
    age: int,
    bmi: float,
    experience: str,
    diabetes_type: str,
    diabetes_probability: float,
) -> str:
    parts: list[str] = []

    if diabetes_type == "DMT1":
        parts.append("DMT1 genetic risk — HIIT avoided due to unpredictable glucose response")
    elif diabetes_type == "DMT2":
        parts.append("DMT2 genetic risk — ADA recommends Strength + Cardio combo for insulin sensitivity")
    elif diabetes_probability >= 0.5:
        parts.append(f"High clinical diabetes probability ({diabetes_probability:.0%}) — resistance and aerobic training prioritized")
    elif diabetes_probability >= 0.35:
        parts.append(f"Elevated diabetes risk ({diabetes_probability:.0%}) — lifestyle exercise recommended for prevention")

    if bmi >= 35:
        parts.append(f"BMI {bmi:.1f} — low-impact exercises preferred to reduce joint stress")
    elif bmi >= 30:
        parts.append(f"BMI {bmi:.1f} — moderate-impact exercises appropriate")

    if age >= 65:
        parts.append(f"Age {age} — flexibility and strength prioritized for mobility and fall prevention")
    elif age >= 50:
        parts.append(f"Age {age} — flexibility included to maintain range of motion")

    parts.append(f"Experience level: {experience}")
    parts.append(f"Recommended type: {top_type}")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def classify_workout_type(
    age: int,
    gender: str,
    weight_kg: float,
    height_cm: float,
    workout_frequency_per_week: int,
    session_duration_hours: float,
    diabetes_type: str = "NONDM",
    diabetes_probability: float = 0.0,
) -> dict:
    """Classify the most suitable workout type and experience level for a patient.

    Uses ADA 2023 clinical guidelines combined with demographic scoring.
    The ML-based demographic model (gym members dataset) will replace the
    demographic scoring layer when available.

    Args:
        age: Patient age in years.
        gender: "Male" or "Female".
        weight_kg: Body weight in kilograms.
        height_cm: Height in centimetres.
        workout_frequency_per_week: Days per week currently exercising (0 if none).
        session_duration_hours: Typical session length in hours (0 if none).
        diabetes_type: Genomics finding — "DMT1", "DMT2", or "NONDM".
        diabetes_probability: Clinical diabetes probability from DoctorAgent (0.0–1.0).

    Returns:
        dict with keys:
          - suggested_type: str  (Cardio | Strength | Flexibility | HIIT)
          - experience_level: str  (Beginner | Intermediate | Expert)
          - all_scores: dict[str, float]
          - reasoning: str
          - bmi: float
    """
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 1)

    experience = _experience_level(workout_frequency_per_week, session_duration_hours)
    scores = _score_types(age, bmi, experience, diabetes_type, diabetes_probability)

    suggested_type = max(scores, key=lambda t: scores[t])
    reasoning = _build_reasoning(
        suggested_type, age, bmi, experience, diabetes_type, diabetes_probability
    )

    return {
        "suggested_type": suggested_type,
        "experience_level": experience,
        "bmi": bmi,
        "all_scores": {t: round(s, 2) for t, s in scores.items()},
        "reasoning": reasoning,
    }
