"""Evaluate the health trainer's classify_workout_type tool against the gym members dataset.

Three evaluation layers:
  1. Experience level  — direct comparison (ground truth available)
  2. Workout type (NONDM baseline) — demographic scoring vs actual gym member choices
  3. Synthetic diabetes overlay — clinical constraint verification

Usage:
    uv run python scripts/evaluate_health_trainer.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from precision_health_agents.tools.workout_type_classifier import classify_workout_type, _experience_level

_PROJECT_ROOT = Path(__file__).parents[1]
_GYM_DATA = _PROJECT_ROOT / "data" / "gym_members" / "raw" / "gym_members.csv"

_EXP_MAP = {1: "Beginner", 2: "Intermediate", 3: "Expert"}


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(_GYM_DATA)
    df["Experience_Label"] = df["Experience_Level"].map(_EXP_MAP)
    return df


# ---------------------------------------------------------------------------
# Layer 1: Experience level evaluation
# ---------------------------------------------------------------------------

def evaluate_experience_level(df: pd.DataFrame) -> dict:
    """Compare our experience level thresholds against gym member ground truth."""
    correct = 0
    total = len(df)
    confusion: dict[str, dict[str, int]] = {
        lvl: {"Beginner": 0, "Intermediate": 0, "Expert": 0} for lvl in _EXP_MAP.values()
    }

    for _, row in df.iterrows():
        predicted = _experience_level(
            int(row["Workout_Frequency (days/week)"]),
            float(row["Session_Duration (hours)"]),
        )
        actual = row["Experience_Label"]
        confusion[actual][predicted] += 1
        if predicted == actual:
            correct += 1

    accuracy = correct / total
    return {"accuracy": accuracy, "total": total, "correct": correct, "confusion": confusion}


# ---------------------------------------------------------------------------
# Layer 2: Workout type baseline (no diabetes context)
# ---------------------------------------------------------------------------

def evaluate_workout_type_baseline(df: pd.DataFrame) -> dict:
    """Run classify_workout_type with NONDM/0.0 and compare to actual choices."""
    correct = 0
    total = len(df)
    type_counts: dict[str, dict[str, int]] = {}

    for _, row in df.iterrows():
        result = classify_workout_type(
            age=int(row["Age"]),
            gender=row["Gender"],
            weight_kg=float(row["Weight (kg)"]),
            height_cm=float(row["Height (m)"]) * 100,  # dataset uses metres
            workout_frequency_per_week=int(row["Workout_Frequency (days/week)"]),
            session_duration_hours=float(row["Session_Duration (hours)"]),
            diabetes_type="NONDM",
            diabetes_probability=0.0,
        )
        predicted = result["suggested_type"]
        actual = row["Workout_Type"]

        if actual not in type_counts:
            type_counts[actual] = {}
        type_counts[actual][predicted] = type_counts[actual].get(predicted, 0) + 1

        if predicted == actual:
            correct += 1

    accuracy = correct / total

    # Per-type precision: how often does our suggestion match when we suggest X
    suggested_counts: dict[str, int] = {}
    suggested_correct: dict[str, int] = {}
    for actual, preds in type_counts.items():
        for pred, count in preds.items():
            suggested_counts[pred] = suggested_counts.get(pred, 0) + count
            if pred == actual:
                suggested_correct[pred] = suggested_correct.get(pred, 0) + count

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "confusion": type_counts,
        "suggested_counts": suggested_counts,
        "suggested_correct": suggested_correct,
    }


# ---------------------------------------------------------------------------
# Layer 3: Synthetic diabetes overlay — clinical constraint verification
# ---------------------------------------------------------------------------

def _assign_synthetic_diabetes(row: pd.Series) -> tuple[str, float]:
    """Assign synthetic diabetes labels based on risk profile."""
    bmi = float(row["BMI"])
    age = int(row["Age"])

    if bmi > 35 and age > 50:
        return "DMT1", 0.80
    elif bmi > 30 and age > 45:
        return "DMT2", 0.70
    elif bmi > 30:
        return "DMT2", 0.55
    elif bmi > 28 and age > 40:
        return "NONDM", 0.40  # pre-diabetic
    else:
        return "NONDM", 0.0


def evaluate_clinical_constraints(df: pd.DataFrame) -> dict:
    """Verify clinical rules never produce unsafe recommendations."""
    violations: list[dict] = []
    total_diabetic = 0
    total_dmt1 = 0

    for _, row in df.iterrows():
        diabetes_type, diabetes_prob = _assign_synthetic_diabetes(row)

        if diabetes_type == "NONDM" and diabetes_prob < 0.3:
            continue  # skip low-risk, nothing to verify

        result = classify_workout_type(
            age=int(row["Age"]),
            gender=row["Gender"],
            weight_kg=float(row["Weight (kg)"]),
            height_cm=float(row["Height (m)"]) * 100,
            workout_frequency_per_week=int(row["Workout_Frequency (days/week)"]),
            session_duration_hours=float(row["Session_Duration (hours)"]),
            diabetes_type=diabetes_type,
            diabetes_probability=diabetes_prob,
        )

        scores = result["all_scores"]
        suggested = result["suggested_type"]

        # DMT1 must never suggest HIIT (hypoglycemia risk)
        if diabetes_type == "DMT1":
            total_dmt1 += 1
            if suggested == "HIIT":
                violations.append({
                    "rule": "DMT1_no_HIIT",
                    "row": row.to_dict(),
                    "result": result,
                })

        # DMT2: Strength or Cardio should score highest (ADA guideline)
        if diabetes_type == "DMT2":
            total_diabetic += 1
            top2 = sorted(scores, key=lambda t: scores[t], reverse=True)[:2]
            if "Strength" not in top2 and "Cardio" not in top2:
                violations.append({
                    "rule": "DMT2_strength_or_cardio_in_top2",
                    "row": row.to_dict(),
                    "result": result,
                })

        # High probability (>0.5): HIIT should not be top suggestion
        if diabetes_prob > 0.5 and suggested == "HIIT":
            exp = result["experience_level"]
            if exp == "Beginner":
                violations.append({
                    "rule": "high_risk_beginner_no_HIIT",
                    "row": row.to_dict(),
                    "result": result,
                })

    return {
        "total_dmt1": total_dmt1,
        "total_dmt2": total_diabetic,
        "violations": violations,
        "violation_count": len(violations),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Health Trainer Evaluation — Gym Members Dataset (973 rows)")
    print("=" * 65)

    df = _load_data()

    # ── Layer 1: Experience level ────────────────────────────────────
    print("\n[Layer 1] Experience Level Evaluation")
    print("-" * 50)

    exp_result = evaluate_experience_level(df)
    print(f"  Accuracy: {exp_result['accuracy']:.1%} ({exp_result['correct']}/{exp_result['total']})")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':>15} {'Beginner':>10} {'Intermediate':>13} {'Expert':>8}")
    for actual in ["Beginner", "Intermediate", "Expert"]:
        row = exp_result["confusion"][actual]
        print(f"  {actual:>15} {row['Beginner']:>10} {row['Intermediate']:>13} {row['Expert']:>8}")

    # ── Layer 2: Workout type baseline ───────────────────────────────
    print("\n[Layer 2] Workout Type Baseline (NONDM, no diabetes context)")
    print("-" * 50)

    wt_result = evaluate_workout_type_baseline(df)
    print(f"  Accuracy: {wt_result['accuracy']:.1%} ({wt_result['correct']}/{wt_result['total']})")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    all_types = sorted(set(
        t for preds in wt_result["confusion"].values() for t in preds
    ))
    header = f"  {'':>12}" + "".join(f"{t:>12}" for t in all_types)
    print(header)
    for actual in ["Cardio", "HIIT", "Strength", "Yoga"]:
        preds = wt_result["confusion"].get(actual, {})
        row_str = f"  {actual:>12}" + "".join(f"{preds.get(t, 0):>12}" for t in all_types)
        print(row_str)

    print()
    print("  Note: Low accuracy is EXPECTED — gym members' choices are personal")
    print("  preference, not clinical recommendations. This baseline measures")
    print("  demographic alignment only.")

    # ── Layer 3: Clinical constraints ────────────────────────────────
    print("\n[Layer 3] Clinical Constraint Verification (synthetic diabetes)")
    print("-" * 50)

    clinical_result = evaluate_clinical_constraints(df)
    print(f"  Synthetic DMT1 patients: {clinical_result['total_dmt1']}")
    print(f"  Synthetic DMT2 patients: {clinical_result['total_dmt2']}")
    print(f"  Constraint violations : {clinical_result['violation_count']}")

    if clinical_result["violations"]:
        print("\n  ⚠ VIOLATIONS:")
        for v in clinical_result["violations"][:5]:
            print(f"    Rule: {v['rule']}")
            print(f"    Suggested: {v['result']['suggested_type']}, Scores: {v['result']['all_scores']}")
    else:
        print("  All clinical safety constraints passed ✓")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Experience level accuracy : {exp_result['accuracy']:.1%}")
    print(f"  Workout type baseline     : {wt_result['accuracy']:.1%} (expected low — see note)")
    print(f"  Clinical violations       : {clinical_result['violation_count']}")
    print()

    # Experience level threshold calibration insight
    print("  Experience level thresholds vs gym data (mean freq/dur):")
    for lvl, label in _EXP_MAP.items():
        sub = df[df["Experience_Level"] == lvl]
        print(f"    {label:>13}: freq={sub['Workout_Frequency (days/week)'].mean():.1f} d/wk, "
              f"dur={sub['Session_Duration (hours)'].mean():.2f} h")
    print("=" * 65)


if __name__ == "__main__":
    main()
