"""Tests for the exercise_recommender tool."""

import pytest

from precision_health_agents.tools.exercise_recommender import recommend_exercises


def test_returns_all_exercises_with_no_filters():
    result = recommend_exercises()
    assert result["total_found"] > 0
    assert len(result["exercises"]) > 0
    assert result["filters_applied"] == {}


def test_filter_by_body_part():
    result = recommend_exercises(body_part="Chest")
    for ex in result["exercises"]:
        assert ex["BodyPart"] in ("Chest", "Full Body")


def test_filter_by_difficulty():
    result = recommend_exercises(difficulty="Beginner")
    for ex in result["exercises"]:
        assert ex["Level"] == "Beginner"


def test_filter_by_exercise_type():
    result = recommend_exercises(exercise_type="Cardio")
    for ex in result["exercises"]:
        assert ex["Type"] == "Cardio"


def test_filter_by_equipment_bodyweight():
    result = recommend_exercises(equipment="Bodyweight")
    for ex in result["exercises"]:
        assert ex["Equipment"] == "Bodyweight"


def test_max_results_respected():
    result = recommend_exercises(max_results=3)
    assert len(result["exercises"]) <= 3


def test_combined_filters():
    result = recommend_exercises(body_part="Legs", difficulty="Beginner")
    assert result["filters_applied"] == {"body_part": "Legs", "difficulty": "Beginner"}
    for ex in result["exercises"]:
        assert ex["BodyPart"] in ("Legs", "Full Body")
        assert ex["Level"] == "Beginner"


def test_no_results_for_impossible_filter():
    result = recommend_exercises(body_part="Chest", exercise_type="Flexibility")
    # No chest-specific flexibility exercises in our dataset (only Full Body ones possible)
    for ex in result["exercises"]:
        assert ex["Type"] == "Flexibility"


def test_exercise_has_required_fields():
    result = recommend_exercises(max_results=1)
    ex = result["exercises"][0]
    for field in ("Name", "Type", "BodyPart", "Equipment", "Level", "Description", "Benefits"):
        assert field in ex, f"Missing field: {field}"
