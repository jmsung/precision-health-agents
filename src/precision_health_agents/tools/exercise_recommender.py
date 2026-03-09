"""Exercise recommendation tool — searches the exercises dataset by criteria."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).parents[3]
_DATA_PATH = _PROJECT_ROOT / "data" / "exercises" / "raw" / "exercises.csv"


@lru_cache(maxsize=1)
def _load_exercises() -> pd.DataFrame:
    return pd.read_csv(_DATA_PATH)


def recommend_exercises(
    body_part: str | None = None,
    exercise_type: str | None = None,
    difficulty: str | None = None,
    equipment: str | None = None,
    max_results: int = 10,
) -> dict:
    """Search and filter exercises from the dataset.

    Args:
        body_part: Target area (e.g. "Chest", "Back", "Legs", "Core", "Full Body").
        exercise_type: "Strength", "Cardio", "Flexibility", or "Plyometric".
        difficulty: "Beginner", "Intermediate", or "Expert".
        equipment: "Bodyweight", "Dumbbell", "Barbell", or "Machine".
        max_results: Maximum number of exercises to return (default 10).

    Returns:
        dict with keys:
          - exercises: list of matching exercise dicts
          - total_found: int
          - filters_applied: dict of non-None filters used
    """
    df = _load_exercises().copy()

    filters: dict[str, str] = {}

    if body_part:
        mask = df["BodyPart"].str.lower() == body_part.lower()
        # also include Full Body exercises when a specific part is requested
        full_body_mask = df["BodyPart"].str.lower() == "full body"
        df = df[mask | full_body_mask]
        filters["body_part"] = body_part

    if exercise_type:
        df = df[df["Type"].str.lower() == exercise_type.lower()]
        filters["exercise_type"] = exercise_type

    if difficulty:
        df = df[df["Level"].str.lower() == difficulty.lower()]
        filters["difficulty"] = difficulty

    if equipment:
        df = df[df["Equipment"].str.lower() == equipment.lower()]
        filters["equipment"] = equipment

    total_found = len(df)
    df = df.head(max_results)

    exercises = df.to_dict(orient="records")
    return {
        "exercises": exercises,
        "total_found": total_found,
        "filters_applied": filters,
    }
