# Health Trainer Agent

Conversational agent that creates personalized exercise plans for patients referred by the doctor agent (`Recommendation.HEALTH_TRAINER`).

## Role in the Pipeline

```
Patient → GenomicsAgent → DoctorAgent → (low/moderate risk) → HealthTrainerAgent
```

When the doctor determines a patient does not need medication, they are referred to the health trainer for a lifestyle-based intervention.

## Files

| File | Purpose |
|---|---|
| `src/bioai/agents/health_trainer.py` | Agent class |
| `src/bioai/tools/workout_type_classifier.py` | Workout type + experience level classifier |
| `src/bioai/tools/exercise_recommender.py` | Exercise lookup/filter tool |
| `src/bioai/prompts/health_trainer.txt` | System prompt with 7-step conversation flow |
| `data/exercises/raw/exercises.csv` | 50-exercise dataset |
| `tests/test_workout_type_classifier.py` | Classifier tests (15 tests) |
| `tests/test_health_trainer_agent.py` | Agent tests (10 tests, mocked API) |
| `tests/test_exercise_recommender.py` | Recommender tests (9 tests) |

---

## Data Flow

```
GenomicsFindings ─┐
                  ├─→ classify_workout_type(age, gender, weight, height,
DoctorFindings ───┘                         diabetes_type, risk_prob)
                                            │
                      clinical rules ────── ┤─ diabetes modifier  (ADA 2023)
                      [gym ML model] ─────── ┘─ demographic score  (placeholder → ML when CSV available)
                                            │
                                            ▼
                                  ranked_workout_types + experience_level
                                            │
                                            ▼
                             recommend_exercises(type, difficulty)
                                            │
                                            ▼
                                  weekly exercise plan
```

---

## Conversation Flow (7 steps)

```
Step 1  Introduce as health trainer, explain the plan.

Step 2  Ask: age, gender, height, weight.

Step 3  Ask: exercise frequency (days/week) and session duration (hours).
        → 0 frequency / 0 duration is fine — records as no prior exercise.

Step 4  Call classify_workout_type()
        → uses vitals + exercise history + diabetes findings from context
        → returns suggested_type (Cardio/Strength/Flexibility/HIIT)
                   experience_level (Beginner/Intermediate/Expert)
                   reasoning (explained to patient in plain language)

Step 5  Ask: available equipment, body parts to focus on or avoid, any injuries.

Step 6  Call recommend_exercises(exercise_type, difficulty, equipment?, body_part?)
        → may be called multiple times for different body parts.

Step 7  Deliver weekly plan with clinical reasoning explained in plain language.
```

---

## Tool: `classify_workout_type`

```python
classify_workout_type(
    age: int,
    gender: str,                      # "Male" / "Female"
    weight_kg: float,
    height_cm: float,
    # from exercise history (asked by trainer)
    workout_frequency_per_week: int,
    session_duration_hours: float,
    # from prior agents (passed as context, injected into system prompt)
    diabetes_type: str,               # "DMT1" / "DMT2" / "NONDM"
    diabetes_probability: float,      # 0.0–1.0 from DoctorAgent
) -> {
    "suggested_type": "Strength",     # highest-scoring workout type
    "experience_level": "Beginner",   # from exercise history
    "bmi": 24.5,
    "reasoning": "...",               # plain-language explanation
    "all_scores": {                   # full ranking
        "Cardio": 1.6,
        "Strength": 1.8,
        "Flexibility": 1.2,
        "HIIT": 0.8,
    },
}
```

### Clinical rules (ADA 2023 + ACSM)

| Condition | Effect on scores |
|---|---|
| DMT1 (genetic) | Cardio +0.5, Strength +0.3, **HIIT −0.8** (hypoglycemia risk) |
| DMT2 (genetic or clinical prob ≥ 0.5) | **Strength +0.5, Cardio +0.5** (ADA gold standard), HIIT +0.1 |
| Pre-diabetic (prob 0.35–0.5) | Cardio +0.2, Strength +0.2, Flexibility +0.1 |
| BMI ≥ 35 | Cardio +0.3, Flexibility +0.2, **HIIT −0.3** (joint stress) |
| BMI ≥ 30 | Cardio +0.1, HIIT −0.1 |
| Age ≥ 65 | **Flexibility +0.4**, Strength +0.2, **HIIT −0.5** (cardiac stress) |
| Age ≥ 50 | Flexibility +0.2, HIIT −0.2 |
| Age < 30 | HIIT +0.2 (recovery capacity) |
| Beginner experience | HIIT −0.3, Cardio +0.1 |
| Expert experience | HIIT +0.3 |

### Experience level thresholds

| Frequency (days/wk) | Session (hours) | Level |
|---|---|---|
| 0 or 0h session | — | Beginner |
| ≤ 1 or < 0.75h | — | Beginner |
| 2–3 and ≥ 0.75h | — | Intermediate |
| ≥ 4 and > 1.0h | — | Expert |

> **Note:** The demographic scoring layer (base scores from gym member population data) is currently a flat 1.0 baseline. It will be replaced by an ML model trained on the [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) when the CSV is available, without changing the clinical override layer above.

---

## Tool: `recommend_exercises`

```python
recommend_exercises(
    body_part="Legs",       # optional
    exercise_type="Cardio", # optional — use suggested_type from classify_workout_type
    difficulty="Beginner",  # optional — use experience_level from classify_workout_type
    equipment="Bodyweight", # optional
    max_results=10,         # optional
) -> {
    "exercises": [...],     # list of exercise dicts
    "total_found": int,
    "filters_applied": {...},
}
```

Reads `data/exercises/raw/exercises.csv` (cached after first load). Full Body exercises are always included when filtering by a specific body part.

---

## Clinical Context Injection

Prior agent findings are injected into the system prompt at agent construction time:

```python
trainer = HealthTrainerAgent(
    context={
        "genomics": genomics_result.model_dump(),
        "doctor":   doctor_result.model_dump(),
    }
)
```

Claude reads this silently (does not repeat it to the patient verbatim) and uses `diabetes_type` + `diabetes_probability` when calling `classify_workout_type`. If no context is provided, defaults to `NONDM` / `0.0`.

---

## Output Model: `HealthTrainerFindings`

```python
HealthTrainerFindings(
    fitness_level="beginner",           # from experience_level
    goals=["weight loss", "cardio"],    # extracted from conversation
    recommended_exercises=[...],        # all exercises from recommend_exercises calls
    weekly_plan="...",                  # trainer's final narrative plan
)
```

Wrapped in the standard `AgentResult` envelope. `findings` is `None` until **both** `classify_workout_type` and `recommend_exercises` have been called.

---

## Usage

```python
from bioai.agents.health_trainer import HealthTrainerAgent

# Standalone (no prior agents)
trainer = HealthTrainerAgent()

# With prior agent context (recommended)
trainer = HealthTrainerAgent(context={
    "genomics": genomics_result.model_dump(),
    "doctor":   doctor_result.model_dump(),
})

reply = trainer.chat("My doctor referred me — I need to start exercising.")
reply = trainer.chat("I'm 45, female, 68kg, 162cm.")
reply = trainer.chat("I exercise once a week for about 30 minutes.")
# → Claude calls classify_workout_type internally here
reply = trainer.chat("I have dumbbells at home and want to focus on my legs.")
# → Claude calls recommend_exercises internally here
# → Delivers weekly plan

result = trainer.result()
print(result.findings.fitness_level)          # "beginner"
print(result.findings.recommended_exercises)  # list of exercise dicts
```

---

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| 50 exercises | [Kaggle: Best 50 Exercises](https://www.kaggle.com/datasets/prajwaldongre/best-50-exercise-for-your-body) | `recommend_exercises` lookup |
| Gym members *(pending)* | [Kaggle: Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) | ML demographic scoring in `classify_workout_type` |
