# Exercises Dataset

Source: [Best 50 Exercises for Your Body](https://www.kaggle.com/datasets/prajwaldongre/best-50-exercise-for-your-body) — Kaggle

## Schema

| Column | Type | Description |
|---|---|---|
| Name | string | Exercise name |
| Type | string | Strength / Cardio / Flexibility / Plyometric |
| BodyPart | string | Chest / Back / Shoulders / Arms / Core / Legs / Full Body |
| Equipment | string | Bodyweight / Dumbbell / Barbell / Machine |
| Level | string | Beginner / Intermediate / Expert |
| Description | string | Brief instruction for performing the exercise |
| Benefits | string | Key health and fitness benefits |
| CaloriesPerMinute | int | Approximate calories burned per minute |

## Coverage

50 exercises across 8 body parts and 4 exercise types:

| Type | Count |
|---|---|
| Strength | 31 |
| Cardio | 11 |
| Flexibility | 5 |
| Plyometric | 1 |
| Other | 2 |

## Used by

`src/precision_health_agents/tools/exercise_recommender.py` — `recommend_exercises()` function
