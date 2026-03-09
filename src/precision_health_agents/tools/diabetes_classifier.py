"""Diabetes risk classifier tool using a pre-trained MLP model.

Predicts whether a patient is at risk for diabetes based on 8 clinical features
from the Pima Indians Diabetes Dataset.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).parents[3]
_DATA_DIR = _PROJECT_ROOT / "data" / "diabetes"
_MODEL_PATH = _DATA_DIR / "models" / "mlp_diabetes.keras"
_SCALER_PATH = _DATA_DIR / "models" / "scaler.npy"

FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]


@lru_cache(maxsize=1)
def _load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(str(_MODEL_PATH))


@lru_cache(maxsize=1)
def _load_scaler() -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, scale) arrays saved during training."""
    params = np.load(str(_SCALER_PATH))
    return params[0], params[1]


def classify_diabetes(
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    diabetes_pedigree_function: float,
    age: float,
) -> dict:
    """Predict diabetes risk from clinical measurements.

    Args:
        pregnancies: Number of times pregnant.
        glucose: Plasma glucose concentration (mg/dL, 2-hour oral glucose test).
        blood_pressure: Diastolic blood pressure (mm Hg).
        skin_thickness: Triceps skin fold thickness (mm).
        insulin: 2-Hour serum insulin (mu U/ml).
        bmi: Body mass index (weight kg / height m²).
        diabetes_pedigree_function: Diabetes pedigree function score.
        age: Age in years.

    Returns:
        dict with keys:
            - prediction: str, "Diabetic" or "Non-Diabetic"
            - probability: float, probability of being diabetic (0–1)
            - risk_level: str, "low" | "moderate" | "high"
    """
    x = np.array(
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ],
        dtype=np.float32,
    ).reshape(1, -1)

    mean, scale = _load_scaler()
    x = (x - mean) / scale

    model = _load_model()
    prob = float(model.predict(x, verbose=0)[0][0])

    if prob < 0.3:
        risk_level = "low"
    elif prob < 0.6:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "prediction": "Diabetic" if prob >= 0.5 else "Non-Diabetic",
        "probability": round(prob, 4),
        "risk_level": risk_level,
    }
