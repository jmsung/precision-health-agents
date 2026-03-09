# Diabetes Dataset

## Overview

Pima Indians Diabetes Dataset — binary classification task predicting whether a patient has diabetes based on clinical measurements.

- **Source:** [Kaggle — mathchi/diabetes-data-set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Original source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **License:** CC0 1.0 (Public Domain)

## Dataset

| Property | Value |
|---|---|
| File | `raw/diabetes.csv` |
| Rows | 768 patients |
| Features | 8 clinical measurements |
| Target | `Outcome` (0 = Non-Diabetic, 1 = Diabetic) |
| Class balance | 500 Non-Diabetic / 268 Diabetic |

### Features

| Column | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of times pregnant | count |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skin fold thickness | mm |
| `Insulin` | 2-hour serum insulin | mu U/ml |
| `BMI` | Body mass index | kg/m² |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (genetic influence score) | — |
| `Age` | Age | years |

## Model

A simple 2-layer MLP trained with TensorFlow/Keras.

| Property | Value |
|---|---|
| File | `models/mlp_diabetes.keras` |
| Scaler | `models/scaler.npy` (mean + scale for StandardScaler) |
| Architecture | Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.2) → Dense(1, sigmoid) |
| Parameters | 2,689 |
| Training script | `scripts/train_diabetes_model.py` |
| Epochs trained | 71 (early stopping, patience=10) |
| Test accuracy | 75% |
| Test loss | 0.516 |

### Classification Report (test set, 154 samples)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-Diabetic | 0.79 | 0.84 | 0.81 |
| Diabetic | 0.66 | 0.57 | 0.61 |

## Usage

```python
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
# {'prediction': 'Non-Diabetic', 'probability': 0.12, 'risk_level': 'low'}
```

Risk levels: `< 0.3` → low, `0.3–0.6` → moderate, `≥ 0.6` → high.
