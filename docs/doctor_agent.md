# Doctor Agent

## Purpose

The Doctor Agent conducts a natural-language intake interview with a patient to gather 8 clinical measurements, then calls the `classify_diabetes` tool to predict diabetes risk, and finally recommends whether the patient should go to **hospital** (medicine) or a **health trainer** (lifestyle intervention).

When combined with the Genomics Agent's DNA classification, the recommendation becomes DNA-informed — catching cases that pure clinical assessment would miss or misclassify.

## Files

| File | Description |
|------|-------------|
| `src/bioai/agents/doctor.py` | Agent implementation |
| `src/bioai/tools/diabetes_classifier.py` | `classify_diabetes()` tool |
| `src/bioai/prompts/doctor.txt` | System prompt |
| `data/diabetes/models/mlp_diabetes.keras` | Pre-trained MLP |
| `data/diabetes/models/scaler.npy` | StandardScaler params |
| `tests/test_doctor_agent.py` | 5 tests |

## How It Works

### 1. Conversational Intake

The agent does **not** receive a structured form. It asks questions naturally, one or two at a time, and extracts values from whatever the patient says:

```
Patient: "I'm 42, female. My sugar was around 160 last week."
Doctor:  "Got it. Could you share your height and weight?"
Patient: "165 cm, 85 kg. I've been pregnant twice."
Doctor:  "Is there any family history of diabetes?"
Patient: "My mother has it. I don't know my insulin level."
→ All 8 values collected → tool called
```

### 2. The 8 Features

| Feature | How it's gathered |
|---|---|
| `pregnancies` | Directly asked; 0 for males |
| `glucose` | Recent blood test result |
| `blood_pressure` | Diastolic BP; estimated if unknown |
| `skin_thickness` | Rarely known; defaulted to 20 if not mentioned |
| `insulin` | Recent test; defaulted to 0 if unknown |
| `bmi` | Calculated from height/weight if given in that form |
| `diabetes_pedigree_function` | Estimated from family history description |
| `age` | Directly stated |

### 3. Tool Call

Once Claude has collected all 8 values, it calls `classify_diabetes` internally. The patient never sees the raw numbers or probabilities.

### 4. Recommendation

| Condition | Recommendation |
|---|---|
| `risk_level == "high"` or `prediction == "Diabetic"` | `hospital` |
| `risk_level == "moderate"` | `health_trainer` |
| `risk_level == "low"` | `health_trainer` |

## API

```python
from bioai.agents.doctor import DoctorAgent

agent = DoctorAgent()

# Multi-turn conversation
reply = agent.chat("Hi, I'd like a diabetes checkup.")
reply = agent.chat("I'm 42 years old, female.")
reply = agent.chat("My glucose was 160, blood pressure 80.")
reply = agent.chat("165 cm, 85 kg, pregnant twice.")
reply = agent.chat("My mother has diabetes, insulin unknown.")
# → tool fires on the turn Claude has enough data

# Structured findings
findings = agent.findings
# DoctorFindings(
#   prediction="Diabetic",
#   probability=0.74,
#   risk_level=RiskLevel.HIGH,
#   recommendation=Recommendation.HOSPITAL,
#   reasoning="..."
# )

# AgentResult for orchestrator
result = agent.result(summary=reply)
```

## Output Models

```python
class DoctorFindings(BaseModel):
    prediction: Literal["Diabetic", "Non-Diabetic"]
    probability: float          # 0.0–1.0
    risk_level: RiskLevel       # low / moderate / high
    recommendation: Recommendation  # hospital / health_trainer
    reasoning: str

class Recommendation(str, Enum):
    HOSPITAL = "hospital"
    HEALTH_TRAINER = "health_trainer"
```

## Integration with Genomics Agent

The Doctor Agent's recommendation should always be read alongside the Genomics Agent's DNA classification:

| Genomics (DNA) | Doctor (Clinical) | Final Decision |
|---|---|---|
| DMT1 or DMT2 | Diabetic | **Hospital** — confirmed diabetes |
| DMT1 or DMT2 | Non-Diabetic | **Hospital** — genetic override, early intervention |
| NONDM | Diabetic | **Reconsider** — may not need medication, try lifestyle first |
| NONDM | Non-Diabetic | **Health Trainer** — prevention |

The DNA result can override the clinical result in both directions — saving unnecessary treatment for NONDM patients and catching DMT2 patients before symptoms appear.

## Underlying Model

A 2-layer MLP trained on the Pima Indians Diabetes Dataset:

```
Input (8 features, StandardScaler normalized)
  → Dense(64, relu) → Dropout(0.3)
  → Dense(32, relu) → Dropout(0.2)
  → Dense(1, sigmoid)
```

- **Test accuracy**: 75%
- **Trained**: `scripts/train_diabetes_model.py`
- **Dataset**: `data/diabetes/` — see `data/diabetes/README.md`

## Tests

```bash
uv run pytest tests/test_doctor_agent.py -v -s
```

5 tests covering:
- Single turn reply
- Tool invocation and findings population
- Low risk → health_trainer routing
- `AgentResult` construction
- Full 5-turn intake conversation gathering all 8 features
