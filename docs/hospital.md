# Hospital Agent — Implementation

## Overview

The Hospital Agent is the **coordinator of the molecular confirmation step**. When DNA + clinical assessment both suggest diabetes, the Hospital Agent manages the patient interaction — explaining the need for blood tests, obtaining consent, running transcriptomics and metabolomics analyses simultaneously, and making the combined confirmation decision.

**Key role**: Patient-facing coordinator that bridges the clinical (Doctor) and molecular (Transcriptomics + Metabolomics) layers. Ensures patients understand why molecular tests are needed and provides a single, clear recommendation.

## Pipeline

```
Patient referred to hospital (DNA=DMT2 + Clinical=Diabetic)
        |
  HospitalAgent.chat("I've been referred. What happens?")
        |
  Claude explains: blood tests needed for molecular confirmation
        |
  Patient: "Yes, I'm willing to do the tests."
        |
  Claude calls run_hospital_tests(consent=True, gene_expression, metabolite_levels)
        |
  +-----+-----+  (runs both analyses)
  |           |
  analyze_gene_expression     analyze_metabolic_profile
  (transcriptomics)           (metabolomics)
  |           |
  +-----+-----+
        |
  Combined decision:
    Both confirm    → confirmed, confidence="high"
    One confirms    → confirmed, confidence="moderate"
    Neither confirms → not confirmed (false positive)
        |
  Claude explains results to patient
        |
  HospitalFindings → Pharmacology (confirmed) or HealthTrainer (false positive)
```

## Tool: `run_hospital_tests()`

**Input**:
- `consent: bool` — whether the patient agreed to blood tests
- `gene_expression: dict[str, float]` — gene expression values
- `metabolite_levels: dict[str, float]` — metabolite concentration values

**Output**:
```python
{
    "patient_consented": True,
    "diabetes_confirmed": True,
    "confidence": "high",
    "recommendation": "pharmacology",
    "reasoning": "Both transcriptomics and metabolomics confirm ...",
    "transcriptomics": {
        "confirmed": True,
        "subtype": {"subtype": "inflammation_dominant", "confidence": "moderate"},
        "active_pathways": ["inflammation_immune", "insulin_resistance"],
        "risk_level": "high",
        "complication_risks": [{"complication": "cardiovascular", ...}],
    },
    "metabolomics": {
        "confirmed": True,
        "pattern": "bcaa_elevation",
        "insulin_resistance_score": 0.82,
        "risk_level": "high",
        "subtype_refinement": {"subtype": "metabolic_insulin_resistant", ...},
    },
}
```

## Combined Decision Logic

| Transcriptomics | Metabolomics | Decision | Confidence |
|----------------|-------------|----------|------------|
| Confirmed | Confirmed | → Pharmacology | high |
| Confirmed | Not confirmed | → Pharmacology | moderate |
| Not confirmed | Confirmed | → Pharmacology | moderate |
| Not confirmed | Not confirmed | → Health Trainer | high (false positive) |
| Patient declined tests | — | → Health Trainer | low |

The design is intentionally permissive — either molecular layer confirming is sufficient for medication, since each captures different aspects (pathway activity vs metabolic state). Both negative gives high confidence in false positive.

## Conversational Flow

The Hospital Agent is designed to be conversational, like the Doctor Agent:

1. **Turn 1**: Patient arrives, agent explains the situation and asks for consent
2. **Turn 2**: Patient responds (yes/no), agent runs tests or acknowledges refusal
3. **Turn 3+**: Agent explains results and next steps

## Data Model

```python
class HospitalRecommendation(str, Enum):
    PHARMACOLOGY = "pharmacology"
    HEALTH_TRAINER = "health_trainer"

class HospitalFindings(BaseModel):
    patient_consented: bool
    transcriptomics_confirmed: bool
    metabolomics_confirmed: bool
    diabetes_confirmed: bool
    confidence: str  # "high" | "moderate" | "low"
    recommendation: HospitalRecommendation
    transcriptomics_summary: dict[str, Any]
    metabolomics_summary: dict[str, Any]
    reasoning: str
```

## Inter-Agent Communication

**Input context** (from prior agents):
- `context["genomics"]` — predicted_class, confidence
- `context["doctor"]` — prediction, probability

**Output flows to**:
- If `recommendation == "pharmacology"` → PharmacologyAgent
- If `recommendation == "health_trainer"` → HealthTrainerAgent

## Tests

- `tests/test_hospital_agent.py` — 14 tests
  - Tool tests (7): consent/no-consent, both confirm, one confirms, neither confirms, summaries
  - Agent tests (7): explains tests, runs after consent, declined consent, false positive, result, error, context injection
- `tests/test_integration_pipeline.py` — test 6 + test 7
  - Test 6: Full 5-agent pipeline (DNA → Doctor → Hospital → Pharmacology)
  - Test 7: Hospital false positive (DNA → Doctor → Hospital → HealthTrainer)

## Files

| File | Purpose |
|------|---------|
| `src/precision_health_agents/agents/hospital.py` | Agent + tool: conversational coordinator + `run_hospital_tests()` |
| `src/precision_health_agents/prompts/hospital.txt` | System prompt (kind, reassuring, explains tests) |
| `src/precision_health_agents/models.py` | HospitalFindings, HospitalRecommendation |
