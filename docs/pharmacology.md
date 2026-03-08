# Pharmacology Agent — Implementation

## Overview

The Pharmacology Agent is the **4th and final agent** in the diabetes pipeline. After 3-layer validation (DNA + clinical + transcriptomics) confirms diabetes, it recommends personalized medications based on the patient's molecular subtype and complication risks. The agent is kind, supportive, and informative — explaining each medication choice in plain language.

## Pipeline

```
TranscriptomicsFindings (confirmed diabetes)
        |
  PharmacologyAgent.chat()
        |
  Claude (tool-use loop, supportive tone)
        |  calls
  recommend_medications(subtype, complications)
        |
  Score 16 ADA guideline drugs:
    +3 subtype match
    +2 ADA first-line
    +2 complication benefit (per match)
    +1.5 severe complication boost
    Contraindication hard exclusion
        |
  Sort by score, return top 8
        |
  Claude composes personalized medication plan:
    Primary medications (score >= 3.0)
    Supportive medications (score < 3.0)
    Monitoring schedule
    Lifestyle integration notes
    Reassurance & encouragement
        |
  PharmacologyFindings
```

## Medication Database

**16 medications** across **8 drug classes**, curated from ADA Standards of Care 2024:

| Class | Drugs | Primary Use |
|-------|-------|-------------|
| Biguanide | Metformin | First-line for most T2DM (insulin resistance) |
| SGLT2 Inhibitor | Empagliflozin, Dapagliflozin, Canagliflozin | Cardio/renal protection |
| GLP-1 Receptor Agonist | Liraglutide, Semaglutide, Dulaglutide | Anti-inflammatory + weight loss |
| Basal Insulin | Insulin Glargine | Beta cell failure |
| Rapid-Acting Insulin | Insulin Aspart | Mealtime coverage |
| Thiazolidinedione | Pioglitazone | Insulin resistance (avoid in HF) |
| DPP-4 Inhibitor | Linagliptin, Sitagliptin | Well-tolerated bridge therapy |
| Statin | Atorvastatin | CV risk reduction |
| ACE Inhibitor | Ramipril | Renal + cardiac protection |
| Neuropathic Pain | Pregabalin, Duloxetine | Diabetic peripheral neuropathy |

Data: `data/medications/raw/diabetes_medications.csv`

## Tool: `recommend_medications()`

**Input**:
- `diabetes_subtype`: from transcriptomics (inflammation_dominant, beta_cell_failure, metabolic_insulin_resistant, fibrotic_complication, mixed)
- `complication_risks`: list of {complication, severity} dicts
- `active_pathways`: optional pathway names
- `max_results`: limit (default 8)

**Scoring**:
| Factor | Points | Condition |
|--------|--------|-----------|
| Subtype match | +3.0 | Drug's `primary_subtype` matches patient's |
| Mixed applicability | +1.0 | Drug marked for `mixed` subtype |
| ADA first-line | +2.0 | Metformin |
| Complication benefit | +2.0 each | Drug's `recommended_complications` overlap patient's |
| Severe complication | +1.5 each | Complication severity is "high" |
| Contraindication | **excluded** | Drug's `contraindicated_complications` overlap patient's |

**Output**:
```python
{
    "medications": [
        {
            "name": "Liraglutide",
            "class": "GLP-1 Receptor Agonist",
            "mechanism": "Incretin mimetic; enhances insulin secretion...",
            "route": "injection",
            "monitoring": "Renal function, thyroid",
            "common_side_effects": "Nausea, vomiting...",
            "notes": "Weight loss benefit; cardiovascular protection",
            "score": 7.0,
            "reasons": ["Primary match for inflammation_dominant", "Addresses: cardiovascular"],
        },
        ...
    ],
    "subtype": "inflammation_dominant",
    "total_matched": 10,
    "complications_considered": ["cardiovascular"],
}
```

## Subtype-to-Medication Strategy

| Subtype | Primary Drugs | Rationale |
|---------|--------------|-----------|
| `inflammation_dominant` | GLP-1 RAs (Liraglutide, Semaglutide, Dulaglutide) | Anti-inflammatory properties + glucose lowering |
| `beta_cell_failure` | Insulin Glargine + Insulin Aspart | Essential when beta cells can't produce enough insulin |
| `metabolic_insulin_resistant` | Metformin + Empagliflozin/Pioglitazone | Target insulin resistance pathways |
| `fibrotic_complication` | Dapagliflozin/Canagliflozin + Ramipril | Organ protection (kidney, heart) |
| `mixed` | Linagliptin/Sitagliptin + combination | Bridge therapy across mechanisms |

## Complication-Aware Selection

| Complication | Beneficial Drugs | Contraindicated |
|-------------|-----------------|-----------------|
| Cardiovascular | SGLT2i, GLP-1 RA, Statin, ACEi | Pioglitazone (if heart failure) |
| Diabetic kidney disease | SGLT2i, ACEi | SGLT2i (if eGFR < 20) |
| Beta cell exhaustion | GLP-1 RA, DPP-4i | — |
| Neuropathy | Pregabalin, Duloxetine | — |

## Data Model

```python
class PharmacologyFindings(BaseModel):
    diabetes_subtype: str                    # from transcriptomics
    primary_medications: list[dict[str, Any]] # score >= 3.0
    supportive_medications: list[dict[str, Any]] # score < 3.0
    monitoring_plan: str                     # aggregated monitoring requirements
    medication_summary: str                  # Claude's narrative plan
```

## Inter-Agent Communication

**Input context** (from prior agents):
- `context["genomics"]` — predicted_class, confidence
- `context["doctor"]` — prediction, probability
- `context["transcriptomics"]` — diabetes_subtype, complication_risks, pathway_scores, active_pathways

All injected into the system prompt so Claude tailors the medication explanation.

## Agent Personality

The PharmacologyAgent is designed to be:
- **Kind**: Acknowledges anxiety about starting medications
- **Supportive**: Reminds patients their care team is there to help
- **Informative**: Explains *why* each drug was chosen for *this* patient
- **Honest**: Doesn't minimize side effects, frames them constructively
- **Empowering**: Medication plans are starting points, adjusted over time

## Tests

- `tests/test_drug_recommender.py` — 13 tests
  - Database loading, subtype matching, contraindication filtering
  - Complication-aware scoring, severity boosting, result structure
- `tests/test_pharmacology_agent.py` — 10 tests
  - Context building (no context, transcriptomics, full context)
  - Agent flow (tool call + text, findings population, result status)
- `tests/test_integration_pipeline.py` — test 2 + test 5
  - Full 4-agent pipeline: DNA -> Doctor -> Transcriptomics -> Pharmacology

## Files

| File | Purpose |
|------|---------|
| `src/bioai/tools/drug_recommender.py` | Tool: ADA guideline medication scoring |
| `src/bioai/agents/pharmacology.py` | Agent: Claude tool-use loop |
| `src/bioai/prompts/pharmacology.txt` | System prompt (kind, supportive, informative) |
| `src/bioai/models.py` | PharmacologyFindings data model |
| `data/medications/raw/diabetes_medications.csv` | 16-drug ADA guideline database |
| `data/medications/README.md` | Dataset documentation |
