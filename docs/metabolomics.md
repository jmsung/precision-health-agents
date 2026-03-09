# Metabolomics Agent — Implementation

## Overview

The Metabolomics Agent analyzes a patient's metabolic profile for diabetes-related metabolic state assessment. It operates as part of the Hospital pathway, running **in parallel with Transcriptomics** to provide independent molecular confirmation. Metabolomics captures the **most dynamic layer** of biology — reflecting the patient's current metabolic state rather than inherited risk or gene activity.

## Pipeline

```
Metabolite concentration data (dict of metabolite -> value)
        |
  MetabolomicsAgent.analyze()
        |
  Claude (tool-use loop)
        |  calls
  analyze_metabolic_profile(metabolite_levels)
        |
  Z-score each metabolite against ST001906 reference
        |
  Compute 5 pathway scores (mean z-score per panel)
        |
  Calculate insulin resistance score (0.0-1.0 sigmoid)
        |
  Classify metabolic pattern
        |
  Confirm or reject diabetes
        |
  Refine diabetes subtype from metabolic signature
        |
  MetabolomicsFindings
```

## Data Source

**ST001906** from Metabolomics Workbench — fasting blood plasma metabolomics:
- 30 healthy controls
- 31 Type 2 diabetes patients
- 78 named metabolites (GC-MS/MS)

Platform: Shimadzu TQ8040 (GC-MS/MS)

## Tool: `analyze_metabolic_profile()`

**Input**: `dict[str, float]` mapping metabolite names to concentration values

**Output**:
```python
{
    "metabolite_scores": {"Glucose": 2.1, "Leucine": 1.8, ...},  # z-scores
    "elevated_metabolites": ["Glucose", "Leucine", "Isoleucine"],
    "insulin_resistance_score": 0.82,  # 0.0-1.0 sigmoid
    "metabolic_pattern": "bcaa_elevation",
    "risk_level": "high",
    "subtype_refinement": {
        "subtype": "metabolic_insulin_resistant",
        "confidence": "high",
        "reasoning": "Elevated BCAAs indicating insulin resistance."
    },
    "diabetes_confirmed": {
        "confirmed": True,
        "confidence": "high",
        "reasoning": "2 active pathways, IR score 0.82."
    },
    "interpretation": "Metabolic confirmation: POSITIVE ..."
}
```

## Metabolite Pathway Panels (78 metabolites)

| Pathway | Count | Key Metabolites |
|---------|-------|-----------------|
| Amino acid | 23 | BCAAs (Leu/Ile/Val), aromatic (Phe/Tyr/Trp), Glu, Gly |
| Carbohydrate | 14 | Glucose, Fructose, Mannose, 1,5-Anhydroglucitol, Inositol |
| Lipid | 12 | Cholesterol, Oleate, Palmitate, Stearate, Linoleate |
| TCA/Energy | 9 | Citrate, Pyruvate, Lactate, Succinate, Malate |
| Ketone/Oxidative | 8 | 3-Hydroxybutyrate, 2-Hydroxybutyrate, Urate, Creatinine |

## Insulin Resistance Score

The IR score (0.0-1.0) integrates multiple metabolic signals via sigmoid mapping:
- **BCAA elevation** — strongest predictor of insulin resistance
- **Glucose elevation** — direct hyperglycemia signal
- **Aromatic amino acid elevation** — Phe/Tyr/Trp (predictive biomarkers)
- **Lipid pathway score** — lipid dysregulation

Score > 0.7 indicates significant insulin resistance. Score ~0.5 is uninformative.

## Metabolic Pattern Classification

| Pattern | Dominant Pathway | Clinical Meaning |
|---------|-----------------|-----------------|
| `bcaa_elevation` | amino_acid | Classic insulin resistance biomarker |
| `lipid_dysregulation` | lipid | Lipid-driven — consider statin therapy |
| `glucose_dysregulation` | carbohydrate | Isolated glucose elevation — early/mild T2DM |
| `ketone_accumulation` | ketone_oxidative | Insulin deficiency component |
| `energy_metabolism_shift` | tca_energy | Mitochondrial dysfunction |
| `mixed` | 3+ pathways active | Multi-pathway dysregulation |
| `normal` | none active | No metabolic evidence of diabetes |

## Subtype Refinement

Metabolomics refines the diabetes subtype identified by transcriptomics:

| Metabolic Pattern | Refined Subtype | Implication |
|-------------------|----------------|-------------|
| BCAA + energy | metabolic_insulin_resistant | Classic T2DM |
| Lipid dominant | lipid_predominant | Statin-first approach |
| Glucose only | glucose_centric | Early/mild — lifestyle may suffice |
| Ketone accumulation | ketotic | Insulin deficiency — may need insulin |

## Diabetes Confirmation

| Condition | Confirmed | Confidence | Route |
|-----------|-----------|------------|-------|
| 2+ active pathways + IR > 0.6 | Yes | high | → Pharmacology (via Hospital) |
| 1 active pathway + IR > 0.55 | Yes | moderate | → Pharmacology (via Hospital) |
| Mild elevation + IR > 0.55 | Yes | low | → Pharmacology (via Hospital) |
| No active pathways | No | — | → Health Trainer (via Hospital) |

## Data Model

```python
class MetabolomicsFindings(BaseModel):
    metabolite_scores: dict[str, float]
    elevated_metabolites: list[str]
    insulin_resistance_score: float
    metabolic_pattern: str
    risk_level: RiskLevel
    subtype_refinement: dict[str, str]
    diabetes_confirmed: dict[str, Any]
    interpretation: str
```

## Tests

- `tests/test_metabolic_profile_analyzer.py` — 19 tests
  - Reference data loading, z-score computation, pathway matching
  - Pattern classification, diabetes confirmation, subtype refinement
  - IR score range validation, elevated metabolite detection
- `tests/test_metabolomics_agent.py` — 5 tests
  - Tool call flow, context injection, error handling, finding types

## Files

| File | Purpose |
|------|---------|
| `src/precision_health_agents/tools/metabolic_profile_analyzer.py` | Tool: z-score pathway scoring + IR score + pattern classification |
| `src/precision_health_agents/agents/metabolomics.py` | Agent: Claude tool-use loop |
| `src/precision_health_agents/prompts/metabolomics.txt` | System prompt |
| `src/precision_health_agents/models.py` | MetabolomicsFindings data model |
| `data/metabolomics/raw/diabetes_metabolomics.csv` | Processed reference dataset (61 × 86) |
| `data/metabolomics/raw/ST001906_raw.txt` | Raw data from Metabolomics Workbench |
| `scripts/process_metabolomics.py` | Data processing pipeline |
