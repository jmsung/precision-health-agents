# Transcriptomics Agent — Implementation

## Overview

The Transcriptomics Agent is the **3rd layer of diabetes validation** — after DNA (Genomics) and clinical assessment (Doctor) route a patient to hospital, it confirms or rejects the diabetes diagnosis at the molecular level. It analyzes gene expression profiles for diabetes-related pathway activity, classifies the diabetes subtype, flags complication risks, and recommends monitoring levels.

**Key role**: False positive filter. If gene expression shows no diabetes pathway activation, the patient is rerouted to the Health Trainer instead of proceeding to unnecessary medication via Pharmacology.

## Pipeline

```
Gene expression data (dict of gene -> value)
        |
  TranscriptomicsAgent.analyze()
        |
  Claude (tool-use loop)
        |  calls
  analyze_gene_expression(gene_expression)
        |
  Z-score each gene against GSE26168 reference
        |
  Compute 5 pathway scores (mean z-score per panel)
        |
  Classify diabetes subtype from dominant pathway
        |
  Flag complication risks from pathway co-activation
        |
  Recommend monitoring level
        |
  Confirm or reject diabetes (false positive filter)
        |
  Route: pharmacology (confirmed) or health_trainer (false positive)
        |
  Claude interprets -> clinical summary
        |
  TranscriptomicsFindings
```

## Data Source

**GSE26168** from NCBI GEO — blood transcriptome profiling of:
- 8 healthy controls
- 7 IFG (impaired fasting glucose / pre-diabetic)
- 9 T2DM patients

Platform: Illumina HumanRef-8 v3.0 (24,526 probes). Processed to 110 diabetes-relevant genes across 5 pathway panels.

## Tool: `analyze_gene_expression()`

**Input**: `dict[str, float]` mapping gene symbols to expression values

**Output**:
```python
{
    "pathway_scores": {"beta_cell_stress": 1.42, "inflammation_immune": 0.8, ...},
    "dominant_pathway": "beta_cell_stress",
    "active_pathways": ["beta_cell_stress", "inflammation_immune"],
    "risk_level": "high",
    "diabetes_subtype": {"subtype": "beta_cell_failure", "confidence": "moderate"},
    "complication_risks": [
        {"complication": "beta_cell_exhaustion", "severity": "high", "evidence": "..."},
        {"complication": "cardiovascular", "severity": "moderate", "evidence": "..."},
    ],
    "monitoring": {
        "level": "actionable",
        "follow_ups": ["C-peptide evaluation", "CV risk assessment"],
    },
    "diabetes_confirmed": {
        "confirmed": True,
        "confidence": "high",
        "reasoning": "2 active diabetes pathways detected"
    },
    "recommendation": "pharmacology",  # or "health_trainer" if false positive
    "dysregulated_genes": [{"gene": "INS", "z_score": 2.1, "direction": "up", ...}],
    "interpretation": "Diabetes subtype: beta cell failure T2DM ..."
}
```

## Diabetes Subtype Classification

| Subtype | Dominant Pathway | Clinical Meaning |
|---------|-----------------|-----------------|
| `inflammation_dominant` | inflammation_immune | Immune-driven insulin resistance |
| `beta_cell_failure` | beta_cell_stress | Primary secretory deficit |
| `metabolic_insulin_resistant` | insulin_resistance or oxidative | Metabolic/mitochondrial dysfunction |
| `fibrotic_complication` | fibrosis_ecm | Organ damage (kidney, etc.) |
| `mixed` | 3+ co-activated | Multi-mechanism, needs broad approach |
| `normal` | none | No transcriptomic diabetes signal |

Confidence based on gap between top 2 pathway scores: >1.0 = high, >0.3 = moderate, else low.

## Complication Risk Flags

| Complication | Trigger | Follow-up |
|-------------|---------|-----------|
| Diabetic kidney disease | fibrosis > 0.5, or fibrosis + inflammation > 0.3 each | Nephrology referral, eGFR/UACR monitoring |
| Cardiovascular | inflammation > 0.5 + oxidative > 0.3 | CV risk assessment, lipid panel |
| Beta cell exhaustion | beta_cell_stress > 0.5 | C-peptide, insulin secretion eval |
| Neuropathy | oxidative > 0.5 + insulin_resistance > 0.3 | Peripheral neuropathy screening |

## Monitoring Levels

| Level | Risk | Meaning |
|-------|------|---------|
| `actionable` | high | Supports guideline-based management decisions |
| `monitoring` | moderate | Closer follow-up for specific complications |
| `exploratory` | low | Hypothesis-generating, routine monitoring |

## Diabetes Confirmation (False Positive Filter)

The key decision: does the gene expression data **confirm** the diabetes diagnosis from DNA + clinical assessment?

| Condition | Confirmed | Confidence | Route |
|-----------|-----------|------------|-------|
| 2+ active pathways OR max score > 1.0 | Yes | high | -> Pharmacology |
| 1 active pathway | Yes | moderate | -> Pharmacology |
| max > 0.3 with 2+ mildly elevated | Yes | low | -> Pharmacology |
| No significant pathway activation | No | — | -> Health Trainer |

**False positive scenario**: Patient has clinical diabetes indicators + high-risk DNA, but gene expression shows no diabetes pathway activation. This suggests the initial diagnosis may be premature — reroute to Health Trainer for lifestyle intervention instead of drugs.

## Data Model

```python
class TranscriptomicsRecommendation(str, Enum):
    PHARMACOLOGY = "pharmacology"
    HEALTH_TRAINER = "health_trainer"

class TranscriptomicsFindings(BaseModel):
    pathway_scores: dict[str, float]
    dominant_pathway: str
    active_pathways: list[str]
    risk_level: RiskLevel
    dysregulated_genes: list[dict[str, Any]]
    diabetes_confirmed: dict[str, Any]
    diabetes_subtype: dict[str, str]
    complication_risks: list[dict[str, str]]
    monitoring: dict[str, Any]
    recommendation: TranscriptomicsRecommendation
    interpretation: str
```

## Inter-Agent Communication

The agent accepts `context: dict` with prior agent findings:
- `context["genomics"]` — predicted_class, confidence
- `context["doctor"]` — prediction, probability

These are injected into the system prompt so the LLM can contextualize the transcriptomic findings.

**Output flows to**:
- If `recommendation == "pharmacology"` → PharmacologyAgent (subtype + complication risks → personalized medication plan from 16-drug ADA guideline database)
- If `recommendation == "health_trainer"` → Health Trainer (false positive — lifestyle intervention, no drugs)

## Tests

- `tests/test_gene_expression_analyzer.py` — 28 tests
  - Pathway detection, subtype classification, complication risks, monitoring levels, reference data
  - Diabetes confirmation (false positive filter): confirmed/not confirmed, confidence levels, interpretation
- `tests/test_transcriptomics_agent.py` — 5 tests
  - Tool call flow, context injection, error handling, finding types

## Files

| File | Purpose |
|------|---------|
| `src/bioai/tools/gene_expression_analyzer.py` | Tool: z-score pathway scoring + subtype/complications |
| `src/bioai/agents/transcriptomics.py` | Agent: Claude tool-use loop |
| `src/bioai/prompts/transcriptomics.txt` | System prompt |
| `src/bioai/models.py` | TranscriptomicsFindings data model |
| `data/transcriptomics/raw/diabetes_transcriptomics.csv` | Processed reference dataset |
| `scripts/process_transcriptomics.py` | Data processing pipeline |
