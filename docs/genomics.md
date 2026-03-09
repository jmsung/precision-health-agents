# Genomics Agent — Implementation

## Overview

The Genomics Agent classifies DNA sequences for diabetes-associated risk using a pre-trained CNN model, then generates a clinical summary via Claude.

## Pipeline

```
DNA sequence (query)
        │
        ▼
  GenomicsAgent.analyze()
        │
        ▼
  Claude (tool-use loop)
        │
        ▼ calls
  classify_dna(sequence)
        │
        ▼
  CNN model (2-layer, 3-mer)
        │
        ▼
  predicted_class + probabilities
        │
        ▼
  Claude interprets → summary
        │
        ▼
  AgentResult (typed, passed to orchestrator)
```

## DNA Classifier Tool (`src/precision_health_agents/tools/dna_classifier.py`)

### Model

- **Architecture**: 2-layer 1D CNN (Embedding → Conv1D → MaxPool → Conv1D → MaxPool → Dropout → Flatten → Dense × 3)
- **Input**: DNA sequence tokenized as 3-mers, padded to length 9203
- **Output**: 3-class softmax — `DMT1`, `DMT2`, `NONDM`
- **Weights**: `data/dna_classification/models/CNN_2Layers_3mers.h5`
- **Accuracy**: ~84% on held-out test set

### Preprocessing

1. Split sequence into overlapping 3-mers (`ATG` → `atg tgc gca ...`)
2. Tokenize with Keras `Tokenizer` fit on the training dataset
3. Pad to `max_length=9203` with post-padding

### Classes

| Class | Meaning | Risk Level |
|-------|---------|------------|
| `DMT1` | Type 1 Diabetes-associated pattern | High |
| `DMT2` | Type 2 Diabetes-associated pattern | High |
| `NONDM` | No diabetes-associated pattern | Low |

### Interface

```python
from precision_health_agents.tools.dna_classifier import classify_dna

result = classify_dna("ATGCGT...")
# {
#   "predicted_class": "DMT2",
#   "probabilities": {"DMT1": 0.12, "DMT2": 0.81, "NONDM": 0.07},
#   "confidence": 0.81
# }
```

## Genomics Agent (`src/precision_health_agents/agents/genomics.py`)

Claude drives the tool-use loop. When a DNA sequence is present in the query, Claude calls `classify_dna`, receives the result, and generates a clinical interpretation.

### Output (`AgentResult`)

```python
AgentResult(
    agent="genomics",
    status=AgentStatus.SUCCESS,
    findings=GenomicsFindings(
        predicted_class="DMT2",
        confidence=0.81,
        probabilities={"DMT1": 0.12, "DMT2": 0.81, "NONDM": 0.07},
        risk_level=RiskLevel.HIGH,
        interpretation="Sequence shows Type 2 Diabetes-associated pattern with 81% confidence."
    ),
    summary="<Claude narrative for orchestrator>"
)
```

### Shared Models (`src/precision_health_agents/models.py`)

| Model | Purpose |
|-------|---------|
| `GenomicsFindings` | Typed output of the genomics agent |
| `AgentResult` | Generic wrapper for any agent's output |
| `HealthAssessment` | Orchestrator's final aggregated report |
| `RiskLevel` | `high` / `moderate` / `low` |
| `AgentStatus` | `success` / `error` |

## Data (`data/dna_classification/`)

```
data/dna_classification/
├── README.md
├── raw/
│   ├── Complete_DM_DNA_Sequence.csv   # labeled sequences (DMT1/DMT2/NONDM)
│   ├── DMT2_1296.fasta                # Type 2 Diabetes sequences
│   └── NONDM.fasta                    # Non-diabetic sequences
└── models/
    └── CNN_2Layers_3mers.h5           # pre-trained weights
```

## Tests

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_dna_classifier.py` | 5 | Tool: load model, load tokenizer, output structure, confidence, short sequences |
| `tests/test_genomics_agent.py` | 3 | Agent: tool invocation, summary, error handling |

All tests use mocked Anthropic API — no real API calls required.

