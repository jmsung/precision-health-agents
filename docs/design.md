# System Design

## Architecture

```
Patient Query + Health Profile
        │
        ▼
  ┌─────────────┐
  │ Orchestrator │
  └─────┬───────┘
        │ (concurrent)
  ┌─────┼─────────────────────────────┐
  │     │     │      │      │         │
  ▼     ▼     ▼      ▼      ▼         ▼
Genomics  Transcr  Proteo  Pharma  Clinical  Literature
  │     │     │      │      │         │
  └─────┼─────────────────────────────┘
        │
        ▼
  ┌─────────────┐
  │  Aggregator  │
  │  (Claude)    │
  └─────────────┘
        │
        ▼
  Structured Report
  - Risk signals
  - Preventive strategies
  - Lifestyle recommendations
  - Clinical referral flags
  - Citations
```

## Components

### BaseAgent
- Abstract class with `analyze(query, context) -> dict`
- Each agent has a `name` and `role`

### Specialized Agents
| Agent | Domain | Data Sources |
|-------|--------|-------------|
| Genomics | Variant interpretation | HyenaDNA / DNABERT-2 |
| Transcriptomics | Gene expression | Expression databases |
| Proteomics | Biomarker inference | Protein databases |
| Pharmacology | Drug interactions | DrugBank, interactions DBs |
| Clinical | Guideline interpretation | Clinical guidelines |
| Literature | Publication review | PubMed API |

### Orchestrator
- Registers agents, runs them concurrently via `asyncio.gather`
- Aggregates results into a unified report

### Config
- Model settings, API keys from environment variables

## Tech Stack
- Python 3.12 + uv
- Anthropic Claude API (agent reasoning)
- HyenaDNA / DNABERT-2 (genomics embeddings)
- asyncio (concurrency)
