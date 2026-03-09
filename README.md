# Precision Health Agents

Multi-agent AI system for DNA-precision diabetes care. Specialized agents analyze patient biological data across multiple omics layers — genomics, transcriptomics, proteomics, metabolomics — and synthesize personalized health assessments with multi-omics validation.

## Architecture

```
Patient
   ├── DNA Sequence ──→ [Genomics Agent] ──→ DMT1 / DMT2 / NONDM
   │                                                │
   ├── Clinical Chat ──→ [Doctor Agent] ──→ Diabetic / Non-Diabetic
   │                                                │
   │                                    ┌───────────┴───────────┐
   │                                 Hospital              Health Trainer
   │                                    │                       │
   │                    [Hospital Agent]                [Health Trainer Agent]
   │                     coordinates:                   (exercise prescription)
   │                    ┌──────┴──────┐
   │              [Transcriptomics] [Metabolomics]
   │                    └──────┬──────┘
   │                    ┌──────┴──────┐
   │                 Confirmed    False Positive
   │                    │               │
   │              [Pharmacology]   Health Trainer
   │              (drug plan)      (no drugs needed)
   │
   └── Evaluation Engine → Ralph Loop (auto-improve prompts) → Dashboard
```

**3-layer validation**: DNA (inherited risk) → Clinical (current state) → Molecular (pathway confirmation). The 3rd layer catches false positives — patients who look diabetic clinically but show no molecular pathway activation.

## Agents

| Agent | Domain | Backend | Status |
|-------|--------|---------|--------|
| **Genomics** | Inherited risk → DMT1/DMT2/NONDM | Pre-trained CNN (3-mer, 84%) | Implemented + evaluated |
| **Doctor** | Conversational intake → Diabetic/Non-Diabetic | Pre-trained MLP (Pima, 75%) | Implemented + evaluated |
| **Health Trainer** | Exercise prescription with clinical rules | ADA 2023 rules + 50-exercise DB | Implemented + evaluated |
| **Transcriptomics** | Pathway activity → subtype, false positive filter | GSE26168 z-score (110 genes, 5 pathways) | Implemented + evaluated |
| **Pharmacology** | Subtype-informed drug recommendations | ADA guideline DB (16 drugs × 8 classes) | Implemented |
| **Hospital** | Coordinates molecular tests (consent → TX + metab) | Transcriptomics + Metabolomics combined | Implemented |
| **Metabolomics** | Metabolic state (insulin resistance, lipids, BCAAs) | ST001906 z-score (78 metabolites, 5 pathways) | Implemented |
| **Proteomics** | Functional biomarkers (inflammatory, kidney/CV) | TBD | In progress (YH) |
| Clinical | Medical guidelines | JSON knowledge base | Stub |
| Literature | Publications | PubMed, Semantic Scholar | Stub |

## Quick Start

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...

# E2E pipeline
uv run python scripts/run.py --list                 # list cases
uv run python scripts/run.py --case 1               # run case 1 (real API)
uv run python scripts/run.py --all --mock            # run all cases (pre-recorded)

# Evaluation
uv run python scripts/evaluate.py --mock             # eval with pre-recorded outputs (15/15)
uv run python scripts/evaluate.py --ralph --iter 3   # Ralph Loop auto-improvement

# Dashboard
uv run streamlit run app/dashboard.py

# Tests
uv run pytest --tb=short -q                          # 200 tests
```

## E2E Pipeline Cases

| Case | DNA | Clinical | Decision | Pathway |
|------|-----|----------|----------|---------|
| 1. Confirmed Diabetic | DMT2 | Diabetic | Hospital | TX confirms → Pharmacology (triple therapy) |
| 2. DNA Override | DMT2 | Non-Diabetic | Hospital | TX confirms early signs → Pharmacology (monotherapy) |
| 3. Clinical Override | NONDM | Diabetic | Reconsider | Health Trainer (lifestyle, avoid drugs) |
| 4. Healthy | NONDM | Non-Diabetic | Health Trainer | Prevention (exercise plan) |

## Tech Stack

- **Python 3.12** + uv
- **Claude API** (Anthropic) — agent reasoning, synthesis, evaluation
- **asyncio** — parallel agent execution
- **Streamlit** — interactive dashboard

## Project Structure

```
precision-health-agents/
├── src/precision_health_agents/
│   ├── agents/          # 10 agents (7 implemented, 1 in progress, 2 stubs)
│   ├── tools/           # Python functions backing agent tools
│   ├── prompts/         # System prompts (.txt, editable by Ralph Loop)
│   ├── eval/            # Metrics, LLM-as-judge, Ralph Loop v2
│   ├── models.py        # AgentResult, *Findings, HealthAssessment
│   ├── orchestrator.py  # Agent execution
│   └── config.py        # Settings
├── data/                # Datasets (gitignored)
├── scripts/             # run.py (E2E pipeline), evaluate.py (eval + Ralph)
├── app/                 # Streamlit eval dashboard
├── tests/               # 200 tests
└── docs/                # Architecture, eval, vision, data, demo
```
