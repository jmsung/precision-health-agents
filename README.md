# BioAI

Multi-agent AI system for personalized healthcare intelligence. Specialized agents analyze patient biological data across six domains and synthesize integrated health assessments with evidence-grounded reasoning.

## Architecture

```
Patient Case → Orchestrator → [6 Agents in Parallel] → Blackboard → Synthesis → Report
                                        ↓                                ↓
                                   Tool Calls                    Evaluation Engine
                              (APIs, databases)                  Ralph Loop (auto-improve)
```

## Agents

| Agent | Domain | Data Sources |
|-------|--------|-------------|
| Genomics | Variant interpretation | myvariant.info, ClinVar |
| Transcriptomics | Gene expression | GSEApy, PAM50 |
| Proteomics | Biomarker inference | UniProt, STRING DB |
| Pharmacology | Drug interactions | DGIpy, OpenFDA |
| Clinical Guidelines | Medical guidelines | JSON knowledge base |
| Literature Review | Publications | PubMed, Semantic Scholar |

## Quick Start

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...
uv run python scripts/run.py --case 1
```

## Tech Stack

- **Python 3.12** + uv
- **Claude API** (Anthropic) — agent reasoning, synthesis, evaluation
- **asyncio** — parallel agent execution
- **Streamlit** — interactive dashboard

## Project Structure

```
bioai/
├── src/bioai/
│   ├── agents/          # 6 domain-specific agents
│   ├── tools/           # Python functions backing agent tools
│   ├── prompts/         # System prompts (.txt, editable by Ralph Loop)
│   ├── eval/            # Metrics, LLM-as-judge, Ralph Loop
│   ├── models.py        # Patient, AgentResult, TestCase
│   ├── blackboard.py    # Shared state for agent communication
│   ├── orchestrator.py  # 2-phase: parallel agents → synthesis
│   └── config.py        # Settings
├── data/                # Datasets (gitignored)
├── scripts/             # CLI entry points
├── app/                 # Streamlit dashboard
├── tests/               # Tests
└── docs/                # Architecture, vision, data, demo
```
