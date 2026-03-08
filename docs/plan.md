# BioAI End-to-End Pipeline Plan

## Goal

Build an **evolving** multi-agent healthcare intelligence system for the Bio x AI Hackathon (March 8, 2026 at YC HQ). Not just a static demo — a pipeline that measurably improves over cycles via the Ralph Loop.

**Team**: JS (infra + omics agents), YH (clinical agents + demo)

## Architecture

```
Patient Case → Orchestrator → [6 Agents in Parallel] → Blackboard → Synthesis → Report
                                                                          ↓
                                                              Evaluation Engine
                                                                          ↓
                                                              Ralph Loop (improve prompts)
                                                                          ↓
                                                              Dashboard (Streamlit)
```

---

## Project Structure

```
bioai/
├── src/bioai/
│   ├── config.py                  Settings (models, API keys, paths)
│   ├── models.py                  Patient, AgentResult, TestCase dataclasses
│   ├── blackboard.py              Shared state for agent communication
│   ├── orchestrator.py            2-phase: parallel agents → synthesis
│   ├── synthesis.py               Claude-powered final report generation
│   ├── agents/
│   │   ├── base.py                BaseAgent with agentic tool-use loop
│   │   ├── genomics.py        JS  myvariant.info, ClinVar
│   │   ├── transcriptomics.py JS  GSEApy pathway enrichment
│   │   ├── proteomics.py      JS  UniProt REST API
│   │   ├── pharmacology.py    YH  DGIpy, OpenFDA
│   │   ├── clinical.py        YH  Guidelines knowledge base
│   │   └── literature.py      YH  PubMed (Entrez), Semantic Scholar
│   ├── tools/                     Plain Python functions backing agent tools
│   │   ├── genomics_tools.py
│   │   ├── transcriptomics_tools.py
│   │   ├── proteomics_tools.py
│   │   ├── pharma_tools.py
│   │   ├── clinical_tools.py
│   │   └── literature_tools.py
│   ├── prompts/                   System prompts as .txt (Ralph Loop edits these)
│   │   ├── genomics.txt
│   │   ├── transcriptomics.txt
│   │   ├── proteomics.txt
│   │   ├── pharmacology.txt
│   │   ├── clinical.txt
│   │   ├── literature.txt
│   │   └── synthesis.txt
│   └── eval/
│       ├── metrics.py             Automated + LLM-as-judge scoring
│       ├── cases.py               Test cases with ground truth
│       ├── judge.py               Claude-as-evaluator
│       └── ralph.py               Ralph Loop prompt optimizer
├── scripts/
│   ├── run.py                     CLI entry point
│   ├── demo.py                    Demo script
│   ├── download_data.py           Dataset download + cache
│   └── evaluate.py                Run eval suite + Ralph Loop
├── app/
│   └── dashboard.py               Streamlit dashboard (3 tabs)
├── data/                          Local data cache (gitignored)
└── pyproject.toml
```

---

## Data Pipeline

### Datasets (all free, <1GB total, <1 hour setup)

| Data Type | Dataset | Use |
|-----------|---------|-----|
| Genomics | ClinVar `variant_summary.txt` (~50MB) | Variant → disease lookup |
| Genomics + Transcriptomics | METABRIC from Kaggle (~50MB) | Mutations + mRNA z-scores + clinical |
| Pharmacology | PharmGKB clinical annotations | Drug-gene associations |
| Drug Safety | Kaggle Drug Side Effects (<50MB) | Adverse reactions |

### Patient Cases (from METABRIC)

- **Case 1**: PIK3CA + TP53 mutations, high ESR1, on tamoxifen → treatment optimization
- **Case 2**: BRCA1 variant, triple-negative, young → risk assessment + clinical trials
- **Case 3**: Multiple low-significance variants, conflicting signals → diagnostic dilemma

### Format
- All data as pandas DataFrames, cached as Parquet in `data/`
- Each tool function: query in (gene, variant, drug) → dict out

---

## Agent Architecture

### Base Agent (agentic tool-use loop)

1. Build messages from blackboard context + query
2. Call Claude API with agent-specific tools
3. If `tool_use` blocks → execute Python tool function → append result → call again
4. Loop until `end_turn` → return structured `AgentResult`

### Agent Tools

| Agent | Tools | Backend | Rating |
|-------|-------|---------|--------|
| **Genomics** | `lookup_variant`, `search_clinvar`, `get_gene_summary` | myvariant.info, Biopython Entrez | 5/5 |
| **Transcriptomics** | `run_gsea`, `classify_subtype` | GSEApy, PAM50 rules | 5/5 |
| **Proteomics** | `lookup_protein`, `get_protein_interactions` | UniProt REST, STRING DB | 5/5 |
| **Pharmacology** | `search_drug_gene_interactions`, `search_adverse_effects` | DGIpy, OpenFDA | 5/5 |
| **Clinical** | `lookup_guidelines`, `check_screening_criteria` | JSON knowledge base | 5/5 |
| **Literature** | `search_pubmed`, `search_semantic_scholar` | Biopython Entrez, semanticscholar | 5/5 |

All tools: **on-device or free API calls** — no GPU, no paid APIs beyond Claude.

---

## Orchestrator + Blackboard

### Blackboard
- `patient: Patient` — input data
- `query: str` — clinical question
- `agent_results: dict[str, AgentResult]` — accumulated findings
- `get_context_for_agent(name)` → patient + prior findings

### Orchestrator (2-phase)
- **Phase 1**: 6 agents run in parallel (`asyncio.gather`). Each reads patient data only.
- **Phase 2**: Synthesis (Claude Opus) reads ALL agent results → unified health assessment.
- **(Optional)**: Re-query agents if synthesis finds contradictions.

---

## Evaluation Pipeline

### Metrics

| Category | Metric | Method | Target |
|----------|--------|--------|--------|
| Clinical | Variant accuracy | vs ClinVar ground truth | >80% |
| Clinical | Drug interaction recall | vs PharmGKB | >70% |
| Quality | Relevance (1-5) | LLM-as-judge | >3.5 |
| Quality | Completeness (1-5) | LLM-as-judge | >3.5 |
| Quality | Citation count | Automated | >3/agent |
| Safety | Harmful recommendation | LLM-as-judge | 0 |
| Safety | Hallucination rate | LLM-as-judge vs ground truth | <10% |
| System | Latency per agent | time.time() | <30s |
| System | Cost per run | API usage field | Track |
| System | Consensus rate | Agent agreement | Track |

### LLM-as-Judge
Claude Sonnet scores each agent on relevance, completeness, accuracy, safety. Returns `{score, explanation}`.

---

## Ralph Loop (Iterative Improvement)

```
while not all_criteria_met (max 5 iterations):
    1. Run all test cases through full pipeline
    2. Score every agent on every metric
    3. Find worst-performing agent + metric
    4. Use Claude Opus to rewrite that agent's system prompt
       (input: current prompt + eval results + failure examples)
    5. Re-run evaluation with new prompt
    6. If improved → keep new prompt, log change
       If degraded → revert, try different approach
    7. Log learnings to guardrails file
```

Prompts are `.txt` files on disk — Ralph Loop reads, rewrites, saves. No code changes per iteration.

---

## Dashboard (Streamlit)

| Tab | Content |
|-----|---------|
| **MDT Meeting** | Select patient → Run Analysis → agent cards (color-coded confidence) → synthesis report |
| **Evaluation** | Test case x Agent score heatmap, aggregate metrics, latency/cost charts |
| **Ralph Loop** | Iteration timeline, before/after scores, prompt viewer |

---

## Demo Plan (2 minutes)

1. **(15s)** Problem: cancer treatment needs genomics + drugs + literature + guidelines — no single clinician has it all
2. **(60s)** Live: METABRIC patient → 6 agents in parallel → show analyses → synthesis report
3. **(30s)** Dashboard: eval scores → Ralph Loop: "3 iterations: relevance 3.2→4.1, hallucination 12%→4%"
4. **(15s)** Architecture slide

**Prep**: Pre-cache API responses. Fallback pre-recorded run.

---

## Work Split

### JS: Infrastructure + Omics Agents

| Time | Task |
|------|------|
| 10:00-10:30 | `pyproject.toml` deps, `models.py`, `blackboard.py` |
| 10:30-11:30 | `base.py` (agentic loop), `orchestrator.py` (2-phase) |
| 11:30-12:30 | Genomics agent + tools |
| 1:00-2:00 | Transcriptomics agent + tools |
| 2:00-2:30 | Proteomics agent + tools |
| 2:30-3:30 | Eval framework (metrics, judge, cases) |
| 3:30-4:30 | Ralph Loop |
| 4:30-6:15 | Integration, bug fixes, polish |

### YH: Clinical Agents + Demo

| Time | Task |
|------|------|
| 10:00-10:30 | `download_data.py`, build 3 patient test cases |
| 10:30-11:30 | Pharmacology agent + tools |
| 11:30-12:30 | Clinical Guidelines agent + knowledge base |
| 1:00-2:00 | Literature Review agent + tools |
| 2:00-2:30 | Synthesis prompt |
| 2:30-3:30 | Streamlit dashboard (MDT Meeting tab) |
| 3:30-4:30 | Evaluation + Ralph Loop tabs |
| 4:30-6:15 | Demo prep, polish, submission |

### Sync Points
- **10:30**: Agree on BaseAgent interface + AgentResult schema
- **12:30** (lunch): First 2-agent test (genomics + pharmacology)
- **2:30**: All 6 agents compile, first full pipeline run
- **4:30**: Full demo end-to-end with eval

---

## Priority Tiers

### P0 — MVP by 4:00 PM
- [ ] `models.py`, `blackboard.py`, `base.py`, `orchestrator.py`
- [ ] 3 agents: Genomics, Pharmacology, Literature
- [ ] 1 patient test case
- [ ] `scripts/run.py` CLI
- [ ] Streamlit MDT Meeting tab

### P1 — Should Have by 5:30 PM
- [ ] Remaining 3 agents (Transcriptomics, Proteomics, Clinical)
- [ ] Evaluation framework with LLM-as-judge
- [ ] Ralph Loop (1-2 iterations)
- [ ] Streamlit evaluation tab

### P2 — Nice to Have
- [ ] Streamlit Ralph Loop tab
- [ ] Debate/contradiction resolution
- [ ] More test cases

### Explicitly Skip
- GPU-requiring models (ESM-2, DNABERT, AlphaFold)
- FHIR data formatting
- Synthea patient generation
- Database/persistent storage
- Auth/user management

---

## Dependencies

```toml
dependencies = [
    "anthropic>=0.84.0",
    "myvariant",
    "biopython",
    "gseapy",
    "dgipy",
    "chembl-webresource-client",
    "semanticscholar",
    "pydantic>=2.0",
    "pandas",
    "pyarrow",
    "streamlit",
    "plotly",
    "requests",
]
```

## Model Strategy

| Component | Model | Why |
|-----------|-------|-----|
| Agent analysis | `claude-sonnet-4-6` | Fast + cheap for tool use |
| Synthesis | `claude-opus-4-6` | Best quality for final report |
| LLM-as-judge | `claude-sonnet-4-6` | Cost-efficient scoring |
| Ralph Loop | `claude-opus-4-6` | Needs strong reasoning |

Cost: ~$0.15/run. Budget: ~$5-10 for hackathon.

## Verification

```bash
uv run python scripts/run.py --case 1              # full pipeline
uv run python scripts/evaluate.py                   # eval suite
uv run python scripts/evaluate.py --ralph --iter 3  # Ralph Loop
uv run streamlit run app/dashboard.py               # dashboard
```
