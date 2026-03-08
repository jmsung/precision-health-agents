# Architecture

## Pipeline

```
Patient Case → Orchestrator → [6 Agents in Parallel] → Blackboard → Synthesis → Report
                                                                          ↓
                                                              Evaluation Engine
                                                                          ↓
                                                              Ralph Loop (improve prompts)
                                                                          ↓
                                                              Dashboard (Streamlit)
```

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

## Specialized Agents

| Agent | Tools | Backend |
|-------|-------|---------|
| **Genomics** | `classify_dna`, `lookup_variant`, `search_clinvar`, `get_gene_summary` | Pre-trained CNN (diabetes), myvariant.info, Biopython Entrez |
| **Transcriptomics** | `run_gsea`, `classify_subtype` | GSEApy, PAM50 rules |
| **Proteomics** | `lookup_protein`, `get_protein_interactions` | UniProt REST, STRING DB |
| **Pharmacology** | `search_drug_gene_interactions`, `search_adverse_effects` | DGIpy, OpenFDA |
| **Clinical** | `lookup_guidelines`, `check_screening_criteria` | JSON knowledge base |
| **Literature** | `search_pubmed`, `search_semantic_scholar` | Biopython Entrez, semanticscholar |

All tools: **on-device or free API calls** — no GPU, no paid APIs beyond Claude.

## Base Agent (agentic tool-use loop)

1. Build messages from blackboard context + query
2. Call Claude API with agent-specific tools
3. If `tool_use` blocks → execute Python tool function → append result → call again
4. Loop until `end_turn` → return structured `AgentResult`

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

## Evaluation Pipeline

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

## Model Strategy

| Component | Model | Why |
|-----------|-------|-----|
| Agent analysis | `claude-sonnet-4-20250514` | Fast + cheap for tool use |
| Synthesis | `claude-opus-4-20250514` | Best quality for final report |
| LLM-as-judge | `claude-sonnet-4-20250514` | Cost-efficient scoring |
| Ralph Loop | `claude-opus-4-20250514` | Needs strong reasoning |

Cost: ~$0.15/run. Budget: ~$5-10 for hackathon.

## Project Structure

```
src/bioai/
├── config.py                  Pydantic settings (models, API keys, paths)
├── models.py                  Patient, AgentResult, TestCase Pydantic models
├── blackboard.py              Shared state for agent communication
├── orchestrator.py            2-phase: parallel agents → synthesis
├── synthesis.py               Claude-powered final report generation
├── agents/
│   ├── base.py                BaseAgent with agentic tool-use loop
│   ├── genomics.py            myvariant.info, ClinVar
│   ├── transcriptomics.py     GSEApy pathway enrichment
│   ├── proteomics.py          UniProt REST API
│   ├── pharmacology.py        DGIpy, OpenFDA
│   ├── clinical.py            Guidelines knowledge base
│   └── literature.py          PubMed (Entrez), Semantic Scholar
├── tools/                     Plain Python functions backing agent tools
│   ├── genomics_tools.py
│   ├── transcriptomics_tools.py
│   ├── proteomics_tools.py
│   ├── pharma_tools.py
│   ├── clinical_tools.py
│   └── literature_tools.py
├── prompts/                   System prompts as .txt (Ralph Loop edits these)
│   ├── genomics.txt
│   ├── transcriptomics.txt
│   ├── proteomics.txt
│   ├── pharmacology.txt
│   ├── clinical.txt
│   ├── literature.txt
│   └── synthesis.txt
└── eval/
    ├── metrics.py             Automated + LLM-as-judge scoring
    ├── cases.py               Test cases with ground truth
    ├── judge.py               Claude-as-evaluator
    └── ralph.py               Ralph Loop prompt optimizer
```

## Tech Stack

- Python 3.12 + uv
- Anthropic Claude API (agent reasoning)
- asyncio (concurrency)
- Streamlit (dashboard)
