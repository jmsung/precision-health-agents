# Architecture

## Core Pipeline: DNA-Precision Diabetes Decision

The primary use case is a multi-stage pipeline with **3-layer validation**: DNA assessment first, then clinical assessment, then transcriptomic confirmation. The system uses three independent evidence sources to confirm or reject a diabetes diagnosis, catching false positives before unnecessary medication.

```
Patient
   |
   +--- DNA Sequence --> [Genomics Agent] --> DMT1 / DMT2 / NONDM
   |                                                  |
   |                                    +-------------+---------------+
   |                                 High risk                   No risk (NONDM)
   |                                    |                            |
   +--- Clinical Chat --> [Doctor Agent] --> Diabetic / Non-Diabetic  |
   |         (8 features gathered                   |               |
   |          through conversation)                 |               |
   |                                    +-----------+-----------+   |
   |                                 Diabetic             Non-Diabetic
   |                                 + High DNA risk       + No DNA risk
   |                                    |                       |
   |                              -> Hospital              -> Reconsider
   |                                    |                  (may not need drugs)
   |                                    |
   |  +---------------------------------+
   |  |  HOSPITAL PATHWAY
   |  |
   |  +--- Gene Expression --> [Transcriptomics Agent]
   |  |                              |
   |  |                    analyze_gene_expression (110-gene panel)
   |  |                    -> 5 pathway scores (beta cell, inflammation,
   |  |                       insulin resistance, fibrosis, oxidative)
   |  |                    -> Diabetes subtype (inflammation_dominant /
   |  |                       beta_cell_failure / metabolic_insulin_resistant /
   |  |                       fibrotic_complication / mixed)
   |  |                    -> Complication risks (kidney, CV, neuropathy)
   |  |                    -> Monitoring level (actionable/monitoring/exploratory)
   |  |                    -> Diabetes confirmed? (false positive filter)
   |  |                              |
   |  |                    +--------+--------+
   |  |                 Confirmed         NOT confirmed
   |  |                 (pathways          (no pathway
   |  |                  active)            activation)
   |  |                    |                    |
   |  +--- Subtype --> [Pharmacology Agent]   Health Trainer
   |                        (TODO)            (false positive --
   |                        |                  no drugs needed)
   |                  ADA guideline-based medication selection
   |                  -> Drug recommendation by subtype
   |                  -> Drug interaction checks
   |                  -> Treatment plan
   |
   +--- (HEALTH_TRAINER) --> [Health Trainer Agent]
   |                              |
   |                    classify_workout_type (ADA 2023 rules)
   |                    + diabetes context from Genomics & Doctor
   |                              |
   |                    recommend_exercises (50-exercise DB)
   |                              |
   |                    -> Personalised weekly plan
   |
   +--- (Background) --> [Proteomics, Clinical, Literature]
                                         |
                                   Synthesis (Claude Opus)
                                         |
                                   Unified Health Report
```

## Full Multi-Agent Pipeline

```
Patient Case -> Orchestrator -> [Agents] -> Blackboard -> Synthesis -> Report
                                                                 |
                                                     Evaluation Engine
                                                                 |
                                                     Ralph Loop (improve prompts)
                                                                 |
                                                     Dashboard (Streamlit)
```

## Specialized Agents

| Agent | Tools | Backend | Role in Diabetes Pipeline |
|-------|-------|---------|--------------------------|
| **Genomics** | `classify_dna` | Pre-trained 2-layer CNN (3-mer tokenization) | Genetic predisposition: DMT1/DMT2/NONDM |
| **Doctor** | `classify_diabetes` | Pre-trained MLP (8 clinical features) | Conversational intake -> hospital/health trainer routing |
| **Health Trainer** | `classify_workout_type`, `recommend_exercises` | ADA 2023 clinical rules + 50-exercise DB | Exercise prescription for HEALTH_TRAINER referrals |
| **Transcriptomics** | `analyze_gene_expression` | GSE26168 reference + z-score pathway scoring (110 genes, 5 pathways) | 3rd validation layer: confirms/rejects diabetes, subtype, complication risks |
| **Pharmacology** | TBD | TBD | Subtype-informed medication selection (TODO) |
| **Proteomics** | `lookup_protein`, `get_protein_interactions` | UniProt REST, STRING DB | Biomarker inference (stub) |
| **Clinical** | `lookup_guidelines`, `check_screening_criteria` | JSON knowledge base | Evidence-based guideline interpretation (stub) |
| **Literature** | `search_pubmed`, `search_semantic_scholar` | Biopython Entrez, semanticscholar | Latest research on DNA-matched treatment (stub) |

### Doctor Agent: Conversational Clinical Intake

Unlike the other agents which receive structured input, the Doctor Agent gathers data through natural conversation:

```
Patient speaks freely:
  "I'm 42, female, my glucose was 160 last week, 85kg at 165cm,
   pregnant twice, my mom has diabetes, I don't know my insulin."

Doctor Agent extracts:
  pregnancies=2, glucose=160, blood_pressure=80, skin_thickness=28,
  insulin=0, bmi=31.2, diabetes_pedigree_function=0.5, age=42

Calls: classify_diabetes(**extracted_values)
Returns: DoctorFindings(prediction, probability, risk_level, recommendation)
```

Recommendation logic (combined with genomics):

| Genomics | Clinical | Decision |
|---|---|---|
| DMT1 or DMT2 | Diabetic | -> Hospital (confirmed) |
| DMT1 or DMT2 | Non-Diabetic | -> Hospital (genetic override -- early intervention) |
| NONDM | Diabetic | -> Reconsider (may not need drugs) |
| NONDM | Non-Diabetic | -> Health trainer (prevention) |

### Transcriptomics Agent: Hospital Pathway Analysis

When a patient is routed to Hospital, the Transcriptomics Agent analyzes their gene expression profile:

```
Gene expression data (110-gene panel)
        |
  TranscriptomicsAgent.analyze()
        |
  Claude (tool-use loop)
        |  calls
  analyze_gene_expression(gene_expression)
        |
  Z-score against GSE26168 reference (24 samples: 8 control, 7 IFG, 9 T2DM)
        |
  5 pathway scores + subtype + complications + monitoring
        |
  Confirm or reject diabetes (false positive filter)
        |
  Claude interprets -> summary
        |
  TranscriptomicsFindings -> PharmacologyAgent (confirmed) or HealthTrainer (false positive)
```

**Pathway panels** (110 diabetes-relevant genes):
1. Beta cell stress (20 genes) -- INS, PDX1, GCK, TCF7L2, ABCC8...
2. Inflammation/immune (25 genes) -- TNF, IL6, IL1B, TLR4, NLRP3...
3. Insulin resistance (25 genes) -- INSR, IRS1, AKT1, PPARG, FOXO1...
4. Fibrosis/ECM (21 genes) -- COL1A1, TGFB1, MMP9, FN1, VIM...
5. Oxidative/mitochondrial (22 genes) -- SOD2, GPX1, SIRT1, UCP2, NFE2L2...

**Subtype classification** (based on dominant pathway):

| Subtype | Dominant Pathway | Clinical Implication |
|---------|-----------------|---------------------|
| `inflammation_dominant` | inflammation_immune | Immune-driven insulin resistance |
| `beta_cell_failure` | beta_cell_stress | Primary secretory deficit |
| `metabolic_insulin_resistant` | insulin_resistance or oxidative | Metabolic/mitochondrial dysfunction |
| `fibrotic_complication` | fibrosis_ecm | Organ damage (kidney, etc.) |
| `mixed` | 3+ co-activated | Multi-mechanism |

**Complication risk flags**:
- Diabetic kidney disease (fibrosis + inflammation)
- Cardiovascular (inflammation + oxidative stress)
- Beta cell exhaustion (severe beta cell stress)
- Neuropathy (oxidative + insulin resistance)

All tools: **on-device or free API calls** -- no GPU, no paid APIs beyond Claude.

## Base Agent (agentic tool-use loop)

1. Build messages from blackboard context + query
2. Call Claude API with agent-specific tools
3. If `tool_use` blocks -> execute Python tool function -> append result -> call again
4. Loop until `end_turn` -> return structured `AgentResult`

## Orchestrator + Blackboard

### Blackboard
- `patient: Patient` -- input data
- `query: str` -- clinical question
- `agent_results: dict[str, AgentResult]` -- accumulated findings
- `get_context_for_agent(name)` -> patient + prior findings

### Orchestrator (2-phase)
- **Phase 1**: Agents run (some parallel, some sequential). Each reads patient data + prior findings.
- **Phase 2**: Synthesis (Claude Opus) reads ALL agent results -> unified health assessment.
- **(Optional)**: Re-query agents if synthesis finds contradictions.

## Evaluation Pipeline

Three agents are evaluated: Genomics, Doctor, and Health Trainer (case-4 only). Four test cases with ground truth cover all branches of the decision matrix.

### Test Cases

| Case | DNA Input | Clinical Input | HT Vitals | Expected Decision |
|------|-----------|----------------|-----------|-------------------|
| case-1 | DMT2 (800bp) | diabetic (glucose=189, bmi=30.1, age=59) | — | hospital |
| case-2 | DMT2 (same) | non_diabetic (glucose=89, bmi=28.1, age=21) | — | hospital |
| case-3 | NONDM (1500bp) | diabetic (same as case-1) | — | reconsider |
| case-4 | NONDM (same) | non_diabetic (same as case-2) | age=21, Male, 170cm, 81.2kg, freq=0, dur=0 | health_trainer |

DNA sequences are real from the DNA classification dataset. Clinical features are real Pima Indians Diabetes rows. Health trainer vitals are derived from case-4 demographics. All stored in `src/bioai/eval/data/case_inputs.json`.

### Evaluation Layers

**Layer 1 — Tool Accuracy (deterministic, no LLM)**
Checks whether the underlying ML/rule tool returned the correct prediction:
- Genomics: `classify_dna(sequence)` → `predicted_class == expected_dna_class`?
- Doctor: `classify_diabetes(features)` → `prediction == expected_clinical_prediction`?
- Health Trainer: `classify_workout_type(vitals)` → `fitness_level == expected_fitness_level`?
- Transcriptomics: `analyze_gene_expression(profile)` → expected subtype/pathway activation

Binary: 1.0 (correct) or 0.0 (wrong).

**Layer 2 — Agent Quality (LLM-as-judge)**
Claude Sonnet scores each agent's output on a 1-5 scale. The judge sees the agent's `AgentResult` (findings JSON + summary) alongside the case ground truth:

| Dimension | Question |
|-----------|----------|
| Relevance (1-5) | Does the output address the clinical question? |
| Completeness (1-5) | Are all relevant findings covered? |
| Accuracy (1-5) | Is the interpretation clinically correct given ground truth? |
| Safety (1-5) | Any harmful or misleading recommendations? (5 = safe) |

**Layer 3 — Decision Correctness (deterministic)**
Combines genomics + doctor outputs through the decision matrix:

| Genomics | Doctor | Expected Decision |
|---|---|---|
| DMT1/DMT2 | Diabetic | Hospital (confirmed) |
| DMT1/DMT2 | Non-Diabetic | Hospital (DNA override) |
| NONDM | Diabetic | Reconsider (lifestyle first) |
| NONDM | Non-Diabetic | Health Trainer (prevention) |

### Current Results (13/13 deterministic pass)

| Case | Genomics Tool | Doctor Tool | HT Tool | Decision | Judge (genomics) | Judge (doctor) | Judge (HT) |
|------|--------------|-------------|---------|----------|-----------------|----------------|------------|
| case-1 | PASS | PASS | — | PASS | 5/4/5/5 | 5/4/5/5 | — |
| case-2 | PASS | PASS | — | PASS | 5/4/5/4 | 2/1/1/1* | — |
| case-3 | PASS | PASS | — | PASS | 4/3/5/3 | 2/1/2/2* | — |
| case-4 | PASS | PASS | PASS | PASS | 5/5/5/5 | 5/4/5/5 | 5/4/5/5 |

*Doctor case-2/3 judge scores are low by design — the doctor agent doesn't see DNA results. The decision matrix (Layer 3) handles the cross-agent logic correctly.

### Execution Modes

```bash
uv run python scripts/evaluate.py              # real mode (all agents + judge via Claude API)
uv run python scripts/evaluate.py --mock       # mock mode (pre-recorded outputs, scoring only)
uv run python scripts/evaluate.py --save       # real mode + save outputs for future mock runs
uv run python scripts/evaluate.py --ralph --iter 3  # Ralph Loop (3 prompt optimization iterations)
```

- **Real mode**: ~$0.15/run, ~30s. Runs agents + judge via Claude API.
- **Mock mode**: Free, instant. Uses saved JSON from `src/bioai/eval/data/mock_outputs/`.
- **Workflow**: real `--save` → iterate on eval logic in mock → Ralph Loop → real again.

## Ralph Loop (Iterative Prompt Optimization)

Automated prompt improvement: evaluate → find weakest agent/metric → rewrite prompt → re-evaluate.

### Data Flow

```
evaluate_case() × 4 cases
    → Layer 1: tool accuracy (deterministic)
    → Layer 2: judge scores (Claude Sonnet, 4 dimensions × 1-5)
    → Layer 3: decision correctness (deterministic)
           ↓
collect_judge_averages()
    → Per-agent averages: {genomics: {rel: 5.0, comp: 4.0, ...}, doctor: {...}, health_trainer: {...}}
           ↓
_find_weakest()
    → Scan all (agent, metric) pairs → pick the single lowest score
    → e.g. ("health_trainer", "completeness", 1.0)
           ↓
ralph_iterate()
    → Read src/bioai/prompts/{agent}.txt
    → Send to Claude Opus:
        System: "You are an expert prompt engineer for biomedical AI agents.
                 Rewrite to improve the weakest metric. Keep tool-calling instructions intact."
        User:   "Agent: health_trainer, Weakest: completeness (1.0),
                 All scores: {rel: 2.0, comp: 1.0, acc: 3.0, safe: 4.0}
                 Current prompt: [full text]"
    → Opus returns rewritten prompt
    → Save to disk (overwrites the .txt file)
           ↓
Re-run full evaluation (all 4 cases × all agents × all 3 layers)
    → New judge averages → next iteration
```

### Results (3 iterations)

| Iter | Target | Score | Prompt Rewritten | Effect |
|------|--------|-------|------------------|--------|
| 1 | health_trainer/completeness | 1.0 | health_trainer.txt | rel 2→4, comp 1→3, acc 3→4, safe 4→5 |
| 2 | doctor/completeness | 2.5 | doctor.txt | Added verification step, systematic checklist |
| 3 | doctor/completeness | 2.8 | doctor.txt | Further persistence + response structure |

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
├── models.py                  AgentResult, GenomicsFindings, DoctorFindings, TranscriptomicsFindings, HealthTrainerFindings
├── blackboard.py              Shared state for agent communication
├── orchestrator.py            2-phase: parallel agents → synthesis
├── agents/
│   ├── base.py                BaseAgent ABC
│   ├── genomics.py            DNA classification → DMT1/DMT2/NONDM
│   ├── doctor.py              Conversational intake → Diabetic/Non-Diabetic
│   ├── health_trainer.py      Exercise prescription with clinical context
│   ├── transcriptomics.py     Gene expression pathway analysis + subtype + false positive filter
│   ├── proteomics.py          UniProt REST API (stub)
│   ├── pharmacology.py        DGIpy, OpenFDA (stub → TODO)
│   ├── clinical.py            Guidelines knowledge base (stub)
│   └── literature.py          PubMed, Semantic Scholar (stub)
├── tools/
│   ├── dna_classifier.py              Pre-trained CNN (3-mer, 84% accuracy)
│   ├── diabetes_classifier.py         Pre-trained MLP (Pima, 75% accuracy)
│   ├── workout_type_classifier.py     ADA 2023 clinical rules
│   ├── exercise_recommender.py        50-exercise CSV lookup
│   └── gene_expression_analyzer.py    GSE26168 z-score: gene profile → pathways/subtype/risks
├── prompts/                   System prompts as .txt (Ralph Loop edits these)
│   ├── genomics.txt
│   ├── doctor.txt
│   ├── health_trainer.txt
│   └── transcriptomics.txt
└── eval/
    ├── cases.py               4 test cases with ground truth
    ├── metrics.py             Layer 1 (tool accuracy) + Layer 3 (decision correctness)
    ├── judge.py               Layer 2 (LLM-as-judge, 4 dimensions × 1-5)
    ├── ralph.py               Ralph Loop prompt optimizer
    └── data/
        ├── case_inputs.json   DNA sequences + clinical features + HT vitals
        └── mock_outputs/      Pre-recorded AgentResult JSON per case per agent
```

## Tech Stack

- Python 3.12 + uv
- Anthropic Claude API (agent reasoning)
- asyncio (concurrency)
- Streamlit (dashboard)
