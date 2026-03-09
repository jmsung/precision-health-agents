# Architecture

## Core Pipeline: DNA-Precision Diabetes Decision

The primary use case is a multi-stage pipeline with **multi-omics validation**: DNA assessment first, then clinical assessment, then molecular confirmation via transcriptomics, proteomics, and metabolomics. The system uses multiple independent evidence sources across the biological spectrum — from stable inherited risk to dynamic metabolic state — to confirm or reject a diabetes diagnosis and inform treatment.

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
   |  +--- [Hospital Agent] — coordinates molecular tests
   |  |         |
   |  |    Explains tests to patient, gets consent
   |  |         |
   |  |    run_hospital_tests(consent, gene_expression, metabolite_levels)
   |  |         |
   |  |    +----+----+  (runs both in parallel)
   |  |    |         |
   |  |  [Transcriptomics]        [Metabolomics]
   |  |  analyze_gene_expression   analyze_metabolic_profile
   |  |  (110-gene panel,          (78 metabolites, 5 pathways:
   |  |   5 pathways: beta cell,    amino acid, carbohydrate,
   |  |   inflammation, insulin     lipid, TCA/energy,
   |  |   resistance, fibrosis,     ketone/oxidative)
   |  |   oxidative)               -> insulin resistance score
   |  |  -> subtype, risks         -> metabolic pattern
   |  |    |         |
   |  |    +----+----+
   |  |         |
   |  |    Combined decision:
   |  |    Both confirm = high confidence
   |  |    One confirms = moderate confidence
   |  |    Neither confirms = false positive
   |  |         |
   |  |    +----+----+
   |  |  Confirmed     NOT confirmed
   |  |    |                |
   |  +--- | --> [Pharmacology Agent]   Health Trainer
   |       |          |                (false positive --
   |       |          |                 no drugs needed)
   |       |    recommend_medications (16-drug ADA guideline DB)
   |       |    -> Score by subtype match + complication benefit
   |       |    -> Contraindication filtering
   |       |    -> Personalized medication plan
   |       |    -> Monitoring schedule
   |       |
   |  +--- (HOSPITAL, parallel) --> [Proteomics Agent]
   |  |                                    |
   |  |                    analyze_protein_biomarkers (inflammatory, signaling,
   |  |                       kidney/CV injury markers)
   |  |                    -> Functional biomarker confirmation
   |  |                    -> Complication risk from protein-level evidence
   |  |
   +--- (HEALTH_TRAINER) --> [Health Trainer Agent]
   |                              |
   |                    classify_workout_type (ADA 2023 rules)
   |                    + diabetes context from Genomics & Doctor
   |                              |
   |                    recommend_exercises (50-exercise DB)
   |                              |
   |                    -> Personalised weekly plan
   |
   +--- (Background) --> [Clinical, Literature]
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

### Multi-Omics Stack

The omics agents form a spectrum from stable inherited risk to dynamic metabolic state:

```
Stable <-----------------------------------------------------> Dynamic

Genomics       Transcriptomics       Proteomics       Metabolomics
(inherited     (pathway activity,     (functional      (current metabolic
 risk, MODY,    inflammation,          biomarkers,      state, insulin
 pharmaco-      beta cell stress,      inflammatory/    resistance, lipid
 genomic        insulin resistance)    signaling        dysregulation,
 markers)                              proteins,        BCAA/acylcarnitine
                                       kidney/CV        patterns)
                                       injury markers)
```

### Agent Table

| Agent | Tools | Backend | Role in Pipeline |
|-------|-------|---------|-----------------|
| **Genomics** | `classify_dna` | Pre-trained 2-layer CNN (3-mer tokenization) | Inherited risk: DMT1/DMT2/NONDM |
| **Doctor** | `classify_diabetes` | Pre-trained MLP (8 clinical features) | Conversational intake → hospital/health trainer routing |
| **Health Trainer** | `classify_workout_type`, `recommend_exercises` | ADA 2023 clinical rules + 50-exercise DB | Exercise prescription for HEALTH_TRAINER referrals |
| **Hospital** | `run_hospital_tests` | Coordinates transcriptomics + metabolomics, combines confirmation | Patient consent → blood tests → combined molecular decision |
| **Transcriptomics** | `analyze_gene_expression` | GSE26168 reference + z-score pathway scoring (110 genes, 5 pathways) | 3rd validation layer: confirms/rejects diabetes, subtype, complication risks |
| **Proteomics** | `analyze_protein_biomarkers` | TBD (YH in progress) | Functional biomarkers: inflammatory/signaling proteins, kidney/CV injury markers |
| **Metabolomics** | `analyze_metabolic_profile` | ST001906 reference + z-score pathway scoring (78 metabolites, 5 pathways) | Current metabolic state: insulin resistance, lipid dysregulation, BCAA patterns |
| **Pharmacology** | `recommend_medications` | ADA guideline DB (16 drugs × 8 classes) + scoring engine | Subtype-informed medication selection with complication-aware ranking |
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

### Hospital Agent: Molecular Test Coordination

When a patient is routed to Hospital, the Hospital Agent manages the patient interaction and coordinates molecular tests:

```
Patient referred to hospital (DNA + Clinical both positive)
        |
  HospitalAgent.chat()
        |
  Explains need for blood tests (transcriptomics + metabolomics)
        |
  Patient consents? --NO--> health_trainer (can't confirm without tests)
        |
       YES
        |
  run_hospital_tests(consent, gene_expression, metabolite_levels)
        |
  +-----+-----+  (runs both analyses)
  |           |
  analyze_gene_expression    analyze_metabolic_profile
  (transcriptomics)          (metabolomics)
  |           |
  +-----+-----+
        |
  Combined decision:
    Both confirm    -> diabetes_confirmed=True,  confidence="high"
    One confirms    -> diabetes_confirmed=True,  confidence="moderate"
    Neither confirms -> diabetes_confirmed=False (false positive)
        |
  HospitalFindings -> Pharmacology (confirmed) or HealthTrainer (false positive)
```

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

### Pharmacology Agent: Medication Recommendation

When transcriptomics confirms diabetes and identifies a molecular subtype, the Pharmacology Agent selects medications:

```
TranscriptomicsFindings (subtype + complication risks)
        |
  PharmacologyAgent.chat()
        |
  Claude (tool-use loop)
        |  calls
  recommend_medications(subtype, complications)
        |
  Score 16 ADA guideline drugs:
    +3 subtype match, +2 ADA first-line, +2 complication benefit,
    +1.5 severe complication boost, contraindication exclusion
        |
  Claude composes personalized medication plan
        |
  PharmacologyFindings (primary + supportive meds, monitoring plan)
```

**Drug classes**: Biguanide (Metformin), SGLT2 Inhibitors, GLP-1 RAs, Basal/Rapid Insulin, TZDs, DPP-4 Inhibitors, Statins, ACE Inhibitors, Neuropathic Pain Agents.

| Subtype | Primary Drug Strategy |
|---------|----------------------|
| `inflammation_dominant` | GLP-1 RAs (anti-inflammatory + glucose) |
| `beta_cell_failure` | Insulin therapy (basal ± rapid) |
| `metabolic_insulin_resistant` | Metformin + SGLT2i |
| `fibrotic_complication` | SGLT2i + ACEi (organ protection) |
| `mixed` | DPP-4i bridge + combination approach |

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

3-layer evaluation (tool accuracy, LLM-as-judge, decision correctness) + Ralph Loop for automated prompt optimization. See **[docs/eval.md](eval.md)** for full details, test cases, results, and Ralph Loop data flow.

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
src/precision_health_agents/
├── config.py                  Pydantic settings (models, API keys, paths)
├── models.py                  AgentResult, GenomicsFindings, DoctorFindings, TranscriptomicsFindings, ProteomicsFindings, MetabolomicsFindings, HospitalFindings, PharmacologyFindings, HealthTrainerFindings
├── blackboard.py              Shared state for agent communication
├── orchestrator.py            2-phase: parallel agents → synthesis
├── agents/
│   ├── base.py                BaseAgent ABC
│   ├── genomics.py            DNA classification → DMT1/DMT2/NONDM
│   ├── doctor.py              Conversational intake → Diabetic/Non-Diabetic
│   ├── health_trainer.py      Exercise prescription with clinical context
│   ├── transcriptomics.py     Gene expression pathway analysis + subtype + false positive filter
│   ├── pharmacology.py        ADA guideline medication recommendation
│   ├── proteomics.py          Protein biomarker analysis (tool-use loop, scaffold for YH)
│   ├── hospital.py            Molecular test coordination (consent → trans+metab → decision)
│   ├── metabolomics.py        Metabolic profile analysis (tool-use loop)
│   ├── clinical.py            Guidelines knowledge base (stub)
│   └── literature.py          PubMed, Semantic Scholar (stub)
├── tools/
│   ├── dna_classifier.py              Pre-trained CNN (3-mer, 84% accuracy)
│   ├── diabetes_classifier.py         Pre-trained MLP (Pima, 75% accuracy)
│   ├── workout_type_classifier.py     ADA 2023 clinical rules
│   ├── exercise_recommender.py        50-exercise CSV lookup
│   ├── gene_expression_analyzer.py    GSE26168 z-score: gene profile → pathways/subtype/risks
│   ├── protein_biomarker_analyzer.py  Protein biomarker analysis (stub for YH)
│   ├── metabolic_profile_analyzer.py  ST001906 z-score: metabolite profile → pathways/IR score/pattern
│   └── drug_recommender.py            ADA guideline medication scoring + recommendation
├── prompts/                   System prompts as .txt (Ralph Loop edits these)
│   ├── genomics.txt
│   ├── doctor.txt
│   ├── health_trainer.txt
│   ├── hospital.txt
│   ├── transcriptomics.txt
│   ├── proteomics.txt
│   ├── metabolomics.txt
│   └── pharmacology.txt
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
