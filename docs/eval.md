# Evaluation Pipeline

## Overview

Three-layer evaluation with 4 test cases covering all branches of the decision matrix. Agents evaluated: Genomics, Doctor, Health Trainer (case-4 only).

## Test Cases

| Case | DNA Input | Clinical Input | HT Vitals | Expected Decision |
|------|-----------|----------------|-----------|-------------------|
| case-1 | DMT2 (800bp) | diabetic (glucose=189, bmi=30.1, age=59) | — | hospital |
| case-2 | DMT2 (same) | non_diabetic (glucose=89, bmi=28.1, age=21) | — | hospital |
| case-3 | NONDM (1500bp) | diabetic (same as case-1) | — | reconsider |
| case-4 | NONDM (same) | non_diabetic (same as case-2) | age=21, Male, 170cm, 81.2kg, freq=0, dur=0 | health_trainer |

DNA sequences are real from the DNA classification dataset. Clinical features are real Pima Indians Diabetes rows. All stored in `src/precision_health_agents/eval/data/case_inputs.json`.

## Evaluation Layers

**Layer 1 — Tool Accuracy (deterministic, no LLM)**
Checks whether the underlying ML/rule tool returned the correct prediction:
- Genomics: `classify_dna(sequence)` → `predicted_class == expected_dna_class`?
- Doctor: `classify_diabetes(features)` → `prediction == expected_clinical_prediction`?
- Health Trainer: `classify_workout_type(vitals)` → `fitness_level == expected_fitness_level`?
- Transcriptomics: `analyze_gene_expression(profile)` → expected subtype/pathway activation

Binary: 1.0 (correct) or 0.0 (wrong).

**Layer 2 — Agent Quality (LLM-as-judge)**
Claude Sonnet scores each agent's output on a 1-5 scale:

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

## Current Results (13/13 deterministic pass)

| Case | Genomics Tool | Doctor Tool | HT Tool | Decision | Judge (genomics) | Judge (doctor) | Judge (HT) |
|------|--------------|-------------|---------|----------|-----------------|----------------|------------|
| case-1 | PASS | PASS | — | PASS | 5/4/5/5 | 5/4/5/5 | — |
| case-2 | PASS | PASS | — | PASS | 5/4/5/4 | 2/1/1/1* | — |
| case-3 | PASS | PASS | — | PASS | 4/3/5/3 | 2/1/2/2* | — |
| case-4 | PASS | PASS | PASS | PASS | 5/5/5/5 | 5/4/5/5 | 5/4/5/5 |

*Doctor case-2/3 judge scores are low by design — the doctor agent doesn't see DNA results. The decision matrix (Layer 3) handles the cross-agent logic correctly.

## Execution Modes

```bash
uv run python scripts/evaluate.py              # real mode (agents + judge via Claude API)
uv run python scripts/evaluate.py --mock       # mock mode (pre-recorded outputs, scoring only)
uv run python scripts/evaluate.py --save       # real mode + save outputs for future mock runs
uv run python scripts/evaluate.py --ralph --iter 3  # Ralph Loop (3 prompt optimization iterations)
```

- **Real mode**: ~$0.15/run, ~30s. Runs agents + judge via Claude API.
- **Mock mode**: Free, instant. Uses saved JSON from `src/precision_health_agents/eval/data/mock_outputs/`.
- **Workflow**: real `--save` → iterate on eval logic in mock → Ralph Loop → real again.

## Ralph Loop (Iterative Prompt Optimization)

Automated prompt improvement: evaluate → find weakest agent/metric → rewrite prompt → re-evaluate.

```
evaluate_case() × 4 cases
    → Layer 1-3 scoring
           ↓
collect_judge_averages()
    → Per-agent averages: {genomics: {rel: 5.0, ...}, doctor: {...}, ...}
           ↓
_find_weakest()
    → Lowest prompt-improvable (agent, metric) pair
    → Skips tool_accuracy and decision (not prompt-fixable)
           ↓
ralph_iterate()
    → Read prompts/{agent}.txt
    → Claude Opus rewrites prompt with:
      - All scores + weakest metric
      - Failure examples + judge explanations (v2)
      - Past iteration history to prevent oscillation (v2)
    → Save backup (.bak) for rollback on regression (v2)
           ↓
Re-run full evaluation → next iteration
```

### Results (3 iterations)

| Iter | Target | Score | Effect |
|------|--------|-------|--------|
| 1 | health_trainer/completeness | 1.0 | rel 2→4, comp 1→3, acc 3→4, safe 4→5 |
| 2 | doctor/completeness | 2.5 | Added verification step, systematic checklist |
| 3 | doctor/completeness | 2.8 | Further persistence + response structure |

## Health Trainer Evaluation

Separate 3-layer evaluation against the gym members dataset (973 rows):

```bash
uv run python scripts/evaluate_health_trainer.py
```

| Layer | What it tests | Ground truth | Key metric |
|---|---|---|---|
| **1. Experience level** | Threshold calibration | `Experience_Level` (1/2/3) | 48.2% accuracy |
| **2. Workout type baseline** | Demographic scoring (NONDM, prob=0.0) | `Workout_Type` | 19.7% accuracy |
| **3. Clinical constraints** | Safety rules with synthetic diabetes overlay | Assertions | **0 violations** |

**Experience level (48.2%)**: Conservative by design — diabetes patients are likely less fit than gym regulars. We under-classify (safe direction), never over-classify.

**Workout type (19.7%)**: Expected — gym data shows no demographic predictor for workout type. Confirms clinical rules (not demographics) should drive type selection.

**Clinical constraints (0 violations)**: The most important metric. DMT1 never gets HIIT, DMT2 always has Strength/Cardio in top 2, high-risk beginners never get HIIT.
