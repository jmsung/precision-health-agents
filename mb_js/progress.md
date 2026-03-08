# Progress — JS (Top-Down: Infra, Orchestration, Architecture)

## Active

## Hold

### Phase 4: Integration & Polish
- [ ] End-to-end test: `scripts/run.py --case 1`
- [ ] Wire all 6 agents into orchestrator
- [ ] Bug fixes, error handling, polish

## Todo

## Completed

### Phase 3: Evaluation Framework
- [x] `eval/cases.py` — 4 test cases with ground truth (EvalCase, ExpectedOutput)
- [x] `eval/metrics.py` — Layer 1 tool accuracy + Layer 3 decision correctness
- [x] `eval/judge.py` — Layer 2 LLM-as-judge (relevance, completeness, accuracy, safety, 1-5)
- [x] `eval/ralph.py` — Ralph Loop v1 (find weakest agent/metric → Claude Opus rewrites prompt)
- [x] `scripts/evaluate.py` — CLI runner (--mock, --save, --ralph --iter N)
- [x] `data/eval/case_inputs.json` — real DNA sequences + Pima clinical features + HT vitals
- [x] `app/dashboard.py` — Streamlit eval dashboard (Overview, Case Details, LLM-as-Judge tabs)
- [x] Health trainer integrated into eval (cases, metrics, evaluate.py, mock I/O)
- [x] Full eval: 13/13 deterministic pass (genomics, doctor, health_trainer, decision)
- [x] Ralph Loop 3 iterations — rewrote health_trainer.txt and doctor.txt
  - health_trainer: rel 2→5, comp 1→4, acc 3→5, safe 4→5
  - doctor: added verification step, systematic collection, comprehensive response
  - Note: doctor case-2/3 low judge scores expected (by design — no DNA context)
- [x] Docs updated: architecture.md (eval pipeline, Ralph Loop flow, project structure), demo.md (priorities), README.md (agents table)
- [x] 23/23 tests pass (metrics, cases, judge, ralph)

### Phase 2: Core Framework — **Reassigned to YH**
- YH owns: BaseAgent rewrite, orchestrator, synthesis, models, blackboard
- YH landed: genomics agent+tool, doctor agent+tool, diabetes classifier, health trainer agent+tools

### phase1-setup-models — Config rewrite
- [x] `config.py` → Pydantic BaseModel with multi-model fields, `from_env()`, paths
- [x] Test coverage for config (`tests/test_config.py`)

### main — Project scaffolding
- [x] Initialize repo with uv + Python 3.12
- [x] BaseAgent ABC and six agent stubs
- [x] Orchestrator skeleton
- [x] Config, tests, entry point
