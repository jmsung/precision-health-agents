# Progress — JS (Top-Down: Infra, Orchestration, Architecture)

## Active

### Phase 5: Publication Quality

**Focus**: Clean architecture, comprehensive tests, reproducible results for journal article.

#### Rename: bioai → precision-health-agents
- [x] `src/bioai/` → `src/precision_health_agents/` (git mv)
- [x] 202 import/patch replacements across 45 .py files
- [x] pyproject.toml: name → `precision-health-agents`, packages path updated
- [x] All docs updated (CLAUDE.md, README.md, 10+ doc files)
- [x] 200/200 tests pass

#### E2E pipeline & eval (in progress)
- [ ] Run E2E test across all cases
- [ ] Run Ralph Loop for prompt quality
- [ ] Publication-ready eval results

## Hold

#### Latency & cost tracking (deferred)
- [ ] Track wall-clock time per agent, API token usage, report + dashboard

#### Agent-aware judge context
- [ ] Tell judge what each agent can/can't see (doctor doesn't see DNA results)

## Completed

### Phase 4: Demo (Hackathon — completed 2025-03-08)
- [x] `scripts/demo_conversation.py` — step-by-step case 1 walkthrough
- [x] `scripts/run.py --case N` — E2E CLI pipeline (4 cases, mock + live)
- [x] Transcriptomics as 3rd validation layer in eval
- [x] Ralph Loop v2 (failure examples, rollback, prompt-improvable filter, history)
- [x] Streamlit eval dashboard

### Phase 3: Evaluation Framework
- [x] eval/cases.py, metrics.py, judge.py, ralph.py
- [x] scripts/evaluate.py CLI (--mock, --save, --ralph --iter N)
- [x] app/dashboard.py — Streamlit eval dashboard
- [x] Health trainer in eval, 13/13 deterministic pass
- [x] Ralph Loop 3 iterations — rewrote health_trainer.txt and doctor.txt

### Phase 2: Core Framework (YH)
- [x] BaseAgent, orchestrator, models, genomics/doctor/health_trainer agents+tools

### Phase 1: Setup
- [x] Config, scaffolding, uv + Python 3.12
