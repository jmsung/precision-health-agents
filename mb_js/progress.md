# Progress — JS (Top-Down: Infra, Orchestration, Architecture)

## Active

## Hold

## Todo

### Phase 1: Setup & Models
- [ ] Extend `pyproject.toml` with all deps, `uv sync`
- [ ] Create `src/bioai/models.py` — Patient, AgentResult, TestCase dataclasses
- [ ] Create `src/bioai/blackboard.py` — shared state for agent communication
- [ ] Extend `src/bioai/config.py` — multi-model settings, prompts dir

### Phase 2: Core Framework
- [ ] Rewrite `src/bioai/agents/base.py` — BaseAgent with agentic tool-use loop
- [ ] Rewrite `src/bioai/orchestrator.py` — 2-phase (parallel agents → synthesis)
- [ ] Create `src/bioai/synthesis.py` — Claude-powered final report generation

### Phase 3: Evaluation Framework
- [ ] `eval/metrics.py` — automated scoring
- [ ] `eval/judge.py` — LLM-as-judge
- [ ] `eval/cases.py` — 3 test cases with ground truth
- [ ] `eval/ralph.py` — Ralph Loop prompt optimizer
- [ ] `scripts/evaluate.py` — CLI runner

### Phase 4: Integration & Polish
- [ ] End-to-end test: `scripts/run.py --case 1`
- [ ] Wire all 6 agents into orchestrator
- [ ] Bug fixes, error handling, polish

## Completed

### main — Project scaffolding
- [x] Initialize repo with uv + Python 3.12
- [x] BaseAgent ABC and six agent stubs
- [x] Orchestrator skeleton
- [x] Config, tests, entry point
