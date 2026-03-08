# Tasks — JS

## Track: Infrastructure + Omics Agents

### Phase 1: Setup (10:00-10:30)
- [ ] Extend `pyproject.toml` with all deps, `uv sync`
- [ ] Create `src/bioai/models.py` — Patient, AgentResult, TestCase dataclasses
- [ ] Create `src/bioai/blackboard.py` — shared state for agent communication

### Phase 2: Core Framework (10:30-11:30)
- [ ] Rewrite `src/bioai/agents/base.py` — BaseAgent with agentic tool-use loop
- [ ] Rewrite `src/bioai/orchestrator.py` — 2-phase (parallel agents → synthesis)
- [ ] Extend `src/bioai/config.py` — multi-model settings, prompts dir

### Phase 3: Omics Agents (11:30-2:30)
- [ ] **Genomics** agent + `tools/genomics_tools.py` (myvariant.info, ClinVar Entrez)
- [ ] **Transcriptomics** agent + `tools/transcriptomics_tools.py` (GSEApy)
- [ ] **Proteomics** agent + `tools/proteomics_tools.py` (UniProt REST API)
- [ ] System prompts: `prompts/genomics.txt`, `transcriptomics.txt`, `proteomics.txt`

### Phase 4: Evaluation (2:30-4:30)
- [ ] `eval/metrics.py` — automated scoring (variant accuracy, citation count, latency, cost)
- [ ] `eval/judge.py` — LLM-as-judge (relevance, completeness, safety)
- [ ] `eval/cases.py` — 3 test cases with ground truth
- [ ] `eval/ralph.py` — Ralph Loop (find weakest agent → rewrite prompt → re-eval)
- [ ] `scripts/evaluate.py` — CLI runner for eval + Ralph Loop

### Phase 5: Integration (4:30-6:15)
- [ ] End-to-end test: `scripts/run.py --case 1`
- [ ] Wire all 6 agents into orchestrator (coordinate with YH)
- [ ] Bug fixes, error handling, polish

### Sync Points
- **10:30** — Share BaseAgent interface + AgentResult schema with YH
- **12:30** — First 2-agent test (genomics + pharmacology)
- **2:30** — All 6 agents compile, first full pipeline
- **4:30** — Full demo with eval

### Model Config
- Agent analysis: `claude-sonnet-4-6` (fast + cheap)
- Synthesis: `claude-opus-4-6` (best quality)
- LLM-as-judge: `claude-sonnet-4-6`
- Ralph Loop: `claude-opus-4-6`
