# Precision Health Agents

Multi-agent healthcare intelligence system — 9 specialized agents analyze patient data and synthesize personalized health assessments with 4-layer validation (DNA + clinical + transcriptomics + metabolomics).

## Project Structure

```
precision-health-agents/
├── src/precision_health_agents/           # Shared package (agents, tools, eval, orchestrator)
├── scripts/             # Entry points (run.py, evaluate.py, process_transcriptomics.py, process_metabolomics.py)
├── app/                 # Streamlit eval dashboard
├── tests/               # Tests (mirror src/ structure, 196 tests)
├── docs/                # Knowledge DB (architecture, vision, data, demo)
├── data/                # Datasets (gitignored)
├── mb_<dev>/            # Per-developer memory banks (gitignored)
```

## Development Model

- **Team**: JS (top-down: infra, orchestration, architecture), YH (bottom-up: data pipeline, tools, agents, testing)
- **Workflow**: Parallel worktrees per task, rebase-and-merge to main
- **Memory banks**: Each developer has `mb_js/` or `mb_yh/` (gitignored) for local task tracking
- **Merge protocol**: rebase → `git push origin <branch>:main` → tell teammate → they rebase

## Key References

- [docs/architecture.md](docs/architecture.md) — System design, agents, pipeline, model strategy
- [docs/eval.md](docs/eval.md) — Evaluation pipeline, test cases, results, Ralph Loop
- [docs/vision.md](docs/vision.md) — Project vision: 3-layer precision validation
- [docs/data.md](docs/data.md) — Datasets, patient cases, data format
- [docs/demo.md](docs/demo.md) — Demo plan, dashboard, priorities
- [docs/doctor_agent.md](docs/doctor_agent.md) — Doctor agent design, API, output models
- [docs/genomics.md](docs/genomics.md) — Genomics agent, CNN model, pipeline
- [docs/metabolomics.md](docs/metabolomics.md) — Metabolomics agent, ST001906, metabolic profiling
- [docs/hospital.md](docs/hospital.md) — Hospital agent, molecular test coordination

## Current Focus

**Publication quality** — clean architecture, comprehensive tests, reproducible results for journal article.

## Rules

- Python 3.12+, `uv` for deps
- Reusable logic in `src/precision_health_agents/`, scripts only orchestrate
- Claude API (`anthropic` SDK) for LLM calls; prompts in `src/precision_health_agents/prompts/*.txt`
- Type hints on all signatures, docstrings on public APIs
- TDD for new functions/classes: write test → implement → pass. Not required for scripts, config, or docs.
- Conventional commits: `type(scope): description`
- Branch naming: `type/description` (kebab-case)
- Never force push to main

## Commands

```bash
uv sync                                      # Install dependencies
uv run pytest                                # Run tests (196 tests)
uv run python scripts/run.py --case 1        # Run pipeline on a case
uv run python scripts/evaluate.py            # Real eval (agents + judge via API)
uv run python scripts/evaluate.py --mock     # Mock eval (pre-recorded outputs)
uv run python scripts/evaluate.py --ralph --iter 3  # Ralph Loop
uv run streamlit run app/dashboard.py        # Eval dashboard
```

## Memory Bank Setup

Memory banks are gitignored. After cloning, create yours:

```bash
mkdir -p mb_<you>/{active,hold,todo,completed}
# Seed progress.md from your task list
```
