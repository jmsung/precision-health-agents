# BioAI

Multi-agent healthcare intelligence system — 6 specialized agents analyze patient data and synthesize personalized health assessments.

## Project Structure

```
bioai/
├── src/bioai/           # Shared package (agents, tools, eval, orchestrator)
├── scripts/             # Entry points (run.py, evaluate.py, demo.py)
├── app/                 # Streamlit dashboard
├── tests/               # Tests (mirror src/ structure)
├── docs/                # Knowledge DB (architecture, vision, data, demo)
├── data/                # Datasets (gitignored)
├── mb_<dev>/            # Per-developer memory banks (gitignored)
```

## Development Model

- **Team**: JS (top-down: infra, orchestration, architecture), YH (bottom-up: data pipeline, tools, agents, testing)
- **Workflow**: Parallel worktrees per task, rebase-and-merge to main
- **Memory banks**: Each developer has `mb_js/` or `mb_yh/` (gitignored) for local task tracking
- **Merge protocol**: Coordinate verbally → rebase → `git merge --ff-only` → push → other rebases

## Key References

- [docs/architecture.md](docs/architecture.md) — System design, agents, eval, Ralph Loop, model strategy
- [docs/vision.md](docs/vision.md) — Project vision and capabilities
- [docs/data.md](docs/data.md) — Datasets, patient cases, data format
- [docs/demo.md](docs/demo.md) — Demo plan, dashboard, priorities

## Rules

- Python 3.12+, `uv` for deps
- Reusable logic in `src/bioai/`, scripts only orchestrate
- Claude API (`anthropic` SDK) for LLM calls; prompts in `src/bioai/prompts/*.txt`
- Type hints on all signatures, docstrings on public APIs
- Conventional commits: `type(scope): description`
- Branch naming: `type/description` (kebab-case)
- Never force push to main

## Commands

```bash
uv sync                  # Install dependencies
uv run pytest            # Run tests
uv run python scripts/   # Run scripts
```

## Memory Bank Setup

Memory banks are gitignored. After cloning, create yours:

```bash
mkdir -p mb_<you>/{active,hold,todo,completed}
# Seed progress.md from your task list
```
