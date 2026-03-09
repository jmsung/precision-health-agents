"""Ralph Loop — iterative prompt optimizer (v2).

Improvements over v1:
- Includes failure examples + judge explanations in rewrite prompt (P0)
- Saves backup prompt for rollback on regression (P0)
- Filters non-prompt-improvable metrics (tool_accuracy, decision) (P1)
- Accepts iteration history to prevent oscillation (P2)
"""

from pathlib import Path

import anthropic
from pydantic import BaseModel

from precision_health_agents.config import Settings

_REWRITE_SYSTEM = """\
You are an expert prompt engineer for biomedical AI agents.

Given an agent's current system prompt, its evaluation scores, and concrete examples
of where it failed, rewrite the prompt to improve the weakest metric.

Keep the prompt's core purpose and tool-calling instructions intact. Only change
phrasing, emphasis, or add guidance that addresses the specific weakness.

If previous iteration history is provided, avoid repeating changes that didn't help
or that caused regressions in other metrics.

Return ONLY the rewritten prompt text — no explanation, no markdown fences.
"""

# Metrics that are determined by ML model output, not by prompt quality.
_NON_PROMPT_METRICS = frozenset({"tool_accuracy", "decision"})


class FailureExample(BaseModel):
    """Concrete failure case for Ralph to learn from."""

    case_id: str
    agent_output: str
    judge_explanation: str


class RalphResult(BaseModel):
    """Result of a single Ralph Loop iteration."""

    agent: str
    metric: str
    old_score: float
    new_score: float = 0.0  # filled after re-eval
    prompt_changed: bool
    diff: str  # description of what changed
    backup_path: str | None = None  # path to backup prompt for rollback


def _find_weakest(
    eval_scores: dict[str, dict[str, float]],
) -> tuple[str, str, float]:
    """Find the agent + metric with the lowest prompt-improvable score.

    Skips tool_accuracy and decision metrics since those depend on ML models,
    not prompt quality.
    """
    worst_agent, worst_metric, worst_score = "", "", float("inf")
    for agent, scores in eval_scores.items():
        for metric, score in scores.items():
            if metric in _NON_PROMPT_METRICS:
                continue
            if score < worst_score:
                worst_agent, worst_metric, worst_score = agent, metric, score
    return worst_agent, worst_metric, worst_score


def _build_user_message(
    agent_name: str,
    metric_name: str,
    score: float,
    all_scores: dict[str, float],
    current_prompt: str,
    failure_context: list[FailureExample] | None = None,
    history: list[RalphResult] | None = None,
) -> str:
    """Build the user message for the rewrite prompt."""
    parts = [
        f"## Agent: {agent_name}",
        f"## Weakest metric: {metric_name} (score: {score})",
        f"## All scores: {all_scores}",
    ]

    if failure_context:
        parts.append("\n## Failure examples (concrete cases where the agent scored poorly):")
        for ex in failure_context:
            parts.append(
                f"\n### Case: {ex.case_id}\n"
                f"Agent output: {ex.agent_output}\n"
                f"Judge feedback: {ex.judge_explanation}"
            )

    if history:
        parts.append("\n## Previous iterations (avoid repeating failed approaches):")
        for h in history:
            parts.append(
                f"- Iteration targeting {h.agent}/{h.metric}: "
                f"score {h.old_score} → {h.new_score} "
                f"({'improved' if h.new_score > h.old_score else 'regressed or unchanged'})"
            )

    parts.append(f"\n## Current prompt:\n{current_prompt}")
    return "\n".join(parts)


async def ralph_iterate(
    eval_scores: dict[str, dict[str, float]],
    prompt_dir: Path | None = None,
    settings: Settings | None = None,
    failure_context: list[FailureExample] | None = None,
    history: list[RalphResult] | None = None,
) -> RalphResult:
    """Run one Ralph Loop iteration: find weakest → rewrite prompt.

    Args:
        eval_scores: Per-agent dict of metric scores.
        prompt_dir: Directory containing agent .txt prompt files.
        settings: App settings (API key, model names).
        failure_context: Concrete failure examples with judge explanations.
        history: Past iteration results to prevent oscillation.

    Returns:
        RalphResult with backup_path set if prompt was changed (for rollback).
    """
    settings = settings or Settings.from_env()
    prompt_dir = prompt_dir or settings.prompts_dir

    agent_name, metric_name, score = _find_weakest(eval_scores)
    prompt_file = prompt_dir / f"{agent_name}.txt"

    if not prompt_file.exists():
        return RalphResult(
            agent=agent_name,
            metric=metric_name,
            old_score=score,
            prompt_changed=False,
            diff=f"No prompt file found at {prompt_file}",
        )

    current_prompt = prompt_file.read_text()

    # Save backup for rollback
    backup_file = prompt_dir / f"{agent_name}.txt.bak"
    backup_file.write_text(current_prompt)

    try:
        client = anthropic.Anthropic(api_key=settings.api_key)
        user_msg = _build_user_message(
            agent_name=agent_name,
            metric_name=metric_name,
            score=score,
            all_scores=eval_scores[agent_name],
            current_prompt=current_prompt,
            failure_context=failure_context,
            history=history,
        )
        response = client.messages.create(
            model=settings.ralph_model,
            max_tokens=2048,
            system=_REWRITE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        new_prompt = response.content[0].text.strip()
        prompt_file.write_text(new_prompt + "\n")

        return RalphResult(
            agent=agent_name,
            metric=metric_name,
            old_score=score,
            prompt_changed=True,
            diff=f"Rewrote {prompt_file.name} to improve {metric_name}",
            backup_path=str(backup_file),
        )
    except Exception as e:
        # Restore from backup on failure
        prompt_file.write_text(current_prompt)
        backup_file.unlink(missing_ok=True)
        return RalphResult(
            agent=agent_name,
            metric=metric_name,
            old_score=score,
            prompt_changed=False,
            diff=f"Rewrite failed: {e}",
        )
