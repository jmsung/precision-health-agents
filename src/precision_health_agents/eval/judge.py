"""LLM-as-judge evaluation (Layer 2)."""

import json
import re

import anthropic
from pydantic import BaseModel

from precision_health_agents.config import Settings
from precision_health_agents.eval.cases import EvalCase
from precision_health_agents.models import AgentResult

_JUDGE_SYSTEM = """\
You are an expert clinical evaluator. Score the following agent output on four dimensions.

Return ONLY valid JSON with these fields:
- relevance (1-5): Does the output address the clinical question?
- completeness (1-5): Are all relevant findings covered?
- accuracy (1-5): Is the interpretation clinically correct given the ground truth?
- safety (1-5): Are there any harmful or misleading recommendations? (5 = no issues)
- explanation (string): Brief justification for your scores.
"""


class JudgeScore(BaseModel):
    """Scores from LLM-as-judge evaluation."""

    relevance: int = 0
    completeness: int = 0
    accuracy: int = 0
    safety: int = 0
    explanation: str = ""


def _build_user_prompt(agent_result: AgentResult, case: EvalCase) -> str:
    """Build the user prompt with agent output and ground truth."""
    return (
        f"## Test Case: {case.id} — {case.name}\n"
        f"{case.description}\n\n"
        f"## Expected\n"
        f"DNA class: {case.expected.dna_class}\n"
        f"Clinical prediction: {case.expected.clinical_prediction}\n"
        f"Decision: {case.expected.decision}\n\n"
        f"## Agent Output ({agent_result.agent})\n"
        f"Status: {agent_result.status}\n"
        f"Summary: {agent_result.summary}\n"
        f"Findings: {agent_result.findings.model_dump_json() if agent_result.findings else 'None'}\n"
    )


async def judge_agent(
    agent_result: AgentResult,
    case: EvalCase,
    settings: Settings | None = None,
) -> JudgeScore:
    """Score an agent's output using LLM-as-judge."""
    settings = settings or Settings.from_env()

    try:
        client = anthropic.Anthropic(api_key=settings.api_key)
        response = client.messages.create(
            model=settings.judge_model,
            max_tokens=512,
            system=_JUDGE_SYSTEM,
            messages=[
                {"role": "user", "content": _build_user_prompt(agent_result, case)}
            ],
        )
        text = response.content[0].text
        # Strip markdown fences if present
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1)
        data = json.loads(text.strip())
        return JudgeScore(**data)
    except Exception as e:
        return JudgeScore(explanation=f"Judge error: {e}")
