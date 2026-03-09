"""Metabolomics agent for metabolic profile analysis."""

from pathlib import Path

import anthropic

from precision_health_agents.agents.base import BaseAgent
from precision_health_agents.config import Settings
from precision_health_agents.models import AgentResult, AgentStatus, MetabolomicsFindings, RiskLevel
from precision_health_agents.tools.metabolic_profile_analyzer import analyze_metabolic_profile

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "metabolomics.txt"

_TOOL_DEF = {
    "name": "analyze_metabolic_profile",
    "description": (
        "Analyze metabolic profile for diabetes-related metabolic state. "
        "Takes a dictionary of metabolite names to concentration values and returns "
        "metabolite scores, insulin resistance score, metabolic pattern, risk level, "
        "subtype refinement, and diabetes confirmation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metabolite_levels": {
                "type": "object",
                "description": (
                    "Dictionary mapping metabolite names to concentration values. "
                    'Example: {"triglycerides": 250.0, "leucine": 180.5, "HDL": 35.0}'
                ),
                "additionalProperties": {"type": "number"},
            }
        },
        "required": ["metabolite_levels"],
    },
}

_RISK_MAP = {
    "high": RiskLevel.HIGH,
    "moderate": RiskLevel.MODERATE,
    "low": RiskLevel.LOW,
}


def _load_prompt(context: dict | None = None) -> str:
    template = _PROMPT_PATH.read_text().strip()
    clinical_context = ""
    if context:
        parts = []
        if "genomics" in context:
            g = context["genomics"]
            parts.append(
                f"Genomics: {g.get('predicted_class', 'unknown')} "
                f"(confidence: {g.get('confidence', 'N/A')})"
            )
        if "doctor" in context:
            d = context["doctor"]
            parts.append(
                f"Doctor: {d.get('prediction', 'unknown')} "
                f"(probability: {d.get('probability', 'N/A')})"
            )
        if "transcriptomics" in context:
            t = context["transcriptomics"]
            parts.append(
                f"Transcriptomics: {t.get('dominant_pathway', 'unknown')} "
                f"(risk: {t.get('risk_level', 'N/A')})"
            )
        if parts:
            clinical_context = (
                "\n## Prior agent findings\n" + "\n".join(f"- {p}" for p in parts)
            )
    return template.replace("{clinical_context}", clinical_context)


class MetabolomicsAgent(BaseAgent):
    name = "metabolomics"
    role = "Metabolic profile analysis for diabetes metabolic state assessment"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings.from_env()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)

    async def analyze(self, query: str, context: dict | None = None) -> AgentResult:
        """Analyze metabolic profile for diabetes-related metabolic state."""
        messages = [{"role": "user", "content": query}]

        try:
            response = self._client.messages.create(
                model=self._settings.agent_model,
                max_tokens=self._settings.max_tokens,
                system=_load_prompt(context),
                tools=[_TOOL_DEF],
                messages=messages,
            )

            findings: MetabolomicsFindings | None = None

            while response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                messages.append({"role": "assistant", "content": response.content})

                tool_result_content = []
                for call in tool_calls:
                    if call.name == "analyze_metabolic_profile":
                        raw = analyze_metabolic_profile(call.input["metabolite_levels"])
                        findings = MetabolomicsFindings(
                            metabolite_scores=raw["metabolite_scores"],
                            elevated_metabolites=raw["elevated_metabolites"],
                            insulin_resistance_score=raw["insulin_resistance_score"],
                            metabolic_pattern=raw["metabolic_pattern"],
                            risk_level=_RISK_MAP[raw["risk_level"]],
                            subtype_refinement=raw["subtype_refinement"],
                            diabetes_confirmed=raw["diabetes_confirmed"],
                            interpretation=raw["interpretation"],
                        )
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": str(raw),
                        })

                messages.append({"role": "user", "content": tool_result_content})
                response = self._client.messages.create(
                    model=self._settings.agent_model,
                    max_tokens=1024,
                    system=_load_prompt(context),
                    tools=[_TOOL_DEF],
                    messages=messages,
                )

            summary = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )

            return AgentResult(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                findings=findings,
                summary=summary,
            )

        except Exception as e:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.ERROR,
                summary="",
                error=str(e),
            )
