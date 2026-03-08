"""Transcriptomics agent for gene expression pathway analysis."""

from pathlib import Path

import anthropic

from bioai.agents.base import BaseAgent
from bioai.config import Settings
from bioai.models import AgentResult, AgentStatus, RiskLevel, TranscriptomicsFindings
from bioai.tools.gene_expression_analyzer import analyze_gene_expression

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "transcriptomics.txt"

_TOOL_DEF = {
    "name": "analyze_gene_expression",
    "description": (
        "Analyze a patient's gene expression profile for diabetes-related pathway activity. "
        "Takes a dictionary of gene symbols to expression values and returns pathway scores, "
        "dominant pathway, active pathways, risk level, and dysregulated genes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "gene_expression": {
                "type": "object",
                "description": (
                    "Dictionary mapping gene symbols to expression values. "
                    "Example: {\"TNF\": 1250.5, \"IL6\": 890.2, \"INS\": 45.0}"
                ),
                "additionalProperties": {"type": "number"},
            }
        },
        "required": ["gene_expression"],
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
        if parts:
            clinical_context = (
                "\n## Prior agent findings\n" + "\n".join(f"- {p}" for p in parts)
            )
    return template.replace("{clinical_context}", clinical_context)


class TranscriptomicsAgent(BaseAgent):
    name = "transcriptomics"
    role = "Gene expression signal analysis"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings.from_env()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)

    async def analyze(self, query: str, context: dict | None = None) -> AgentResult:
        """Analyze gene expression data for diabetes pathway activity."""
        messages = [{"role": "user", "content": query}]

        try:
            response = self._client.messages.create(
                model=self._settings.agent_model,
                max_tokens=self._settings.max_tokens,
                system=_load_prompt(context),
                tools=[_TOOL_DEF],
                messages=messages,
            )

            findings: TranscriptomicsFindings | None = None

            while response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                messages.append({"role": "assistant", "content": response.content})

                tool_result_content = []
                for call in tool_calls:
                    if call.name == "analyze_gene_expression":
                        raw = analyze_gene_expression(call.input["gene_expression"])
                        findings = TranscriptomicsFindings(
                            pathway_scores=raw["pathway_scores"],
                            dominant_pathway=raw["dominant_pathway"],
                            active_pathways=raw["active_pathways"],
                            risk_level=_RISK_MAP[raw["risk_level"]],
                            dysregulated_genes=raw["dysregulated_genes"],
                            diabetes_confirmed=raw["diabetes_confirmed"],
                            diabetes_subtype=raw["diabetes_subtype"],
                            complication_risks=raw["complication_risks"],
                            monitoring=raw["monitoring"],
                            recommendation=raw["recommendation"],
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
