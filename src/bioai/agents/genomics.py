"""Genomics agent for variant interpretation and DNA risk classification."""

import anthropic

from bioai.agents.base import BaseAgent
from bioai.config import Settings
from bioai.models import AgentResult, AgentStatus, GenomicsFindings, RiskLevel
from bioai.tools.dna_classifier import classify_dna

_TOOL_DEF = {
    "name": "classify_dna",
    "description": (
        "Classify a DNA sequence for diabetes-associated genomic risk. "
        "Returns predicted class (DMT1, DMT2, or NONDM) with confidence scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sequence": {
                "type": "string",
                "description": "Raw DNA sequence string (A/T/G/C characters).",
            }
        },
        "required": ["sequence"],
    },
}

_SYSTEM_PROMPT = (
    "You are a genomics specialist agent. "
    "When given a DNA sequence, use the classify_dna tool to assess diabetes-related genomic risk. "
    "Interpret the result and provide a concise clinical summary."
)

_RISK_MAP = {
    "DMT1": RiskLevel.HIGH,
    "DMT2": RiskLevel.HIGH,
    "NONDM": RiskLevel.LOW,
}


class GenomicsAgent(BaseAgent):
    name = "genomics"
    role = "Variant interpretation and genomic risk analysis"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)

    async def analyze(self, query: str, context: dict | None = None) -> AgentResult:
        """Analyze a genomics query, calling the DNA classifier tool if a sequence is present."""
        messages = [{"role": "user", "content": query}]

        try:
            response = self._client.messages.create(
                model=self._settings.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=[_TOOL_DEF],
                messages=messages,
            )

            findings: GenomicsFindings | None = None

            while response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                messages.append({"role": "assistant", "content": response.content})

                tool_result_content = []
                for call in tool_calls:
                    if call.name == "classify_dna":
                        raw = classify_dna(call.input["sequence"])
                        findings = GenomicsFindings(
                            predicted_class=raw["predicted_class"],
                            confidence=raw["confidence"],
                            probabilities=raw["probabilities"],
                            risk_level=_RISK_MAP[raw["predicted_class"]],
                            interpretation=_interpret(raw),
                        )
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": str(raw),
                        })

                messages.append({"role": "user", "content": tool_result_content})
                response = self._client.messages.create(
                    model=self._settings.model,
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
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


def _interpret(raw: dict) -> str:
    cls = raw["predicted_class"]
    conf = raw["confidence"]
    if cls == "DMT2":
        return f"Sequence shows Type 2 Diabetes-associated pattern with {conf:.0%} confidence."
    if cls == "DMT1":
        return f"Sequence shows Type 1 Diabetes-associated pattern with {conf:.0%} confidence."
    return f"No diabetes-associated pattern detected ({conf:.0%} confidence)."
