"""Hospital agent — coordinates molecular tests and makes final diabetes decision.

Flow:
  1. Explains to patient that blood tests are needed for molecular confirmation.
  2. Gets patient consent.
  3. Runs transcriptomics + metabolomics analyses (via run_hospital_tests tool).
  4. Combines results → confirmed diabetes → pharmacology, or false positive → health trainer.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from precision_health_agents.config import Settings
from precision_health_agents.models import (
    AgentResult,
    AgentStatus,
    HospitalFindings,
    HospitalRecommendation,
)
from precision_health_agents.tools.gene_expression_analyzer import analyze_gene_expression
from precision_health_agents.tools.metabolic_profile_analyzer import analyze_metabolic_profile

_PROMPTS_DIR = Path(__file__).parents[1] / "prompts"

_TOOL_DEF = {
    "name": "run_hospital_tests",
    "description": (
        "Run transcriptomics (gene expression) and metabolomics (metabolic profile) "
        "tests on the patient's blood samples. Call this after the patient consents "
        "to blood tests. Returns combined molecular confirmation results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "consent": {
                "type": "boolean",
                "description": "Whether the patient has consented to blood tests.",
            },
            "gene_expression": {
                "type": "object",
                "description": "Gene expression values from transcriptomics panel.",
                "additionalProperties": {"type": "number"},
            },
            "metabolite_levels": {
                "type": "object",
                "description": "Metabolite concentration values from metabolomics panel.",
                "additionalProperties": {"type": "number"},
            },
        },
        "required": ["consent", "gene_expression", "metabolite_levels"],
    },
}


def _load_prompt(context: dict | None = None) -> str:
    template = (_PROMPTS_DIR / "hospital.txt").read_text().strip()
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


def run_hospital_tests(
    consent: bool,
    gene_expression: dict[str, float],
    metabolite_levels: dict[str, float],
) -> dict:
    """Run transcriptomics + metabolomics and combine results.

    This is the tool function called by the HospitalAgent. It runs both
    molecular analyses and produces a combined diabetes confirmation decision.
    """
    if not consent:
        return {
            "patient_consented": False,
            "diabetes_confirmed": False,
            "confidence": "low",
            "recommendation": "health_trainer",
            "reasoning": "Patient declined blood tests. Cannot confirm diabetes molecularly.",
            "transcriptomics": {},
            "metabolomics": {},
        }

    # Run both analyses
    trans_result = analyze_gene_expression(gene_expression)
    metab_result = analyze_metabolic_profile(metabolite_levels)

    trans_confirmed = trans_result["diabetes_confirmed"]["confirmed"]
    metab_confirmed = metab_result["diabetes_confirmed"]["confirmed"]

    # Combined decision: either layer confirming is sufficient,
    # but both confirming gives high confidence
    if trans_confirmed and metab_confirmed:
        confirmed = True
        confidence = "high"
        reasoning = (
            "Both transcriptomics and metabolomics confirm active diabetes. "
            f"Transcriptomics: {trans_result['diabetes_confirmed']['reasoning']} "
            f"Metabolomics: {metab_result['diabetes_confirmed']['reasoning']}"
        )
    elif trans_confirmed or metab_confirmed:
        confirmed = True
        confidence = "moderate"
        confirming = "transcriptomics" if trans_confirmed else "metabolomics"
        reasoning = (
            f"Diabetes confirmed by {confirming}. "
            f"Transcriptomics confirmed: {trans_confirmed}. "
            f"Metabolomics confirmed: {metab_confirmed}."
        )
    else:
        confirmed = False
        confidence = "high"
        reasoning = (
            "Neither transcriptomics nor metabolomics shows active diabetes. "
            "Likely false positive from DNA/clinical assessment. "
            "Recommend lifestyle management instead of medication."
        )

    recommendation = "pharmacology" if confirmed else "health_trainer"

    return {
        "patient_consented": True,
        "diabetes_confirmed": confirmed,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "transcriptomics": {
            "confirmed": trans_confirmed,
            "subtype": trans_result.get("diabetes_subtype", {}),
            "active_pathways": trans_result.get("active_pathways", []),
            "risk_level": trans_result.get("risk_level", "low"),
            "complication_risks": trans_result.get("complication_risks", []),
        },
        "metabolomics": {
            "confirmed": metab_confirmed,
            "pattern": metab_result.get("metabolic_pattern", "normal"),
            "insulin_resistance_score": metab_result.get("insulin_resistance_score", 0.0),
            "risk_level": metab_result.get("risk_level", "low"),
            "subtype_refinement": metab_result.get("subtype_refinement", {}),
        },
    }


class HospitalAgent:
    """Conversational hospital agent that coordinates molecular tests."""

    name = "hospital"
    role = "Molecular test coordination and final diabetes decision"

    def __init__(self, settings: Settings | None = None, context: dict | None = None):
        self._settings = settings or Settings.from_env()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)
        self._context = context
        self._system = _load_prompt(context)
        self._messages: list[dict] = []
        self._findings: HospitalFindings | None = None

    def chat(self, patient_message: str) -> str:
        """Send a patient message and get the hospital specialist's reply."""
        self._messages.append({"role": "user", "content": patient_message})
        return self._run()

    @property
    def findings(self) -> HospitalFindings | None:
        return self._findings

    def result(self) -> AgentResult:
        """Build final AgentResult."""
        last_reply = ""
        for msg in reversed(self._messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], str):
                last_reply = msg["content"]
                break

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS if self._findings else AgentStatus.ERROR,
            findings=self._findings,
            summary=last_reply,
            error=None if self._findings else "Tests not completed.",
        )

    def _run(self) -> str:
        """Drive the Claude loop until a text response is ready."""
        while True:
            response = self._client.messages.create(
                model=self._settings.agent_model,
                max_tokens=self._settings.max_tokens,
                system=self._system,
                tools=[_TOOL_DEF],
                messages=self._messages,
            )

            if response.stop_reason == "tool_use":
                self._messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if block.name == "run_hospital_tests":
                        raw = run_hospital_tests(
                            consent=block.input["consent"],
                            gene_expression=block.input["gene_expression"],
                            metabolite_levels=block.input["metabolite_levels"],
                        )
                        self._findings = HospitalFindings(
                            patient_consented=raw["patient_consented"],
                            transcriptomics_confirmed=raw["transcriptomics"].get("confirmed", False),
                            metabolomics_confirmed=raw["metabolomics"].get("confirmed", False),
                            diabetes_confirmed=raw["diabetes_confirmed"],
                            confidence=raw["confidence"],
                            recommendation=HospitalRecommendation(raw["recommendation"]),
                            transcriptomics_summary=raw["transcriptomics"],
                            metabolomics_summary=raw["metabolomics"],
                            reasoning=raw["reasoning"],
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(raw),
                        })

                self._messages.append({"role": "user", "content": tool_results})

            else:
                reply = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                self._messages.append({"role": "assistant", "content": reply})
                return reply
