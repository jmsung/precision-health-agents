"""Pharmacology agent — recommends personalized diabetes medication plans based on
transcriptomics findings (molecular subtype, complication risks, pathway scores).

Flow:
  1. Inject transcriptomics + upstream context into system prompt.
  2. Call recommend_medications() with subtype + complications.
  3. Claude composes a kind, supportive medication plan with clinical reasoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic

from precision_health_agents.config import Settings
from precision_health_agents.models import AgentResult, AgentStatus, PharmacologyFindings
from precision_health_agents.tools.drug_recommender import recommend_medications

_PROMPTS_DIR = Path(__file__).parents[1] / "prompts"

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_RECOMMEND_TOOL_DEF = {
    "name": "recommend_medications",
    "description": (
        "Search the ADA guideline-based medication database for diabetes drugs "
        "matching the patient's molecular subtype and complication profile. "
        "Call this with the diabetes_subtype from transcriptomics and any "
        "complication_risks identified."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "diabetes_subtype": {
                "type": "string",
                "description": (
                    "Molecular diabetes subtype from transcriptomics: "
                    "inflammation_dominant, beta_cell_failure, "
                    "metabolic_insulin_resistant, fibrotic_complication, or mixed."
                ),
            },
            "complication_risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "complication": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                },
                "description": "List of complication risk objects with complication name and severity.",
            },
            "active_pathways": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Active pathway names from transcriptomics analysis.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum medications to return (default 8).",
            },
        },
        "required": ["diabetes_subtype"],
    },
}

_TOOLS = [_RECOMMEND_TOOL_DEF]

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

_NO_CONTEXT_BLOCK = (
    "No transcriptomics findings available. "
    "Provide general diabetes medication guidance based on clinical best practices."
)


def _build_clinical_context(context: dict | None) -> str:
    """Format transcriptomics + upstream findings into the system prompt."""
    if not context:
        return _NO_CONTEXT_BLOCK

    lines: list[str] = ["## Clinical context (from prior agents — use to inform medication selection)"]

    # Genomics context
    genomics = context.get("genomics")
    if genomics and genomics.get("status") == "success":
        f = genomics.get("findings", {})
        lines.append(
            f"- **Genomics**: predicted_class={f.get('predicted_class', 'unknown')}, "
            f"confidence={f.get('confidence', 0):.0%}, "
            f"risk_level={f.get('risk_level', 'unknown')}"
        )

    # Doctor context
    doctor = context.get("doctor")
    if doctor and doctor.get("status") == "success":
        f = doctor.get("findings", {})
        lines.append(
            f"- **Doctor**: prediction={f.get('prediction', 'unknown')}, "
            f"probability={f.get('probability', 0):.0%}, "
            f"risk_level={f.get('risk_level', 'unknown')}"
        )

    # Transcriptomics context (primary input)
    trans = context.get("transcriptomics")
    if trans and trans.get("status") == "success":
        f = trans.get("findings", {})
        subtype = f.get("diabetes_subtype", {})
        confirmed = f.get("diabetes_confirmed", {})
        pathways = f.get("pathway_scores", {})
        complications = f.get("complication_risks", [])
        active = f.get("active_pathways", [])

        lines.append(f"- **Diabetes confirmed**: {confirmed.get('confirmed', False)} "
                      f"(confidence: {confirmed.get('confidence', 'unknown')})")
        lines.append(f"- **Molecular subtype**: {subtype.get('subtype', 'unknown')} "
                      f"(confidence: {subtype.get('confidence', 'unknown')})")
        lines.append(f"- **Active pathways**: {', '.join(active) if active else 'none'}")

        if pathways:
            pathway_str = ", ".join(f"{k}={v:.2f}" for k, v in pathways.items())
            lines.append(f"- **Pathway scores**: {pathway_str}")

        if complications:
            for c in complications:
                lines.append(f"- **Complication risk**: {c.get('complication', '?')} "
                              f"(severity: {c.get('severity', '?')}, "
                              f"evidence: {c.get('evidence', '?')})")

        lines.append("")
        lines.append(
            "Use the diabetes_subtype and complication_risks above when calling "
            "recommend_medications. Tailor your explanation to this patient's specific "
            "molecular profile."
        )
    else:
        return _NO_CONTEXT_BLOCK

    return "\n".join(lines)


def _load_system_prompt(context: dict | None) -> str:
    template = (_PROMPTS_DIR / "pharmacology.txt").read_text()
    clinical_block = _build_clinical_context(context)
    return template.replace("{clinical_context}", clinical_block)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PharmacologyAgent:
    """Recommends personalized diabetes medication plans based on molecular subtype
    and complication risks from transcriptomics analysis."""

    name = "pharmacology"
    role = "Evidence-based diabetes medication recommendation"

    def __init__(self, settings: Settings | None = None, context: dict | None = None):
        """
        Args:
            settings: App settings (uses env defaults if None).
            context: Dict with 'transcriptomics' (required) and optionally
                     'genomics' and 'doctor' AgentResult dicts.
        """
        self._settings = settings or Settings.from_env()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)
        self._context = context
        self._system = _load_system_prompt(context)
        self._messages: list[dict] = []
        self._medications: list[dict[str, Any]] = []
        self._plan_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, patient_message: str) -> str:
        """Send a message and get the pharmacologist's reply."""
        self._messages.append({"role": "user", "content": patient_message})
        return self._run()

    @property
    def findings(self) -> PharmacologyFindings | None:
        """Structured findings — available after recommend_medications has been called."""
        if not self._medications:
            return None

        primary = [m for m in self._medications if m.get("score", 0) >= 3.0]
        supportive = [m for m in self._medications if m.get("score", 0) < 3.0]

        return PharmacologyFindings(
            diabetes_subtype=self._get_subtype(),
            primary_medications=primary,
            supportive_medications=supportive,
            monitoring_plan=self._extract_monitoring(),
            medication_summary=self._plan_text,
        )

    def result(self, summary: str = "") -> AgentResult:
        """Build a final AgentResult from the completed conversation."""
        if summary:
            self._plan_text = summary
        f = self.findings
        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS if f else AgentStatus.ERROR,
            findings=f,
            summary=self._plan_text,
            error=None if f else "Conversation ended before medication recommendations were made.",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_subtype(self) -> str:
        if self._context:
            trans = self._context.get("transcriptomics", {})
            findings = trans.get("findings", {})
            subtype = findings.get("diabetes_subtype", {})
            return subtype.get("subtype", "unknown")
        return "unknown"

    def _extract_monitoring(self) -> str:
        """Collect monitoring requirements from all recommended medications."""
        monitors = []
        for med in self._medications:
            m = med.get("monitoring", "")
            if m:
                monitors.append(f"{med['name']}: {m}")
        return "; ".join(monitors) if monitors else "Standard diabetes monitoring"

    def _run(self) -> str:
        """Drive the Claude tool-use loop until a text response is ready."""
        while True:
            response = self._client.messages.create(
                model=self._settings.agent_model,
                max_tokens=self._settings.max_tokens,
                system=self._system,
                tools=_TOOLS,
                messages=self._messages,
            )

            if response.stop_reason == "tool_use":
                self._messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    if block.name == "recommend_medications":
                        raw = recommend_medications(**block.input)
                        self._medications.extend(raw.get("medications", []))
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
                self._plan_text = reply
                return reply
