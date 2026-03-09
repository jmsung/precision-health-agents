"""Doctor agent — conversationally gathers clinical data and assesses diabetes risk.

Flow:
  1. Agent chats with the patient to collect 8 clinical measurements.
  2. Once all values are known, Claude calls the classify_diabetes tool.
  3. Agent returns a recommendation: hospital (medicine) or health_trainer.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from precision_health_agents.config import Settings
from precision_health_agents.models import AgentResult, AgentStatus, DoctorFindings, Recommendation, RiskLevel
from precision_health_agents.tools.diabetes_classifier import classify_diabetes

_PROMPTS_DIR = Path(__file__).parents[1] / "prompts"

_TOOL_DEF = {
    "name": "classify_diabetes",
    "description": (
        "Predict diabetes risk from 8 clinical measurements. "
        "Call this once you have collected all values from the patient."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pregnancies":                 {"type": "number", "description": "Number of times pregnant (0 for males)."},
            "glucose":                     {"type": "number", "description": "Plasma glucose concentration (mg/dL)."},
            "blood_pressure":              {"type": "number", "description": "Diastolic blood pressure (mm Hg)."},
            "skin_thickness":              {"type": "number", "description": "Triceps skin fold thickness (mm)."},
            "insulin":                     {"type": "number", "description": "2-hour serum insulin (mu U/ml)."},
            "bmi":                         {"type": "number", "description": "Body mass index (kg/m²)."},
            "diabetes_pedigree_function":  {"type": "number", "description": "Family history score (0.0–2.5)."},
            "age":                         {"type": "number", "description": "Age in years."},
        },
        "required": [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age",
        ],
    },
}


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "doctor.txt").read_text()


def _risk_level(raw_risk: str) -> RiskLevel:
    return {"low": RiskLevel.LOW, "moderate": RiskLevel.MODERATE, "high": RiskLevel.HIGH}[raw_risk]


def _recommendation(findings_raw: dict) -> Recommendation:
    if findings_raw["risk_level"] == "high" or findings_raw["prediction"] == "Diabetic":
        return Recommendation.HOSPITAL
    return Recommendation.HEALTH_TRAINER


class DoctorAgent:
    """Conversational doctor agent that gathers patient data and predicts diabetes risk."""

    name = "doctor"
    role = "Clinical intake and diabetes risk assessment"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings.from_env()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)
        self._system = _load_system_prompt()
        self._messages: list[dict] = []
        self._findings_raw: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, patient_message: str) -> str:
        """Send a patient message and get the doctor's reply.

        Internally calls classify_diabetes when enough data is collected.
        Returns the doctor's natural-language response.
        """
        self._messages.append({"role": "user", "content": patient_message})
        return self._run()

    @property
    def findings(self) -> DoctorFindings | None:
        """Structured findings — available after classify_diabetes has been called."""
        return self._findings_raw and DoctorFindings(
            prediction=self._findings_raw["prediction"],
            probability=self._findings_raw["probability"],
            risk_level=_risk_level(self._findings_raw["risk_level"]),
            recommendation=_recommendation(self._findings_raw),
            reasoning=self._findings_raw.get("reasoning", ""),
        )

    def result(self, summary: str) -> AgentResult:
        """Build a final AgentResult from the completed conversation."""
        f = self.findings
        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS if f else AgentStatus.ERROR,
            findings=f,
            summary=summary,
            error=None if f else "Conversation ended before classification.",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> str:
        """Drive the Claude loop until a text response is ready for the patient."""
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
                    if block.name == "classify_diabetes":
                        raw = classify_diabetes(**block.input)
                        self._findings_raw = raw
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(raw),
                        })

                self._messages.append({"role": "user", "content": tool_results})
                # Loop again so Claude can formulate the final reply to the patient

            else:
                # end_turn — extract text reply
                reply = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                self._messages.append({"role": "assistant", "content": reply})
                return reply
