"""Health trainer agent — gathers patient vitals and exercise history, classifies workout type
using clinical rules informed by diabetes findings, then recommends specific exercises.

Flow:
  1. [Silent] Inject genomics + doctor context into system prompt if available.
  2. Ask: age, gender, height, weight.
  3. Ask: exercise frequency and session duration.
  4. Call classify_workout_type() → get exercise type + experience level.
  5. Ask: equipment, body focus, limitations.
  6. Call recommend_exercises() → get matching exercises.
  7. Deliver plan with clinical reasoning explained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import anthropic

from bioai.config import Settings
from bioai.models import AgentResult, AgentStatus, HealthTrainerFindings
from bioai.tools.exercise_recommender import recommend_exercises
from bioai.tools.workout_type_classifier import classify_workout_type

_PROMPTS_DIR = Path(__file__).parents[1] / "prompts"

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_CLASSIFY_TOOL_DEF = {
    "name": "classify_workout_type",
    "description": (
        "Determine the most suitable workout type (Cardio/Strength/Flexibility/HIIT) "
        "and experience level (Beginner/Intermediate/Expert) based on patient vitals, "
        "exercise history, and diabetes clinical findings. "
        "Call this once you have age, gender, height, weight, workout frequency, and session duration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "age":                        {"type": "integer", "description": "Age in years."},
            "gender":                     {"type": "string",  "description": "Male or Female."},
            "weight_kg":                  {"type": "number",  "description": "Body weight in kilograms."},
            "height_cm":                  {"type": "number",  "description": "Height in centimetres."},
            "workout_frequency_per_week": {"type": "integer", "description": "Days per week currently exercising (0 if none)."},
            "session_duration_hours":     {"type": "number",  "description": "Typical session length in hours (0 if none)."},
            "diabetes_type":              {"type": "string",  "description": "Genomics finding: DMT1, DMT2, or NONDM."},
            "diabetes_probability":       {"type": "number",  "description": "Clinical diabetes probability from doctor (0.0–1.0)."},
        },
        "required": [
            "age", "gender", "weight_kg", "height_cm",
            "workout_frequency_per_week", "session_duration_hours",
        ],
    },
}

_RECOMMEND_TOOL_DEF = {
    "name": "recommend_exercises",
    "description": (
        "Search the exercise database for suitable workouts. "
        "Call after classify_workout_type to find exercises matching the suggested type and level. "
        "May be called more than once for different body parts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "body_part":     {"type": "string",  "description": "Chest, Back, Shoulders, Arms, Core, Legs, or Full Body."},
            "exercise_type": {"type": "string",  "description": "Strength, Cardio, Flexibility, or Plyometric."},
            "difficulty":    {"type": "string",  "description": "Beginner, Intermediate, or Expert."},
            "equipment":     {"type": "string",  "description": "Bodyweight, Dumbbell, Barbell, or Machine."},
            "max_results":   {"type": "integer", "description": "Maximum exercises to return (default 10)."},
        },
        "required": [],
    },
}

_TOOLS = [_CLASSIFY_TOOL_DEF, _RECOMMEND_TOOL_DEF]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FITNESS_LEVELS: dict[str, Literal["beginner", "intermediate", "advanced"]] = {
    "beginner": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "expert": "advanced",
}

_NO_CONTEXT_BLOCK = (
    "No prior clinical findings are available for this patient. "
    "Proceed using demographic and exercise history information only. "
    "Use diabetes_type='NONDM' and diabetes_probability=0.0 when calling classify_workout_type."
)


def _build_clinical_context(context: dict | None) -> str:
    """Format genomics + doctor findings into the system prompt block."""
    if not context:
        return _NO_CONTEXT_BLOCK

    lines: list[str] = ["## Clinical context (from prior agents — read silently, do not repeat verbatim)"]

    genomics = context.get("genomics")
    if genomics and genomics.get("status") == "success":
        f = genomics.get("findings", {})
        lines.append(
            f"- Genomics: predicted_class={f.get('predicted_class', 'NONDM')}, "
            f"confidence={f.get('confidence', 0):.0%}, "
            f"risk_level={f.get('risk_level', 'low')}"
        )

    doctor = context.get("doctor")
    if doctor and doctor.get("status") == "success":
        f = doctor.get("findings", {})
        lines.append(
            f"- Doctor: prediction={f.get('prediction', 'Non-Diabetic')}, "
            f"probability={f.get('probability', 0):.0%}, "
            f"risk_level={f.get('risk_level', 'low')}"
        )

    if len(lines) == 1:
        return _NO_CONTEXT_BLOCK

    lines.append(
        "When calling classify_workout_type, use the diabetes_type from genomics "
        "predicted_class and the diabetes_probability from doctor probability."
    )
    return "\n".join(lines)


def _load_system_prompt(context: dict | None) -> str:
    template = (_PROMPTS_DIR / "health_trainer.txt").read_text()
    clinical_block = _build_clinical_context(context)
    return template.replace("{clinical_context}", clinical_block)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class HealthTrainerAgent:
    """Conversational health trainer — classifies workout type using clinical + demographic
    rules, then recommends specific exercises from the 50-exercise database."""

    name = "health_trainer"
    role = "Exercise prescription informed by diabetes risk findings"

    def __init__(self, settings: Settings | None = None, context: dict | None = None):
        """
        Args:
            settings: App settings (uses env defaults if None).
            context: Optional dict with 'genomics' and/or 'doctor' AgentResult dicts.
                     Injected into the system prompt so the trainer is aware of
                     prior clinical findings without exposing raw data to the patient.
        """
        self._settings = settings or Settings.from_env()
        self._client = anthropic.Anthropic(api_key=self._settings.api_key)
        self._system = _load_system_prompt(context)
        self._messages: list[dict] = []
        self._all_exercises: list[dict] = []
        self._classification: dict | None = None
        self._plan_summary: str = ""
        self._goals: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, patient_message: str) -> str:
        """Send a patient message and get the trainer's reply."""
        self._messages.append({"role": "user", "content": patient_message})
        return self._run()

    @property
    def findings(self) -> HealthTrainerFindings | None:
        """Structured findings — available after both tools have been called."""
        if not self._all_exercises or not self._classification:
            return None
        raw_level = self._classification.get("experience_level", "Beginner").lower()
        level = _FITNESS_LEVELS.get(raw_level, "beginner")
        return HealthTrainerFindings(
            fitness_level=level,
            goals=self._goals,
            recommended_exercises=self._all_exercises,
            weekly_plan=self._plan_summary,
        )

    def result(self, summary: str = "") -> AgentResult:
        """Build a final AgentResult from the completed conversation."""
        if summary:
            self._plan_summary = summary
        f = self.findings
        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS if f else AgentStatus.ERROR,
            findings=f,
            summary=self._plan_summary,
            error=None if f else "Conversation ended before workout classification or exercise recommendation.",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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

                    if block.name == "classify_workout_type":
                        raw = classify_workout_type(**block.input)
                        self._classification = raw
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(raw),
                        })

                    elif block.name == "recommend_exercises":
                        raw = recommend_exercises(**block.input)
                        self._all_exercises.extend(raw.get("exercises", []))
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
                self._plan_summary = reply
                return reply
