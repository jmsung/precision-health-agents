"""Deterministic evaluation metrics (Layer 1 + Layer 3)."""

from pydantic import BaseModel

from precision_health_agents.eval.cases import EvalCase
from precision_health_agents.models import (
    AgentResult,
    AgentStatus,
    DoctorFindings,
    GenomicsFindings,
    HealthTrainerFindings,
    TranscriptomicsFindings,
)


class MetricResult(BaseModel):
    """Result of a single metric evaluation."""

    metric: str
    agent: str
    score: float  # 0.0-1.0
    passed: bool
    detail: str


# -- Decision matrix ----------------------------------------------------------

_DECISION_MATRIX: dict[tuple[str, str], str] = {
    # (dna_risk, clinical_prediction) → decision
    # dna_risk: "high" for DMT1/DMT2, "low" for NONDM
    ("high", "Diabetic"): "hospital",
    ("high", "Non-Diabetic"): "hospital",
    ("low", "Diabetic"): "reconsider",
    ("low", "Non-Diabetic"): "health_trainer",
}


def _dna_risk(predicted_class: str) -> str:
    """Map DNA class to risk category for decision matrix."""
    return "low" if predicted_class == "NONDM" else "high"


# -- Layer 1: Tool accuracy --------------------------------------------------


def score_tool_accuracy(agent_result: AgentResult, case: EvalCase) -> MetricResult:
    """Score whether an agent's tool produced the correct prediction."""
    if agent_result.status == AgentStatus.ERROR:
        return MetricResult(
            metric="tool_accuracy",
            agent=agent_result.agent,
            score=0.0,
            passed=False,
            detail=f"Agent error: {agent_result.error}",
        )

    findings = agent_result.findings

    if isinstance(findings, GenomicsFindings):
        expected = case.expected.dna_class
        actual = findings.predicted_class
        correct = actual == expected
        return MetricResult(
            metric="tool_accuracy",
            agent="genomics",
            score=1.0 if correct else 0.0,
            passed=correct,
            detail=f"expected={expected}, actual={actual}",
        )

    if isinstance(findings, DoctorFindings):
        expected = case.expected.clinical_prediction
        actual = findings.prediction
        correct = actual == expected
        return MetricResult(
            metric="tool_accuracy",
            agent="doctor",
            score=1.0 if correct else 0.0,
            passed=correct,
            detail=f"expected={expected}, actual={actual}",
        )

    if isinstance(findings, TranscriptomicsFindings):
        expected_confirmed = case.expected.transcriptomics_confirmed
        actual_confirmed = findings.diabetes_confirmed.get("confirmed", False)
        correct = actual_confirmed == expected_confirmed
        return MetricResult(
            metric="tool_accuracy",
            agent="transcriptomics",
            score=1.0 if correct else 0.0,
            passed=correct,
            detail=f"expected_confirmed={expected_confirmed}, actual_confirmed={actual_confirmed}",
        )

    if isinstance(findings, HealthTrainerFindings):
        expected_level = case.expected.fitness_level
        actual_level = findings.fitness_level
        correct = actual_level == expected_level
        return MetricResult(
            metric="tool_accuracy",
            agent="health_trainer",
            score=1.0 if correct else 0.0,
            passed=correct,
            detail=f"expected_level={expected_level}, actual_level={actual_level}",
        )

    return MetricResult(
        metric="tool_accuracy",
        agent=agent_result.agent,
        score=0.0,
        passed=False,
        detail=f"Unknown findings type: {type(findings).__name__}",
    )


# -- Layer 3: Decision correctness -------------------------------------------


def score_decision(
    genomics: AgentResult,
    doctor: AgentResult,
    case: EvalCase,
    transcriptomics: AgentResult | None = None,
) -> MetricResult:
    """Score whether the combined agent outputs produce the correct decision.

    When transcriptomics is present and diabetes_confirmed is False, overrides
    hospital → reconsider (false positive filter from 3rd validation layer).
    """
    if not isinstance(genomics.findings, GenomicsFindings):
        return MetricResult(
            metric="decision",
            agent="combined",
            score=0.0,
            passed=False,
            detail="Missing genomics findings",
        )

    if not isinstance(doctor.findings, DoctorFindings):
        return MetricResult(
            metric="decision",
            agent="combined",
            score=0.0,
            passed=False,
            detail="Missing doctor findings",
        )

    risk = _dna_risk(genomics.findings.predicted_class)
    prediction = doctor.findings.prediction
    actual_decision = _DECISION_MATRIX.get((risk, prediction))

    # 3rd layer: transcriptomics override — false positive filter
    tx_override = False
    if (
        transcriptomics
        and isinstance(transcriptomics.findings, TranscriptomicsFindings)
        and actual_decision == "hospital"
        and not transcriptomics.findings.diabetes_confirmed.get("confirmed", True)
    ):
        actual_decision = "reconsider"
        tx_override = True

    expected_decision = case.expected.decision
    correct = actual_decision == expected_decision

    detail = (
        f"dna={genomics.findings.predicted_class} "
        f"clinical={prediction} → {actual_decision} "
        f"(expected {expected_decision})"
    )
    if tx_override:
        detail += " [TX override: hospital→reconsider]"

    return MetricResult(
        metric="decision",
        agent="combined",
        score=1.0 if correct else 0.0,
        passed=correct,
        detail=detail,
    )
