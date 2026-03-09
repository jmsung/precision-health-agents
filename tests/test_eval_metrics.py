"""Tests for deterministic evaluation metrics."""

from precision_health_agents.eval.cases import EvalCase, ExpectedOutput
from precision_health_agents.eval.metrics import MetricResult, score_decision, score_tool_accuracy
from precision_health_agents.models import (
    AgentResult,
    AgentStatus,
    DoctorFindings,
    GenomicsFindings,
    HealthTrainerFindings,
    Recommendation,
    RiskLevel,
    TranscriptomicsFindings,
    TranscriptomicsRecommendation,
)


def _genomics_result(predicted_class: str = "DMT2") -> AgentResult:
    return AgentResult(
        agent="genomics",
        status=AgentStatus.SUCCESS,
        findings=GenomicsFindings(
            predicted_class=predicted_class,
            confidence=0.85,
            probabilities={"DMT1": 0.05, "DMT2": 0.85, "NONDM": 0.10},
            risk_level=RiskLevel.HIGH,
            interpretation="High risk for Type 2.",
        ),
        summary="DMT2 detected.",
    )


def _doctor_result(prediction: str = "Diabetic") -> AgentResult:
    return AgentResult(
        agent="doctor",
        status=AgentStatus.SUCCESS,
        findings=DoctorFindings(
            prediction=prediction,
            probability=0.74,
            risk_level=RiskLevel.HIGH if prediction == "Diabetic" else RiskLevel.LOW,
            recommendation=(
                Recommendation.HOSPITAL
                if prediction == "Diabetic"
                else Recommendation.HEALTH_TRAINER
            ),
            reasoning="Based on clinical features.",
        ),
        summary=f"Patient is {prediction}.",
    )


def _health_trainer_result(fitness_level: str = "beginner") -> AgentResult:
    return AgentResult(
        agent="health_trainer",
        status=AgentStatus.SUCCESS,
        findings=HealthTrainerFindings(
            fitness_level=fitness_level,
            goals=["improve cardiovascular health"],
            recommended_exercises=[{"Name": "Walking", "Type": "Cardio"}],
            weekly_plan="Walk 30 minutes, 3 days per week.",
        ),
        summary="Exercise plan created.",
    )


def _transcriptomics_result(confirmed: bool = True) -> AgentResult:
    return AgentResult(
        agent="transcriptomics",
        status=AgentStatus.SUCCESS,
        findings=TranscriptomicsFindings(
            pathway_scores={"inflammation_immune": 2.1, "insulin_resistance": 1.8},
            dominant_pathway="inflammation_immune",
            active_pathways=["inflammation_immune", "insulin_resistance"],
            risk_level=RiskLevel.HIGH if confirmed else RiskLevel.LOW,
            dysregulated_genes=[{"gene": "TNF", "z_score": 2.8}],
            diabetes_confirmed={
                "confirmed": confirmed,
                "confidence": 0.92 if confirmed else 0.3,
                "evidence": "test",
            },
            diabetes_subtype={"primary": "Type 2", "evidence": "test"},
            complication_risks=[],
            monitoring={"priority_genes": ["TNF"], "recheck_interval": "3 months"},
            recommendation=TranscriptomicsRecommendation.PHARMACOLOGY,
            interpretation="Test interpretation.",
        ),
        summary="Test transcriptomics result.",
    )


def _case(
    dna_class: str = "DMT2",
    clinical_prediction: str = "Diabetic",
    decision: str = "hospital",
    fitness_level: str | None = None,
    transcriptomics_confirmed: bool | None = None,
) -> EvalCase:
    return EvalCase(
        id="test",
        name="test",
        description="test",
        expected=ExpectedOutput(
            dna_class=dna_class,
            clinical_prediction=clinical_prediction,
            decision=decision,
            fitness_level=fitness_level,
            transcriptomics_confirmed=transcriptomics_confirmed,
        ),
    )


# -- Layer 1: Tool accuracy --------------------------------------------------


class TestToolAccuracy:
    def test_genomics_correct(self):
        result = score_tool_accuracy(_genomics_result("DMT2"), _case(dna_class="DMT2"))
        assert result.score == 1.0
        assert result.passed

    def test_genomics_wrong(self):
        result = score_tool_accuracy(
            _genomics_result("NONDM"), _case(dna_class="DMT2")
        )
        assert result.score == 0.0
        assert not result.passed

    def test_doctor_correct(self):
        result = score_tool_accuracy(
            _doctor_result("Diabetic"), _case(clinical_prediction="Diabetic")
        )
        assert result.score == 1.0
        assert result.passed

    def test_doctor_wrong(self):
        result = score_tool_accuracy(
            _doctor_result("Non-Diabetic"), _case(clinical_prediction="Diabetic")
        )
        assert result.score == 0.0
        assert not result.passed

    def test_health_trainer_correct(self):
        result = score_tool_accuracy(
            _health_trainer_result("beginner"),
            _case(decision="health_trainer", fitness_level="beginner"),
        )
        assert result.score == 1.0
        assert result.passed

    def test_health_trainer_wrong(self):
        result = score_tool_accuracy(
            _health_trainer_result("advanced"),
            _case(decision="health_trainer", fitness_level="beginner"),
        )
        assert result.score == 0.0
        assert not result.passed

    def test_error_agent(self):
        error_result = AgentResult(
            agent="genomics",
            status=AgentStatus.ERROR,
            summary="",
            error="Model not found",
        )
        result = score_tool_accuracy(error_result, _case())
        assert result.score == 0.0
        assert not result.passed


# -- Layer 3: Decision correctness -------------------------------------------


class TestDecisionCorrectness:
    def test_hospital_confirmed(self):
        """DMT2 + Diabetic → hospital."""
        result = score_decision(
            _genomics_result("DMT2"),
            _doctor_result("Diabetic"),
            _case(dna_class="DMT2", clinical_prediction="Diabetic", decision="hospital"),
        )
        assert result.score == 1.0
        assert result.passed

    def test_hospital_dna_override(self):
        """DMT2 + Non-Diabetic → hospital (DNA override)."""
        result = score_decision(
            _genomics_result("DMT2"),
            _doctor_result("Non-Diabetic"),
            _case(decision="hospital"),
        )
        assert result.score == 1.0

    def test_reconsider(self):
        """NONDM + Diabetic → reconsider."""
        result = score_decision(
            _genomics_result("NONDM"),
            _doctor_result("Diabetic"),
            _case(decision="reconsider"),
        )
        assert result.score == 1.0

    def test_health_trainer(self):
        """NONDM + Non-Diabetic → health_trainer."""
        result = score_decision(
            _genomics_result("NONDM"),
            _doctor_result("Non-Diabetic"),
            _case(decision="health_trainer"),
        )
        assert result.score == 1.0

    def test_wrong_decision(self):
        """DMT2 + Diabetic should be hospital, not health_trainer."""
        result = score_decision(
            _genomics_result("DMT2"),
            _doctor_result("Diabetic"),
            _case(decision="health_trainer"),
        )
        assert result.score == 0.0
        assert not result.passed

    def test_decision_with_transcriptomics_override(self):
        """TX confirmed=False overrides hospital → reconsider (false positive filter)."""
        result = score_decision(
            _genomics_result("DMT2"),
            _doctor_result("Diabetic"),
            _case(decision="reconsider"),
            transcriptomics=_transcriptomics_result(confirmed=False),
        )
        assert result.score == 1.0
        assert result.passed
        assert "TX override" in result.detail

    def test_decision_without_transcriptomics(self):
        """Backward compatible — no TX result, 2-layer matrix still works."""
        result = score_decision(
            _genomics_result("DMT2"),
            _doctor_result("Diabetic"),
            _case(decision="hospital"),
            transcriptomics=None,
        )
        assert result.score == 1.0
        assert result.passed


# -- Layer 1: Transcriptomics tool accuracy ---------------------------------


class TestTranscriptomicsAccuracy:
    def test_transcriptomics_correct(self):
        """confirmed=True matches expected=True."""
        result = score_tool_accuracy(
            _transcriptomics_result(confirmed=True),
            _case(transcriptomics_confirmed=True),
        )
        assert result.score == 1.0
        assert result.passed

    def test_transcriptomics_wrong(self):
        """confirmed=True but expected=False."""
        result = score_tool_accuracy(
            _transcriptomics_result(confirmed=True),
            _case(transcriptomics_confirmed=False),
        )
        assert result.score == 0.0
        assert not result.passed
