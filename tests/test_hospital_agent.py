"""Tests for the HospitalAgent and run_hospital_tests tool."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from bioai.agents.hospital import HospitalAgent, run_hospital_tests
from bioai.models import AgentStatus, HospitalRecommendation
from bioai.tools.gene_expression_analyzer import PATHWAY_GENES, _get_reference_stats as _get_gene_ref
from bioai.tools.metabolic_profile_analyzer import _get_reference_stats as _get_metab_ref


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

def _build_activated_gene_expression(pathways: list[str]) -> dict[str, float]:
    """Build gene expression dict with specified pathways activated (2 std above mean)."""
    ref = _get_gene_ref()
    expr = {}
    for pathway in pathways:
        for gene in PATHWAY_GENES[pathway]:
            if gene in ref:
                mean, std = ref[gene]
                expr[gene] = mean + 2.0 * std
    return expr


def _build_normal_gene_expression() -> dict[str, float]:
    ref = _get_gene_ref()
    return {gene: mean for gene, (mean, _) in ref.items()}


def _build_elevated_metabolites() -> dict[str, float]:
    """Build metabolite levels 3 std above mean for key diabetes markers."""
    ref = _get_metab_ref()
    levels = {}
    for m in ["Glucose", "Fructose", "Mannose", "Leucine", "Isoleucine",
              "Valine", "Cholesterol", "Palmitate", "Lactate", "3-Hydroxybutyrate"]:
        mean, std = ref[m]
        levels[m] = mean + 3 * std
    return levels


def _build_normal_metabolites() -> dict[str, float]:
    ref = _get_metab_ref()
    return {m: mean for m, (mean, _) in ref.items()}


# ---------------------------------------------------------------------------
# Tests: run_hospital_tests tool
# ---------------------------------------------------------------------------

class TestRunHospitalTests:

    def test_no_consent_returns_not_confirmed(self):
        result = run_hospital_tests(
            consent=False,
            gene_expression={},
            metabolite_levels={},
        )
        assert result["patient_consented"] is False
        assert result["diabetes_confirmed"] is False
        assert result["recommendation"] == "health_trainer"

    def test_both_confirm_high_confidence(self):
        gene_expr = _build_activated_gene_expression(["inflammation_immune", "beta_cell_stress"])
        metab = _build_elevated_metabolites()
        result = run_hospital_tests(consent=True, gene_expression=gene_expr, metabolite_levels=metab)
        assert result["patient_consented"] is True
        assert result["diabetes_confirmed"] is True
        assert result["confidence"] == "high"
        assert result["recommendation"] == "pharmacology"

    def test_only_transcriptomics_confirms(self):
        gene_expr = _build_activated_gene_expression(["inflammation_immune", "insulin_resistance"])
        metab = _build_normal_metabolites()
        result = run_hospital_tests(consent=True, gene_expression=gene_expr, metabolite_levels=metab)
        assert result["diabetes_confirmed"] is True
        assert result["confidence"] == "moderate"
        assert result["recommendation"] == "pharmacology"

    def test_only_metabolomics_confirms(self):
        gene_expr = _build_normal_gene_expression()
        metab = _build_elevated_metabolites()
        result = run_hospital_tests(consent=True, gene_expression=gene_expr, metabolite_levels=metab)
        assert result["diabetes_confirmed"] is True
        assert result["confidence"] == "moderate"
        assert result["recommendation"] == "pharmacology"

    def test_neither_confirms_false_positive(self):
        gene_expr = _build_normal_gene_expression()
        metab = _build_normal_metabolites()
        result = run_hospital_tests(consent=True, gene_expression=gene_expr, metabolite_levels=metab)
        assert result["diabetes_confirmed"] is False
        assert result["recommendation"] == "health_trainer"
        assert "false positive" in result["reasoning"].lower()

    def test_result_has_transcriptomics_summary(self):
        gene_expr = _build_activated_gene_expression(["inflammation_immune"])
        metab = _build_normal_metabolites()
        result = run_hospital_tests(consent=True, gene_expression=gene_expr, metabolite_levels=metab)
        assert "confirmed" in result["transcriptomics"]
        assert "active_pathways" in result["transcriptomics"]

    def test_result_has_metabolomics_summary(self):
        gene_expr = _build_normal_gene_expression()
        metab = _build_elevated_metabolites()
        result = run_hospital_tests(consent=True, gene_expression=gene_expr, metabolite_levels=metab)
        assert "confirmed" in result["metabolomics"]
        assert "insulin_resistance_score" in result["metabolomics"]
        assert "pattern" in result["metabolomics"]


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_text(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _mock_tool_use(name: str, tool_input: dict, tool_id: str = "tool_1") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# Tests: HospitalAgent conversational flow
# ---------------------------------------------------------------------------

class TestHospitalAgent:

    def test_agent_explains_tests_to_patient(self):
        """First turn: agent explains blood tests are needed."""
        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent(context={
                "genomics": {"predicted_class": "DMT2", "confidence": 0.88},
                "doctor": {"prediction": "Diabetic", "probability": 0.76},
            })
            agent._client = MagicMock()
            agent._client.messages.create.return_value = _mock_text(
                "Based on your genetic and clinical results, I'd like to run "
                "some blood tests to confirm the diagnosis. Would you be willing?"
            )

            reply = agent.chat("I've been referred to the hospital. What happens now?")
            assert "blood test" in reply.lower() or "confirm" in reply.lower()
            assert agent.findings is None  # no findings yet

    def test_agent_runs_tests_after_consent(self):
        """Patient consents, agent calls run_hospital_tests tool."""
        gene_expr = _build_activated_gene_expression(["inflammation_immune", "beta_cell_stress"])
        metab = _build_elevated_metabolites()

        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent(context={
                "genomics": {"predicted_class": "DMT2", "confidence": 0.88},
                "doctor": {"prediction": "Diabetic", "probability": 0.76},
            })
            agent._client = MagicMock()

            # Turn 1: explain tests
            agent._client.messages.create.side_effect = [
                _mock_text("I'd like to run blood tests. Are you willing?"),
            ]
            agent.chat("What happens now?")

            # Turn 2: patient consents -> tool call -> final response
            agent._client.messages.create.side_effect = [
                _mock_tool_use("run_hospital_tests", {
                    "consent": True,
                    "gene_expression": gene_expr,
                    "metabolite_levels": metab,
                }),
                _mock_text(
                    "Your tests confirm active diabetes with inflammation. "
                    "I recommend seeing a pharmacology specialist."
                ),
            ]
            reply = agent.chat("Yes, I'm willing to do the tests.")

            assert agent.findings is not None
            assert agent.findings.patient_consented is True
            assert agent.findings.diabetes_confirmed is True
            assert agent.findings.recommendation == HospitalRecommendation.PHARMACOLOGY

    def test_agent_handles_declined_consent(self):
        """Patient declines tests."""
        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent()
            agent._client = MagicMock()

            agent._client.messages.create.side_effect = [
                _mock_tool_use("run_hospital_tests", {
                    "consent": False,
                    "gene_expression": {},
                    "metabolite_levels": {},
                }),
                _mock_text("I understand. We'll focus on lifestyle management."),
            ]
            agent.chat("No, I don't want blood tests.")

            assert agent.findings is not None
            assert agent.findings.patient_consented is False
            assert agent.findings.diabetes_confirmed is False
            assert agent.findings.recommendation == HospitalRecommendation.HEALTH_TRAINER

    def test_false_positive_routes_to_health_trainer(self):
        """Both molecular tests negative -> false positive -> health trainer."""
        gene_expr = _build_normal_gene_expression()
        metab = _build_normal_metabolites()

        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent()
            agent._client = MagicMock()

            agent._client.messages.create.side_effect = [
                _mock_tool_use("run_hospital_tests", {
                    "consent": True,
                    "gene_expression": gene_expr,
                    "metabolite_levels": metab,
                }),
                _mock_text("Good news — your molecular tests are normal."),
            ]
            agent.chat("Yes, please run the tests.")

            assert agent.findings.diabetes_confirmed is False
            assert agent.findings.recommendation == HospitalRecommendation.HEALTH_TRAINER

    def test_result_returns_agent_result(self):
        """result() wraps findings in AgentResult."""
        gene_expr = _build_activated_gene_expression(["inflammation_immune"])
        metab = _build_elevated_metabolites()

        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent()
            agent._client = MagicMock()

            agent._client.messages.create.side_effect = [
                _mock_tool_use("run_hospital_tests", {
                    "consent": True,
                    "gene_expression": gene_expr,
                    "metabolite_levels": metab,
                }),
                _mock_text("Tests confirm diabetes."),
            ]
            agent.chat("Yes, run the tests.")

            result = agent.result()
            assert result.status == AgentStatus.SUCCESS
            assert result.agent == "hospital"
            assert result.findings is not None

    def test_result_error_before_tests(self):
        """result() returns ERROR if tests haven't been run."""
        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent()
            result = agent.result()
            assert result.status == AgentStatus.ERROR
            assert result.error == "Tests not completed."

    def test_context_injected_into_prompt(self):
        """Prior genomics/doctor findings appear in system prompt."""
        with patch("bioai.agents.hospital.anthropic.Anthropic"):
            agent = HospitalAgent(context={
                "genomics": {"predicted_class": "DMT2", "confidence": 0.92},
                "doctor": {"prediction": "Diabetic", "probability": 0.80},
            })
            assert "DMT2" in agent._system
            assert "Diabetic" in agent._system
