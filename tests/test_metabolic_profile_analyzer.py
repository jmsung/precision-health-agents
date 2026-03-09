"""Tests for the metabolic profile analyzer tool."""

import pytest

from bioai.tools.metabolic_profile_analyzer import (
    PATHWAY_METABOLITES,
    PATHWAYS,
    _get_reference_stats,
    analyze_metabolic_profile,
)


class TestReferenceData:
    def test_reference_stats_loads(self):
        stats = _get_reference_stats()
        assert len(stats) > 0
        assert "Glucose" in stats

    def test_reference_stats_have_mean_and_std(self):
        stats = _get_reference_stats()
        mean, std = stats["Glucose"]
        assert mean > 0
        assert std > 0

    def test_all_pathway_metabolites_in_reference(self):
        """Every metabolite in our pathway panels should exist in the reference data."""
        stats = _get_reference_stats()
        for pathway, metabolites in PATHWAY_METABOLITES.items():
            for m in metabolites:
                assert m in stats, f"{m} from {pathway} not in reference"


class TestAnalyzeMetabolicProfile:
    def test_empty_input(self):
        result = analyze_metabolic_profile({})
        assert result["metabolic_pattern"] == "normal"
        assert result["risk_level"] == "low"
        assert result["diabetes_confirmed"]["confirmed"] is False

    def test_unknown_metabolites(self):
        result = analyze_metabolic_profile({"FakeMetabolite": 999.0})
        assert result["metabolite_scores"] == {}
        assert result["metabolic_pattern"] == "normal"

    def test_normal_profile(self):
        """Mean values from reference should produce a normal profile."""
        stats = _get_reference_stats()
        # Use mean values for key metabolites
        levels = {m: stats[m][0] for m in ["Glucose", "Leucine", "Cholesterol", "Lactate"]}
        result = analyze_metabolic_profile(levels)
        assert result["risk_level"] == "low"
        assert result["insulin_resistance_score"] == pytest.approx(0.5, abs=0.1)

    def test_diabetic_profile_high_glucose(self):
        """Very high glucose + BCAAs should flag as diabetic."""
        stats = _get_reference_stats()
        levels = {}
        # Set glucose to 3 std above mean
        for m in ["Glucose", "Leucine", "Isoleucine", "Valine", "Phenylalanine"]:
            mean, std = stats[m]
            levels[m] = mean + 3 * std
        result = analyze_metabolic_profile(levels)
        assert result["insulin_resistance_score"] > 0.7
        assert result["diabetes_confirmed"]["confirmed"] is True
        assert len(result["elevated_metabolites"]) > 0

    def test_elevated_metabolites_detected(self):
        stats = _get_reference_stats()
        mean, std = stats["Glucose"]
        result = analyze_metabolic_profile({"Glucose": mean + 2 * std})
        assert "Glucose" in result["elevated_metabolites"]

    def test_metabolite_scores_are_zscores(self):
        stats = _get_reference_stats()
        mean, std = stats["Glucose"]
        result = analyze_metabolic_profile({"Glucose": mean + 2 * std})
        assert result["metabolite_scores"]["Glucose"] == pytest.approx(2.0, abs=0.01)

    def test_output_keys(self):
        result = analyze_metabolic_profile({"Glucose": 5000000.0})
        expected_keys = {
            "metabolite_scores", "elevated_metabolites", "insulin_resistance_score",
            "metabolic_pattern", "risk_level", "subtype_refinement",
            "diabetes_confirmed", "interpretation",
        }
        assert set(result.keys()) == expected_keys

    def test_ir_score_range(self):
        """IR score should always be in [0, 1]."""
        stats = _get_reference_stats()
        # Very high values
        levels = {m: stats[m][0] + 5 * stats[m][1] for m in ["Glucose", "Leucine", "Valine"]}
        result = analyze_metabolic_profile(levels)
        assert 0.0 <= result["insulin_resistance_score"] <= 1.0

        # Very low values
        levels = {m: stats[m][0] - 5 * stats[m][1] for m in ["Glucose", "Leucine", "Valine"]}
        result = analyze_metabolic_profile(levels)
        assert 0.0 <= result["insulin_resistance_score"] <= 1.0


class TestMetabolicPatterns:
    def test_bcaa_elevation_pattern(self):
        stats = _get_reference_stats()
        levels = {}
        for m in ["Leucine", "Isoleucine", "Valine", "Alanine", "Glycine"]:
            mean, std = stats[m]
            levels[m] = mean + 2 * std
        result = analyze_metabolic_profile(levels)
        assert result["metabolic_pattern"] in ("bcaa_elevation", "mixed")

    def test_lipid_dysregulation_pattern(self):
        stats = _get_reference_stats()
        levels = {}
        for m in ["Cholesterol", "Oleate", "Palmitate", "Stearate"]:
            mean, std = stats[m]
            levels[m] = mean + 2 * std
        result = analyze_metabolic_profile(levels)
        assert result["metabolic_pattern"] in ("lipid_dysregulation", "mixed")

    def test_normal_pattern(self):
        stats = _get_reference_stats()
        levels = {m: stats[m][0] for m in ["Glucose", "Leucine"]}
        result = analyze_metabolic_profile(levels)
        assert result["metabolic_pattern"] == "normal"


class TestDiabetesConfirmation:
    def test_confirmed_high_confidence(self):
        stats = _get_reference_stats()
        # Elevate multiple pathways strongly
        levels = {}
        for m in ["Glucose", "Fructose", "Mannose", "Leucine", "Isoleucine", "Valine",
                   "Cholesterol", "Palmitate", "Lactate", "3-Hydroxybutyrate"]:
            mean, std = stats[m]
            levels[m] = mean + 3 * std
        result = analyze_metabolic_profile(levels)
        assert result["diabetes_confirmed"]["confirmed"] is True
        assert result["diabetes_confirmed"]["confidence"] == "high"

    def test_not_confirmed_normal_profile(self):
        stats = _get_reference_stats()
        levels = {m: stats[m][0] for m in ["Glucose", "Leucine", "Cholesterol"]}
        result = analyze_metabolic_profile(levels)
        assert result["diabetes_confirmed"]["confirmed"] is False

    def test_confirmation_has_reasoning(self):
        result = analyze_metabolic_profile({"Glucose": 9999999.0})
        assert len(result["diabetes_confirmed"]["reasoning"]) > 0


class TestSubtypeRefinement:
    def test_subtype_has_required_fields(self):
        result = analyze_metabolic_profile({"Glucose": 5000000.0})
        sub = result["subtype_refinement"]
        assert "subtype" in sub
        assert "confidence" in sub
        assert "reasoning" in sub

    def test_normal_subtype_for_normal_profile(self):
        stats = _get_reference_stats()
        levels = {m: stats[m][0] for m in ["Glucose", "Leucine"]}
        result = analyze_metabolic_profile(levels)
        assert result["subtype_refinement"]["subtype"] == "normal"
