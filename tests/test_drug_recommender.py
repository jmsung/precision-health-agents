"""Tests for the drug recommendation tool."""

import pytest

from precision_health_agents.tools.drug_recommender import recommend_medications, _load_medications


class TestLoadMedications:
    def test_loads_all_medications(self):
        meds = _load_medications()
        assert len(meds) >= 16

    def test_parses_ada_first_line_as_bool(self):
        meds = _load_medications()
        metformin = next(m for m in meds if m["name"] == "Metformin")
        assert metformin["ada_first_line"] is True

    def test_parses_complications_as_lists(self):
        meds = _load_medications()
        empa = next(m for m in meds if m["name"] == "Empagliflozin")
        assert isinstance(empa["recommended_complications"], list)
        assert "cardiovascular" in empa["recommended_complications"]


class TestRecommendMedications:
    def test_inflammation_dominant_prefers_glp1(self):
        result = recommend_medications(diabetes_subtype="inflammation_dominant")
        names = [m["name"] for m in result["medications"]]
        # GLP-1 RAs should be highly ranked for inflammation
        glp1_present = any("GLP-1" in m["class"] for m in result["medications"])
        assert glp1_present

    def test_beta_cell_failure_includes_insulin(self):
        result = recommend_medications(diabetes_subtype="beta_cell_failure")
        names = [m["name"] for m in result["medications"]]
        assert any("Insulin" in n for n in names)

    def test_metabolic_insulin_resistant_includes_metformin(self):
        result = recommend_medications(diabetes_subtype="metabolic_insulin_resistant")
        names = [m["name"] for m in result["medications"]]
        assert "Metformin" in names

    def test_fibrotic_includes_sglt2_and_acei(self):
        result = recommend_medications(
            diabetes_subtype="fibrotic_complication",
            complication_risks=[{"complication": "diabetic_kidney_disease", "severity": "high"}],
        )
        classes = [m["class"] for m in result["medications"]]
        assert "SGLT2 Inhibitor" in classes
        assert "ACE Inhibitor" in classes

    def test_contraindicated_medications_excluded(self):
        result = recommend_medications(
            diabetes_subtype="metabolic_insulin_resistant",
            complication_risks=[{"complication": "cardiovascular_heart_failure", "severity": "high"}],
        )
        names = [m["name"] for m in result["medications"]]
        assert "Pioglitazone" not in names  # contraindicated in heart failure

    def test_neuropathy_complication_adds_pain_meds(self):
        result = recommend_medications(
            diabetes_subtype="mixed",
            complication_risks=[{"complication": "neuropathy", "severity": "moderate"}],
        )
        names = [m["name"] for m in result["medications"]]
        assert any(n in names for n in ["Pregabalin", "Duloxetine"])

    def test_cv_risk_adds_statin(self):
        result = recommend_medications(
            diabetes_subtype="mixed",
            complication_risks=[{"complication": "cardiovascular", "severity": "high"}],
        )
        names = [m["name"] for m in result["medications"]]
        assert "Atorvastatin" in names

    def test_max_results_limits_output(self):
        result = recommend_medications(diabetes_subtype="mixed", max_results=3)
        assert len(result["medications"]) <= 3

    def test_result_structure(self):
        result = recommend_medications(diabetes_subtype="inflammation_dominant")
        assert "medications" in result
        assert "subtype" in result
        assert result["subtype"] == "inflammation_dominant"
        for med in result["medications"]:
            assert "name" in med
            assert "class" in med
            assert "mechanism" in med
            assert "score" in med
            assert "reasons" in med

    def test_severe_complications_boost_score(self):
        mild = recommend_medications(
            diabetes_subtype="fibrotic_complication",
            complication_risks=[{"complication": "diabetic_kidney_disease", "severity": "moderate"}],
        )
        severe = recommend_medications(
            diabetes_subtype="fibrotic_complication",
            complication_risks=[{"complication": "diabetic_kidney_disease", "severity": "high"}],
        )
        # Kidney-protective drugs should score higher with severe risk
        def _get_score(result, name):
            return next((m["score"] for m in result["medications"] if m["name"] == name), 0)

        assert _get_score(severe, "Ramipril") > _get_score(mild, "Ramipril")
