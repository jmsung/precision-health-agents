"""Tests for the gene expression analyzer tool."""

import pytest

from precision_health_agents.tools.gene_expression_analyzer import (
    PATHWAY_GENES,
    PATHWAYS,
    _get_reference_stats,
    analyze_gene_expression,
)


class TestAnalyzeGeneExpression:
    """Tests for analyze_gene_expression()."""

    def test_empty_input_returns_low_risk(self):
        result = analyze_gene_expression({})
        assert result["risk_level"] == "low"
        assert result["dominant_pathway"] == "none"
        assert result["active_pathways"] == []
        assert result["dysregulated_genes"] == []

    def test_unknown_genes_returns_low_risk(self):
        result = analyze_gene_expression({"FAKE_GENE": 999.0, "ANOTHER": 50.0})
        assert result["risk_level"] == "low"
        assert result["diabetes_confirmed"]["confirmed"] is False
        assert result["recommendation"] == "health_trainer"

    def test_returns_all_pathway_scores(self):
        # Use a few real genes with normal-ish values
        expr = {"TNF": 100.0, "IL6": 50.0, "INS": 200.0, "SOD2": 300.0}
        result = analyze_gene_expression(expr)
        for pathway in PATHWAYS:
            assert pathway in result["pathway_scores"]

    def test_high_inflammation_detected(self):
        """High values for inflammatory genes should activate inflammation pathway."""
        ref_stats = _get_reference_stats()
        # Set inflammatory genes to mean + 2*std (strong upregulation)
        expr = {}
        for gene in PATHWAY_GENES["inflammation_immune"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std

        result = analyze_gene_expression(expr)
        assert result["pathway_scores"]["inflammation_immune"] > 1.0
        assert "inflammation_immune" in result["active_pathways"]

    def test_high_beta_cell_stress_detected(self):
        """High values for beta cell genes should activate beta cell pathway."""
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["beta_cell_stress"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std

        result = analyze_gene_expression(expr)
        assert result["pathway_scores"]["beta_cell_stress"] > 1.0
        assert result["dominant_pathway"] == "beta_cell_stress"

    def test_dysregulated_genes_identified(self):
        """Genes with |z-score| > 1 should appear in dysregulated list."""
        ref_stats = _get_reference_stats()
        # Set TNF very high
        if "TNF" in ref_stats:
            mean, std = ref_stats["TNF"]
            expr = {"TNF": mean + 3.0 * std}
            result = analyze_gene_expression(expr)
            gene_names = [g["gene"] for g in result["dysregulated_genes"]]
            assert "TNF" in gene_names
            tnf_entry = next(g for g in result["dysregulated_genes"] if g["gene"] == "TNF")
            assert tnf_entry["direction"] == "up"
            assert tnf_entry["z_score"] > 2.0

    def test_risk_level_high_with_multiple_active_pathways(self):
        """Multiple active pathways should yield high risk."""
        ref_stats = _get_reference_stats()
        expr = {}
        # Activate 3 pathways
        for pathway in ["inflammation_immune", "beta_cell_stress", "oxidative_mitochondrial"]:
            for gene in PATHWAY_GENES[pathway]:
                if gene in ref_stats:
                    mean, std = ref_stats[gene]
                    expr[gene] = mean + 2.0 * std

        result = analyze_gene_expression(expr)
        assert result["risk_level"] == "high"

    def test_normal_values_give_low_risk(self):
        """Mean expression values should give near-zero pathway scores."""
        ref_stats = _get_reference_stats()
        # Use mean values for all genes
        expr = {gene: mean for gene, (mean, _) in ref_stats.items()}
        result = analyze_gene_expression(expr)
        assert result["risk_level"] == "low"
        # All pathway scores should be near zero
        for score in result["pathway_scores"].values():
            assert abs(score) < 0.5

    def test_interpretation_string_not_empty(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["fibrosis_ecm"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std

        result = analyze_gene_expression(expr)
        assert len(result["interpretation"]) > 0
        assert "fibrosis" in result["interpretation"].lower() or "extracellular" in result["interpretation"].lower()


class TestDiabetesConfirmation:
    """Tests for diabetes confirmation (false positive filter)."""

    def test_confirmed_with_active_pathways(self):
        """Active diabetes pathways should confirm diabetes."""
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["inflammation_immune"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert result["diabetes_confirmed"]["confirmed"] is True
        assert result["recommendation"] == "pharmacology"

    def test_not_confirmed_with_normal_values(self):
        """Normal gene expression should NOT confirm — false positive."""
        ref_stats = _get_reference_stats()
        expr = {gene: mean for gene, (mean, _) in ref_stats.items()}
        result = analyze_gene_expression(expr)
        assert result["diabetes_confirmed"]["confirmed"] is False
        assert result["recommendation"] == "health_trainer"

    def test_not_confirmed_with_empty_input(self):
        """Empty input should NOT confirm."""
        result = analyze_gene_expression({})
        assert result["diabetes_confirmed"]["confirmed"] is False
        assert result["recommendation"] == "health_trainer"

    def test_high_confidence_with_multiple_active(self):
        """Multiple active pathways should give high confidence confirmation."""
        ref_stats = _get_reference_stats()
        expr = {}
        for pathway in ["inflammation_immune", "beta_cell_stress"]:
            for gene in PATHWAY_GENES[pathway]:
                if gene in ref_stats:
                    mean, std = ref_stats[gene]
                    expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert result["diabetes_confirmed"]["confirmed"] is True
        assert result["diabetes_confirmed"]["confidence"] == "high"

    def test_interpretation_mentions_false_positive(self):
        """False positive should mention health trainer in interpretation."""
        ref_stats = _get_reference_stats()
        expr = {gene: mean for gene, (mean, _) in ref_stats.items()}
        result = analyze_gene_expression(expr)
        assert "false positive" in result["interpretation"].lower()
        assert "health trainer" in result["interpretation"].lower()

    def test_interpretation_mentions_pharmacology_when_confirmed(self):
        """Confirmed diabetes should mention pharmacology in interpretation."""
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["beta_cell_stress"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert "pharmacology" in result["interpretation"].lower()


class TestSubtypeClassification:
    """Tests for diabetes subtype classification."""

    def test_inflammation_dominant_subtype(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["inflammation_immune"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert result["diabetes_subtype"]["subtype"] == "inflammation_dominant"

    def test_beta_cell_failure_subtype(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["beta_cell_stress"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert result["diabetes_subtype"]["subtype"] == "beta_cell_failure"

    def test_normal_subtype_when_no_activation(self):
        ref_stats = _get_reference_stats()
        expr = {gene: mean for gene, (mean, _) in ref_stats.items()}
        result = analyze_gene_expression(expr)
        assert result["diabetes_subtype"]["subtype"] == "normal"

    def test_mixed_subtype_with_many_active_pathways(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for pathway in ["inflammation_immune", "beta_cell_stress", "oxidative_mitochondrial"]:
            for gene in PATHWAY_GENES[pathway]:
                if gene in ref_stats:
                    mean, std = ref_stats[gene]
                    expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert result["diabetes_subtype"]["subtype"] == "mixed"


class TestComplicationRisks:
    """Tests for complication risk assessment."""

    def test_kidney_disease_risk_with_fibrosis(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["fibrosis_ecm"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        complications = [r["complication"] for r in result["complication_risks"]]
        assert "diabetic_kidney_disease" in complications

    def test_cardiovascular_risk_with_inflammation_and_oxidative(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for pathway in ["inflammation_immune", "oxidative_mitochondrial"]:
            for gene in PATHWAY_GENES[pathway]:
                if gene in ref_stats:
                    mean, std = ref_stats[gene]
                    expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        complications = [r["complication"] for r in result["complication_risks"]]
        assert "cardiovascular" in complications

    def test_no_complications_at_normal_levels(self):
        ref_stats = _get_reference_stats()
        expr = {gene: mean for gene, (mean, _) in ref_stats.items()}
        result = analyze_gene_expression(expr)
        assert result["complication_risks"] == []

    def test_beta_cell_exhaustion_risk(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["beta_cell_stress"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        complications = [r["complication"] for r in result["complication_risks"]]
        assert "beta_cell_exhaustion" in complications


class TestMonitoring:
    """Tests for monitoring recommendations."""

    def test_actionable_for_high_risk(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for pathway in ["inflammation_immune", "beta_cell_stress", "oxidative_mitochondrial"]:
            for gene in PATHWAY_GENES[pathway]:
                if gene in ref_stats:
                    mean, std = ref_stats[gene]
                    expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        assert result["monitoring"]["level"] == "actionable"

    def test_exploratory_for_low_risk(self):
        ref_stats = _get_reference_stats()
        expr = {gene: mean for gene, (mean, _) in ref_stats.items()}
        result = analyze_gene_expression(expr)
        assert result["monitoring"]["level"] == "exploratory"

    def test_follow_ups_for_kidney_risk(self):
        ref_stats = _get_reference_stats()
        expr = {}
        for gene in PATHWAY_GENES["fibrosis_ecm"]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std
        result = analyze_gene_expression(expr)
        follow_ups = " ".join(result["monitoring"]["follow_ups"])
        assert "nephrology" in follow_ups.lower() or "kidney" in follow_ups.lower()


class TestReferenceStats:
    """Tests for reference data loading."""

    def test_reference_stats_load(self):
        stats = _get_reference_stats()
        assert len(stats) > 50  # Should have 110 genes
        # All stats should have mean and std
        for gene, (mean, std) in stats.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)

    def test_reference_has_key_genes(self):
        stats = _get_reference_stats()
        key_genes = ["TNF", "IL6", "INS", "SOD2", "COL1A1", "INSR"]
        for gene in key_genes:
            assert gene in stats, f"Missing key gene: {gene}"
