"""Gene expression analysis tool for diabetes transcriptomics.

Analyzes a patient's gene expression profile against the GSE26168-derived
diabetes pathway gene panels. Classifies expression patterns into 5 categories:
1. Beta cell stress / insulin secretion
2. Inflammation & immune activation
3. Insulin resistance / signaling
4. Fibrosis & extracellular matrix remodeling
5. Oxidative & mitochondrial stress
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parents[3] / "data" / "transcriptomics"
_DATASET_PATH = _DATA_DIR / "raw" / "diabetes_transcriptomics.csv"

PATHWAYS = [
    "beta_cell_stress",
    "inflammation_immune",
    "insulin_resistance",
    "fibrosis_ecm",
    "oxidative_mitochondrial",
]

# Genes belonging to each pathway
PATHWAY_GENES: dict[str, list[str]] = {
    "beta_cell_stress": [
        "INS", "GCG", "PDX1", "NKX6-1", "MAFA", "SLC2A2", "GCK",
        "PCSK1", "PCSK2", "ABCC8", "KCNJ11", "SLC30A8", "TCF7L2",
        "NEUROD1", "PAX6", "IAPP", "CHGA", "CHGB", "UCN3", "ERO1B",
    ],
    "inflammation_immune": [
        "TNF", "IL6", "IL1B", "CXCL8", "CCL2", "CCL5",
        "NFKB1", "RELA", "IKBKB", "TLR4", "TLR2", "NLRP3",
        "CASP1", "IL18", "IFNG", "IL10", "TGFB1", "CD68", "ITGAX",
        "CD14", "CRP", "SAA1", "SOCS3", "JAK2", "STAT3",
    ],
    "insulin_resistance": [
        "INSR", "IRS1", "IRS2", "PIK3CA", "PIK3R1", "AKT1", "AKT2",
        "SLC2A4", "PPARG", "PPARGC1A", "ADIPOQ", "LEP", "LEPR",
        "PTPN1", "GRB2", "SOS1", "MAPK1", "MAPK3", "GSK3B",
        "FOXO1", "SREBF1", "PCK1", "G6PC", "PRKAA1", "PRKAA2",
    ],
    "fibrosis_ecm": [
        "COL1A1", "COL3A1", "COL4A1", "FN1", "TGFB1", "TGFBR1",
        "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "CTGF",
        "MMP2", "MMP9", "TIMP1", "TIMP2", "ACTA2", "VIM",
        "LOX", "SERPINE1", "THBS1", "SPARC",
    ],
    "oxidative_mitochondrial": [
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "NFE2L2", "KEAP1",
        "HMOX1", "NQO1", "TXNRD1", "TXN", "PRDX1", "PRDX3",
        "NDUFS1", "SDHA", "UQCRC1",
        "COX5A", "ATP5F1A", "UCP2", "PPARGC1A", "SIRT1", "SIRT3",
    ],
}


@lru_cache(maxsize=1)
def _load_reference() -> pd.DataFrame:
    """Load the reference dataset for computing z-scores."""
    return pd.read_csv(_DATASET_PATH, index_col=0)


def _get_reference_stats() -> dict[str, tuple[float, float]]:
    """Get mean and std for each gene from the reference dataset."""
    ref = _load_reference()
    stats = {}
    for col in ref.columns:
        if col in ("condition", "condition_numeric") or col.startswith("pathway_"):
            continue
        values = ref[col].astype(float)
        stats[col] = (values.mean(), values.std())
    return stats


def analyze_gene_expression(gene_expression: dict[str, float]) -> dict:
    """Analyze a patient's gene expression profile for diabetes-related pathway activity.

    Args:
        gene_expression: Dictionary mapping gene symbols to expression values.
            Example: {"TNF": 1250.5, "IL6": 890.2, "INS": 45.0, ...}
            At minimum, provide values for genes in the pathway panels.

    Returns:
        dict with keys:
            - pathway_scores: dict mapping pathway name to z-score (float)
            - dominant_pathway: str, the pathway with highest activation
            - active_pathways: list[str], pathways with z-score > 0.5
            - risk_level: str, "low" | "moderate" | "high"
            - dysregulated_genes: list[dict], genes with |z-score| > 1.0
            - interpretation: str, summary of findings
    """
    ref_stats = _get_reference_stats()

    # Compute per-gene z-scores
    gene_zscores: dict[str, float] = {}
    for gene, value in gene_expression.items():
        if gene in ref_stats:
            mean, std = ref_stats[gene]
            if std > 0:
                gene_zscores[gene] = (value - mean) / std

    if not gene_zscores:
        return {
            "pathway_scores": {p: 0.0 for p in PATHWAYS},
            "dominant_pathway": "none",
            "active_pathways": [],
            "risk_level": "low",
            "dysregulated_genes": [],
            "diabetes_confirmed": {
                "confirmed": False,
                "confidence": "low",
                "reasoning": "No matching genes found in expression profile.",
            },
            "diabetes_subtype": {"subtype": "normal", "confidence": "high"},
            "complication_risks": [],
            "monitoring": {"level": "exploratory", "follow_ups": []},
            "recommendation": "health_trainer",
            "interpretation": "No matching genes found in expression profile. "
            "No molecular evidence of diabetes. Recommend health trainer pathway.",
        }

    # Compute pathway scores (mean z-score of pathway genes)
    pathway_scores: dict[str, float] = {}
    for pathway in PATHWAYS:
        genes = PATHWAY_GENES[pathway]
        scores = [gene_zscores[g] for g in genes if g in gene_zscores]
        pathway_scores[pathway] = round(float(np.mean(scores)), 4) if scores else 0.0

    # Identify active pathways (z-score > 0.5)
    active_pathways = [p for p, s in pathway_scores.items() if s > 0.5]

    # Dominant pathway
    dominant = max(pathway_scores, key=pathway_scores.get)
    if pathway_scores[dominant] <= 0.0:
        dominant = "none"

    # Dysregulated genes (|z-score| > 1.0)
    dysregulated = []
    for gene, zscore in sorted(gene_zscores.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(zscore) > 1.0:
            # Find which pathways this gene belongs to
            pathways = [p for p, genes in PATHWAY_GENES.items() if gene in genes]
            dysregulated.append({
                "gene": gene,
                "z_score": round(zscore, 3),
                "direction": "up" if zscore > 0 else "down",
                "pathways": pathways,
            })

    # Risk level
    max_score = max(pathway_scores.values())
    num_active = len(active_pathways)
    if max_score > 1.5 or num_active >= 3:
        risk_level = "high"
    elif max_score > 0.5 or num_active >= 1:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Diabetes confirmation — molecular evidence check
    diabetes_confirmed = _confirm_diabetes(pathway_scores, active_pathways, max_score)

    # Diabetes subtype classification (only meaningful if confirmed)
    diabetes_subtype = _classify_subtype(pathway_scores, dominant)

    # Complication risk flags
    complication_risks = _assess_complication_risks(pathway_scores, active_pathways)

    # Monitoring recommendation
    monitoring = _recommend_monitoring(risk_level, active_pathways, complication_risks)

    # Routing decision: pharmacology (drugs) vs health_trainer (false positive)
    recommendation = (
        "pharmacology" if diabetes_confirmed["confirmed"] else "health_trainer"
    )

    # Interpretation
    interpretation = _build_interpretation(
        pathway_scores, active_pathways, dominant, dysregulated,
        risk_level, diabetes_subtype, complication_risks,
        diabetes_confirmed, recommendation,
    )

    return {
        "pathway_scores": pathway_scores,
        "dominant_pathway": dominant,
        "active_pathways": active_pathways,
        "risk_level": risk_level,
        "dysregulated_genes": dysregulated[:10],  # top 10
        "diabetes_confirmed": diabetes_confirmed,
        "diabetes_subtype": diabetes_subtype,
        "complication_risks": complication_risks,
        "monitoring": monitoring,
        "recommendation": recommendation,
        "interpretation": interpretation,
    }


def _confirm_diabetes(
    pathway_scores: dict[str, float],
    active_pathways: list[str],
    max_score: float,
) -> dict:
    """Determine if gene expression provides molecular evidence of active diabetes.

    This serves as a third validator after Genomics and Doctor agents.
    If no diabetes-related pathways are active, the patient may be a false positive
    and should be routed to health trainer instead of pharmacology.

    Confirmation criteria:
    - At least 1 pathway with z-score > 0.5 (active), OR
    - Max pathway score > 0.3 with 2+ pathways showing mild elevation (> 0.2)
    """
    num_active = len(active_pathways)
    mildly_elevated = sum(1 for s in pathway_scores.values() if s > 0.2)

    if num_active >= 2 or max_score > 1.0:
        return {
            "confirmed": True,
            "confidence": "high",
            "reasoning": (
                f"{num_active} active pathway(s) with max score {max_score:.2f}. "
                "Strong molecular evidence of diabetes-related dysregulation."
            ),
        }

    if num_active >= 1:
        return {
            "confirmed": True,
            "confidence": "moderate",
            "reasoning": (
                f"{num_active} active pathway(s) with max score {max_score:.2f}. "
                "Moderate molecular evidence of diabetes-related dysregulation."
            ),
        }

    if max_score > 0.3 and mildly_elevated >= 2:
        return {
            "confirmed": True,
            "confidence": "low",
            "reasoning": (
                f"No strongly active pathways, but {mildly_elevated} pathways show "
                f"mild elevation (max: {max_score:.2f}). Weak but present molecular signal."
            ),
        }

    return {
        "confirmed": False,
        "confidence": "high" if max_score < 0.1 else "moderate",
        "reasoning": (
            f"No active diabetes pathways (max score: {max_score:.2f}). "
            "Gene expression is within normal range. "
            "Possible false positive from clinical/genomic assessment. "
            "Recommend health trainer pathway instead of pharmacology."
        ),
    }


def _classify_subtype(pathway_scores: dict[str, float], dominant: str) -> dict:
    """Classify diabetes subtype based on pathway activation pattern.

    Subtypes based on transcriptomic literature:
    - inflammation_dominant: immune/inflammatory pathways drive insulin resistance
    - beta_cell_failure: primary beta cell dysfunction with secretory deficit
    - metabolic_insulin_resistant: insulin signaling + mitochondrial dysfunction
    - fibrotic_complication: tissue remodeling suggesting organ damage (e.g. kidney)
    - mixed: multiple pathways co-activated without clear dominant pattern
    - normal: no significant pathway activation
    """
    subtype_labels = {
        "inflammation_immune": "inflammation_dominant",
        "beta_cell_stress": "beta_cell_failure",
        "insulin_resistance": "metabolic_insulin_resistant",
        "fibrosis_ecm": "fibrotic_complication",
        "oxidative_mitochondrial": "metabolic_insulin_resistant",
    }

    if dominant == "none":
        return {"subtype": "normal", "confidence": "high"}

    active = [p for p, s in pathway_scores.items() if s > 0.5]

    # If 3+ pathways active, it's mixed
    if len(active) >= 3:
        return {"subtype": "mixed", "confidence": "low"}

    subtype = subtype_labels.get(dominant, "mixed")

    # Confidence based on how clearly dominant the top pathway is
    scores = sorted(pathway_scores.values(), reverse=True)
    gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    if gap > 1.0:
        confidence = "high"
    elif gap > 0.3:
        confidence = "moderate"
    else:
        confidence = "low"

    return {"subtype": subtype, "confidence": confidence}


def _assess_complication_risks(
    pathway_scores: dict[str, float],
    active_pathways: list[str],
) -> list[dict]:
    """Flag complication risks based on pathway activation.

    Maps pathway patterns to known diabetes complications per ADA guidelines.
    """
    risks = []

    # Diabetic kidney disease: fibrosis + inflammation
    fibrosis = pathway_scores.get("fibrosis_ecm", 0)
    inflammation = pathway_scores.get("inflammation_immune", 0)
    if fibrosis > 0.5 or (fibrosis > 0.3 and inflammation > 0.3):
        severity = "high" if fibrosis > 1.0 else "moderate"
        risks.append({
            "complication": "diabetic_kidney_disease",
            "severity": severity,
            "evidence": "fibrosis_ecm and/or inflammation_immune pathway activation",
        })

    # Cardiovascular risk: inflammation + oxidative stress
    oxidative = pathway_scores.get("oxidative_mitochondrial", 0)
    if inflammation > 0.5 and oxidative > 0.3:
        severity = "high" if inflammation > 1.0 else "moderate"
        risks.append({
            "complication": "cardiovascular",
            "severity": severity,
            "evidence": "inflammation_immune + oxidative_mitochondrial co-activation",
        })

    # Beta cell exhaustion → insulin dependency risk
    beta = pathway_scores.get("beta_cell_stress", 0)
    if beta > 1.0:
        risks.append({
            "complication": "beta_cell_exhaustion",
            "severity": "high",
            "evidence": "severe beta_cell_stress pathway activation",
        })
    elif beta > 0.5:
        risks.append({
            "complication": "beta_cell_exhaustion",
            "severity": "moderate",
            "evidence": "beta_cell_stress pathway activation",
        })

    # Neuropathy risk: oxidative + insulin resistance
    insulin_res = pathway_scores.get("insulin_resistance", 0)
    if oxidative > 0.5 and insulin_res > 0.3:
        risks.append({
            "complication": "neuropathy",
            "severity": "moderate",
            "evidence": "oxidative_mitochondrial + insulin_resistance co-activation",
        })

    return risks


def _recommend_monitoring(
    risk_level: str,
    active_pathways: list[str],
    complication_risks: list[dict],
) -> dict:
    """Recommend monitoring level and follow-up actions.

    Categories per the transcriptomics note:
    - actionable: supports guideline-based management decisions
    - monitoring: suggests closer follow-up for specific complications
    - exploratory: hypothesis-generating, requires further validation
    """
    if risk_level == "high":
        level = "actionable"
    elif risk_level == "moderate":
        level = "monitoring"
    else:
        level = "exploratory"

    follow_ups = []
    for risk in complication_risks:
        comp = risk["complication"]
        if comp == "diabetic_kidney_disease":
            follow_ups.append("nephrology referral and kidney function monitoring (eGFR, UACR)")
        elif comp == "cardiovascular":
            follow_ups.append("cardiovascular risk assessment and lipid panel")
        elif comp == "beta_cell_exhaustion":
            follow_ups.append("C-peptide and insulin secretion capacity evaluation")
        elif comp == "neuropathy":
            follow_ups.append("peripheral neuropathy screening")

    if not follow_ups and active_pathways:
        follow_ups.append("routine diabetes monitoring with pathway-informed attention")

    return {
        "level": level,
        "follow_ups": follow_ups,
    }


def _build_interpretation(
    pathway_scores: dict[str, float],
    active_pathways: list[str],
    dominant: str,
    dysregulated: list[dict],
    risk_level: str,
    diabetes_subtype: dict,
    complication_risks: list[dict],
    diabetes_confirmed: dict,
    recommendation: str,
) -> str:
    """Build a human-readable interpretation of the analysis."""
    pathway_labels = {
        "beta_cell_stress": "beta cell stress and insulin secretion dysfunction",
        "inflammation_immune": "inflammation and immune activation",
        "insulin_resistance": "insulin resistance and impaired signaling",
        "fibrosis_ecm": "fibrosis and extracellular matrix remodeling",
        "oxidative_mitochondrial": "oxidative and mitochondrial stress",
    }

    parts = []

    # Diabetes confirmation decision (most important)
    if diabetes_confirmed["confirmed"]:
        parts.append(
            f"Molecular confirmation: POSITIVE (confidence: {diabetes_confirmed['confidence']}). "
            f"Recommend: pharmacology pathway."
        )
    else:
        parts.append(
            f"Molecular confirmation: NEGATIVE (confidence: {diabetes_confirmed['confidence']}). "
            f"No molecular evidence of active diabetes despite clinical/genomic indicators. "
            f"Possible false positive. Recommend: health trainer pathway instead of drugs."
        )
        # For false positives, keep interpretation brief
        parts.append(f"Overall risk assessment: {risk_level}.")
        return " ".join(parts)

    # Subtype (only for confirmed cases)
    subtype = diabetes_subtype["subtype"]
    subtype_labels = {
        "inflammation_dominant": "inflammation-dominant T2DM",
        "beta_cell_failure": "beta cell failure T2DM",
        "metabolic_insulin_resistant": "metabolic/insulin-resistant T2DM",
        "fibrotic_complication": "fibrotic complication-associated T2DM",
        "mixed": "mixed-mechanism T2DM",
    }
    parts.append(
        f"Diabetes subtype: {subtype_labels.get(subtype, subtype)} "
        f"(confidence: {diabetes_subtype['confidence']})."
    )

    if dominant != "none":
        parts.append(
            f"Dominant pathway: {pathway_labels.get(dominant, dominant)} "
            f"(z-score: {pathway_scores[dominant]:.2f})."
        )

    if active_pathways:
        labels = [pathway_labels.get(p, p) for p in active_pathways]
        parts.append(f"Active pathways: {', '.join(labels)}.")

    # Complication risks
    if complication_risks:
        risk_strs = [f"{r['complication']} ({r['severity']})" for r in complication_risks]
        parts.append(f"Complication risks: {', '.join(risk_strs)}.")

    if dysregulated:
        up = [g for g in dysregulated if g["direction"] == "up"]
        down = [g for g in dysregulated if g["direction"] == "down"]
        if up:
            genes_str = ", ".join(f"{g['gene']} ({g['z_score']:+.2f})" for g in up[:5])
            parts.append(f"Upregulated: {genes_str}.")
        if down:
            genes_str = ", ".join(f"{g['gene']} ({g['z_score']:+.2f})" for g in down[:5])
            parts.append(f"Downregulated: {genes_str}.")

    parts.append(f"Overall risk assessment: {risk_level}.")
    return " ".join(parts)
