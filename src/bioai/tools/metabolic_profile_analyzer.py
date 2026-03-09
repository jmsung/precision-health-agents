"""Metabolic profile analysis tool for diabetes metabolomics.

Analyzes a patient's metabolic profile against the ST001906-derived
reference dataset. Classifies metabolic patterns into 5 categories:
1. Amino acid metabolism (BCAAs + aromatic amino acids)
2. Carbohydrate / glucose metabolism
3. Lipid / fatty acid metabolism
4. TCA cycle / energy metabolism
5. Ketone body / oxidative metabolism
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parents[3] / "data" / "metabolomics"
_DATASET_PATH = _DATA_DIR / "raw" / "diabetes_metabolomics.csv"

PATHWAYS = [
    "amino_acid",
    "carbohydrate",
    "lipid",
    "tca_energy",
    "ketone_oxidative",
]

PATHWAY_METABOLITES: dict[str, list[str]] = {
    "amino_acid": [
        "Alanine", "Glycine", "Leucine", "Isoleucine", "Valine",
        "Phenylalanine", "Tyrosine", "Tryptophan", "Lysine", "Histidine",
        "Methionine", "Proline", "Serine", "Threonine", "Glutamine",
        "Glutamate", "Asparagine", "Aspartate", "Arginine", "Ornithine",
        "Cysteine", "Cystine", "Hydroxyproline",
    ],
    "carbohydrate": [
        "Glucose", "Fructose", "Galactose", "Mannose", "Allose",
        "Arabinose", "Xylulose", "Gluconate", "Glucuronate",
        "1,5-Anhydroglucitol", "Inositol", "Mannitol", "Erythritol",
        "N-Acetylglucosamine",
    ],
    "lipid": [
        "Cholesterol", "Oleate", "Palmitate", "Stearate", "Linoleate",
        "Palmitoleate", "Myristate", "Laureate", "Heptadecanoate",
        "Pentadecanoate", "Elaidiate", "alpha-Tocopherol",
    ],
    "tca_energy": [
        "Citrate", "Succinate", "Malate", "Pyruvate", "Lactate",
        "Glycerate", "Glycolate", "Phosphate", "Gluconate",
    ],
    "ketone_oxidative": [
        "3-Hydroxybutyrate", "2-Hydroxybutyrate", "2-Aminobutyrate",
        "3-Aminoisobutyrate", "Ketoisoleucine", "Ketovaline",
        "Urate", "Creatinine",
    ],
}

# BCAAs are key insulin resistance biomarkers
_BCAA = {"Leucine", "Isoleucine", "Valine"}
# Aromatic AAs are also T2D predictive
_AROMATIC_AA = {"Phenylalanine", "Tyrosine", "Tryptophan"}


@lru_cache(maxsize=1)
def _load_reference() -> pd.DataFrame:
    """Load the reference dataset."""
    return pd.read_csv(_DATASET_PATH)


def _get_reference_stats() -> dict[str, tuple[float, float]]:
    """Get mean and std for each metabolite from the reference dataset."""
    ref = _load_reference()
    stats = {}
    skip = {"sample_id", "condition", "condition_numeric"}
    for col in ref.columns:
        if col in skip or col.startswith("pathway_"):
            continue
        values = ref[col].astype(float)
        stats[col] = (values.mean(), values.std())
    return stats


def analyze_metabolic_profile(metabolite_levels: dict[str, float]) -> dict:
    """Analyze metabolic profile for diabetes-related metabolic state.

    Args:
        metabolite_levels: dict mapping metabolite names to concentration values.
            Example: {"Glucose": 5000000, "Leucine": 200000, "Cholesterol": 20000}

    Returns:
        dict with metabolite_scores, elevated_metabolites, insulin_resistance_score,
        metabolic_pattern, risk_level, subtype_refinement, diabetes_confirmed,
        interpretation.
    """
    ref_stats = _get_reference_stats()

    # Compute per-metabolite z-scores
    metabolite_zscores: dict[str, float] = {}
    for name, value in metabolite_levels.items():
        if name in ref_stats:
            mean, std = ref_stats[name]
            if std > 0:
                metabolite_zscores[name] = round((value - mean) / std, 4)

    if not metabolite_zscores:
        return {
            "metabolite_scores": {},
            "elevated_metabolites": [],
            "insulin_resistance_score": 0.0,
            "metabolic_pattern": "normal",
            "risk_level": "low",
            "subtype_refinement": {"subtype": "normal", "confidence": "high", "reasoning": "No matching metabolites found."},
            "diabetes_confirmed": {"confirmed": False, "confidence": "low", "reasoning": "No matching metabolites in profile."},
            "interpretation": "No matching metabolites found in profile.",
        }

    # Compute pathway scores (mean z-score of pathway metabolites)
    pathway_scores: dict[str, float] = {}
    for pathway in PATHWAYS:
        mets = PATHWAY_METABOLITES[pathway]
        scores = [metabolite_zscores[m] for m in mets if m in metabolite_zscores]
        pathway_scores[pathway] = round(float(np.mean(scores)), 4) if scores else 0.0

    # Elevated metabolites (z-score > 1.0)
    elevated = [m for m, z in metabolite_zscores.items() if z > 1.0]

    # Insulin resistance score (0.0-1.0) based on BCAA + glucose + lipid signals
    ir_signals = []
    # BCAA elevation is a strong IR predictor
    bcaa_zscores = [metabolite_zscores[m] for m in _BCAA if m in metabolite_zscores]
    if bcaa_zscores:
        ir_signals.append(np.mean(bcaa_zscores))
    # Glucose elevation
    if "Glucose" in metabolite_zscores:
        ir_signals.append(metabolite_zscores["Glucose"])
    # Aromatic AA elevation
    aaa_zscores = [metabolite_zscores[m] for m in _AROMATIC_AA if m in metabolite_zscores]
    if aaa_zscores:
        ir_signals.append(np.mean(aaa_zscores))
    # Lipid pathway
    if pathway_scores.get("lipid", 0) != 0:
        ir_signals.append(pathway_scores["lipid"])

    if ir_signals:
        raw_ir = float(np.mean(ir_signals))
        # Sigmoid-like mapping to 0-1 range
        insulin_resistance_score = round(1.0 / (1.0 + np.exp(-raw_ir)), 4)
    else:
        insulin_resistance_score = 0.5  # uninformative

    # Metabolic pattern classification
    metabolic_pattern = _classify_pattern(pathway_scores)

    # Risk level
    active_pathways = [p for p, s in pathway_scores.items() if s > 0.5]
    max_score = max(pathway_scores.values()) if pathway_scores else 0.0
    if max_score > 1.5 or len(active_pathways) >= 3:
        risk_level = "high"
    elif max_score > 0.5 or len(active_pathways) >= 1:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Diabetes confirmation
    diabetes_confirmed = _confirm_diabetes(pathway_scores, active_pathways, max_score, insulin_resistance_score)

    # Subtype refinement
    subtype_refinement = _refine_subtype(pathway_scores, metabolic_pattern, insulin_resistance_score)

    # Interpretation
    interpretation = _build_interpretation(
        pathway_scores, metabolic_pattern, elevated, risk_level,
        insulin_resistance_score, diabetes_confirmed, subtype_refinement,
    )

    return {
        "metabolite_scores": {m: metabolite_zscores[m] for m in sorted(metabolite_zscores)},
        "elevated_metabolites": sorted(elevated),
        "insulin_resistance_score": insulin_resistance_score,
        "metabolic_pattern": metabolic_pattern,
        "risk_level": risk_level,
        "subtype_refinement": subtype_refinement,
        "diabetes_confirmed": diabetes_confirmed,
        "interpretation": interpretation,
    }


def _classify_pattern(pathway_scores: dict[str, float]) -> str:
    """Classify the dominant metabolic pattern."""
    active = {p: s for p, s in pathway_scores.items() if s > 0.5}

    if len(active) >= 3:
        return "mixed"
    if not active:
        return "normal"

    dominant = max(pathway_scores, key=pathway_scores.get)
    pattern_map = {
        "amino_acid": "bcaa_elevation",
        "carbohydrate": "glucose_dysregulation",
        "lipid": "lipid_dysregulation",
        "tca_energy": "energy_metabolism_shift",
        "ketone_oxidative": "ketone_accumulation",
    }
    return pattern_map.get(dominant, "mixed")


def _confirm_diabetes(
    pathway_scores: dict[str, float],
    active_pathways: list[str],
    max_score: float,
    ir_score: float,
) -> dict:
    """Determine if metabolic profile confirms diabetes.

    Metabolomics is the most dynamic layer — reflects current metabolic state.
    Confirmation requires converging evidence from multiple metabolite panels.
    """
    num_active = len(active_pathways)
    mildly_elevated = sum(1 for s in pathway_scores.values() if s > 0.2)

    if (num_active >= 2 or max_score > 1.0) and ir_score > 0.6:
        return {
            "confirmed": True,
            "confidence": "high",
            "reasoning": (
                f"{num_active} active pathway(s), max score {max_score:.2f}, "
                f"IR score {ir_score:.2f}. Strong metabolic evidence of diabetes."
            ),
        }

    if num_active >= 1 and ir_score > 0.55:
        return {
            "confirmed": True,
            "confidence": "moderate",
            "reasoning": (
                f"{num_active} active pathway(s), max score {max_score:.2f}, "
                f"IR score {ir_score:.2f}. Moderate metabolic evidence."
            ),
        }

    if max_score > 0.3 and mildly_elevated >= 2 and ir_score > 0.55:
        return {
            "confirmed": True,
            "confidence": "low",
            "reasoning": (
                f"No strongly active pathways, but {mildly_elevated} mildly elevated "
                f"(max: {max_score:.2f}), IR score {ir_score:.2f}. Weak metabolic signal."
            ),
        }

    return {
        "confirmed": False,
        "confidence": "high" if max_score < 0.1 else "moderate",
        "reasoning": (
            f"No active metabolic pathways (max: {max_score:.2f}), "
            f"IR score {ir_score:.2f}. Metabolic profile within normal range. "
            "No metabolomic evidence of active diabetes."
        ),
    }


def _refine_subtype(
    pathway_scores: dict[str, float],
    metabolic_pattern: str,
    ir_score: float,
) -> dict:
    """Refine diabetes subtype based on metabolic signature.

    Metabolomics can distinguish:
    - metabolic_insulin_resistant: high BCAA + lipid dysregulation (classic T2DM)
    - lipid_predominant: lipid-driven with normal amino acids
    - glucose_centric: isolated glucose dysregulation (early/mild T2DM)
    - ketotic: ketone accumulation suggesting insulin deficiency
    """
    if metabolic_pattern == "normal":
        return {"subtype": "normal", "confidence": "high", "reasoning": "No metabolic dysregulation."}

    if metabolic_pattern == "mixed":
        return {"subtype": "metabolic_insulin_resistant", "confidence": "moderate",
                "reasoning": "Multiple metabolic pathways affected — classic insulin resistance pattern."}

    pattern_to_subtype = {
        "bcaa_elevation": "metabolic_insulin_resistant",
        "lipid_dysregulation": "lipid_predominant",
        "glucose_dysregulation": "glucose_centric",
        "ketone_accumulation": "ketotic",
        "energy_metabolism_shift": "metabolic_insulin_resistant",
    }
    subtype = pattern_to_subtype.get(metabolic_pattern, "metabolic_insulin_resistant")

    # Confidence from how extreme the IR score is
    if ir_score > 0.8 or ir_score < 0.3:
        confidence = "high"
    elif ir_score > 0.65 or ir_score < 0.4:
        confidence = "moderate"
    else:
        confidence = "low"

    reasoning_map = {
        "metabolic_insulin_resistant": "Elevated BCAAs/energy metabolites indicating insulin resistance.",
        "lipid_predominant": "Lipid dysregulation dominant — consider statin therapy.",
        "glucose_centric": "Isolated glucose/carbohydrate dysregulation — early or mild T2DM.",
        "ketotic": "Ketone body accumulation — possible insulin deficiency component.",
    }

    return {
        "subtype": subtype,
        "confidence": confidence,
        "reasoning": reasoning_map.get(subtype, f"Pattern: {metabolic_pattern}"),
    }


def _build_interpretation(
    pathway_scores: dict[str, float],
    metabolic_pattern: str,
    elevated: list[str],
    risk_level: str,
    ir_score: float,
    diabetes_confirmed: dict,
    subtype_refinement: dict,
) -> str:
    """Build human-readable interpretation."""
    parts = []

    # Diabetes confirmation
    if diabetes_confirmed["confirmed"]:
        parts.append(
            f"Metabolic confirmation: POSITIVE (confidence: {diabetes_confirmed['confidence']}). "
            f"Insulin resistance score: {ir_score:.2f}."
        )
    else:
        parts.append(
            f"Metabolic confirmation: NEGATIVE (confidence: {diabetes_confirmed['confidence']}). "
            f"Insulin resistance score: {ir_score:.2f}. "
            "No metabolomic evidence of active diabetes."
        )
        parts.append(f"Overall risk: {risk_level}.")
        return " ".join(parts)

    # Pattern
    pattern_labels = {
        "bcaa_elevation": "branched-chain amino acid elevation (insulin resistance biomarker)",
        "lipid_dysregulation": "lipid metabolism dysregulation",
        "glucose_dysregulation": "glucose/carbohydrate metabolism dysregulation",
        "ketone_accumulation": "ketone body accumulation (possible insulin deficiency)",
        "energy_metabolism_shift": "TCA cycle/energy metabolism shift",
        "mixed": "multi-pathway metabolic dysregulation",
    }
    parts.append(f"Dominant pattern: {pattern_labels.get(metabolic_pattern, metabolic_pattern)}.")

    # Subtype
    parts.append(
        f"Metabolic subtype: {subtype_refinement['subtype']} "
        f"(confidence: {subtype_refinement['confidence']})."
    )

    # Active pathways
    active = [p for p, s in pathway_scores.items() if s > 0.5]
    if active:
        pathway_labels = {
            "amino_acid": "amino acid metabolism",
            "carbohydrate": "carbohydrate metabolism",
            "lipid": "lipid metabolism",
            "tca_energy": "TCA/energy metabolism",
            "ketone_oxidative": "ketone/oxidative metabolism",
        }
        labels = [f"{pathway_labels.get(p, p)} ({pathway_scores[p]:+.2f})" for p in active]
        parts.append(f"Active pathways: {', '.join(labels)}.")

    if elevated:
        parts.append(f"Elevated metabolites: {', '.join(elevated[:8])}.")

    parts.append(f"Overall risk: {risk_level}.")
    return " ".join(parts)
