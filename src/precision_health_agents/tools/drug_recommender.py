"""Drug recommendation tool — matches diabetes subtype and complication risks
to evidence-based medications from the curated ADA guideline database.

Returns a ranked list of medications with clinical reasoning for each.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).parents[3] / "data" / "medications" / "raw" / "diabetes_medications.csv"


def _load_medications() -> list[dict[str, Any]]:
    """Load the medication database from CSV."""
    rows: list[dict[str, Any]] = []
    with _DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["ada_first_line"] = row["ada_first_line"].strip().lower() == "true"
            row["contraindicated_complications"] = [
                c.strip() for c in row["contraindicated_complications"].split(",") if c.strip()
            ]
            row["recommended_complications"] = [
                c.strip() for c in row["recommended_complications"].split(",") if c.strip()
            ]
            rows.append(row)
    return rows


def recommend_medications(
    diabetes_subtype: str,
    complication_risks: list[dict[str, str]] | None = None,
    active_pathways: list[str] | None = None,
    max_results: int = 8,
) -> dict[str, Any]:
    """Recommend medications based on molecular subtype and complication profile.

    Args:
        diabetes_subtype: Molecular subtype from transcriptomics
            (inflammation_dominant, beta_cell_failure, metabolic_insulin_resistant,
             fibrotic_complication, mixed).
        complication_risks: List of dicts with 'complication' and 'severity' keys.
        active_pathways: List of active pathway names from transcriptomics.
        max_results: Maximum medications to return.

    Returns:
        Dict with 'medications' list (scored and sorted), 'subtype', and 'reasoning'.
    """
    medications = _load_medications()
    complication_risks = complication_risks or []
    active_pathways = active_pathways or []

    patient_complications = {r["complication"] for r in complication_risks}
    severe_complications = {
        r["complication"] for r in complication_risks if r.get("severity") == "high"
    }

    scored: list[dict[str, Any]] = []

    for med in medications:
        score = 0.0
        reasons: list[str] = []

        # Contraindication check
        contra = set(med["contraindicated_complications"])
        if contra & patient_complications:
            continue  # skip contraindicated medications

        # Subtype match
        if med["primary_subtype"] == diabetes_subtype:
            score += 3.0
            reasons.append(f"Primary match for {diabetes_subtype} subtype")
        elif med["primary_subtype"] == "mixed":
            score += 1.0
            reasons.append("Broadly applicable across subtypes")

        # ADA first-line bonus
        if med["ada_first_line"]:
            score += 2.0
            reasons.append("ADA first-line recommendation")

        # Complication benefit match
        recommended = set(med["recommended_complications"])
        matched_complications = recommended & patient_complications
        if matched_complications:
            score += 2.0 * len(matched_complications)
            reasons.append(f"Addresses: {', '.join(matched_complications)}")

        # Severe complication priority
        severe_matched = recommended & severe_complications
        if severe_matched:
            score += 1.5 * len(severe_matched)
            reasons.append(f"High-priority for severe: {', '.join(severe_matched)}")

        if score > 0:
            scored.append({
                "name": med["name"],
                "class": med["class"],
                "mechanism": med["mechanism"],
                "route": med["route"],
                "monitoring": med["monitoring"],
                "common_side_effects": med["common_side_effects"],
                "notes": med["notes"],
                "score": score,
                "reasons": reasons,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:max_results]

    return {
        "medications": results,
        "subtype": diabetes_subtype,
        "total_matched": len(scored),
        "complications_considered": [r["complication"] for r in complication_risks],
    }
