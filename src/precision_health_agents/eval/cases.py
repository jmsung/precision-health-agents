"""Test cases with ground truth for evaluation."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

_INPUTS_PATH = Path(__file__).parent / "data" / "case_inputs.json"


class ExpectedOutput(BaseModel):
    """Ground truth for a test case."""

    dna_class: Literal["DMT1", "DMT2", "NONDM"] | None = None
    clinical_prediction: Literal["Diabetic", "Non-Diabetic"] | None = None
    decision: Literal["hospital", "reconsider", "health_trainer"]
    drug_class: str | None = None
    # Transcriptomics ground truth (only for hospital-path cases with gene data)
    transcriptomics_confirmed: bool | None = None
    # Health trainer ground truth (only for health_trainer decision)
    fitness_level: Literal["beginner", "intermediate", "advanced"] | None = None
    workout_type: str | None = None  # Cardio, Strength, Flexibility, HIIT


class EvalCase(BaseModel):
    """A single evaluation test case."""

    id: str
    name: str
    description: str
    # Inputs
    dna_sequence: str | None = None
    clinical_features: dict[str, float] | None = None
    gene_expression: dict[str, float] | None = None
    patient_description: str | None = None
    health_trainer_vitals: dict[str, float | int | str] | None = None
    # Ground truth
    expected: ExpectedOutput


def _load_inputs() -> dict | None:
    """Load case inputs from data/eval/case_inputs.json if available."""
    if _INPUTS_PATH.exists():
        return json.loads(_INPUTS_PATH.read_text())
    return None


def _build_cases() -> list[EvalCase]:
    """Build the 4 eval cases, with inputs if available."""
    inputs = _load_inputs()
    dna = inputs["dna_sequences"] if inputs else {}
    clinical = inputs["clinical_features"] if inputs else {}
    gene_expr = inputs.get("gene_expression", {}) if inputs else {}
    ht_vitals = inputs.get("health_trainer_vitals", {}) if inputs else {}

    return [
        EvalCase(
            id="case-1",
            name="Confirmed Diabetic",
            description=(
                "Clinical positive + DMT2 DNA → hospital, Type 2 drugs (metformin). "
                "Both agents agree — strongest signal. "
                "Transcriptomics confirms via inflammatory + insulin resistance pathways."
            ),
            dna_sequence=dna.get("DMT2"),
            clinical_features=clinical.get("diabetic"),
            gene_expression=gene_expr.get("case-1"),
            expected=ExpectedOutput(
                dna_class="DMT2",
                clinical_prediction="Diabetic",
                decision="hospital",
                drug_class="metformin",
                transcriptomics_confirmed=True,
            ),
        ),
        EvalCase(
            id="case-2",
            name="DNA Override — Early Intervention",
            description=(
                "Clinical negative + DMT2 DNA → hospital despite clean labs. "
                "DNA overrides clinical — catch it early. "
                "Transcriptomics confirms early molecular signs."
            ),
            dna_sequence=dna.get("DMT2"),
            clinical_features=clinical.get("non_diabetic"),
            gene_expression=gene_expr.get("case-2"),
            expected=ExpectedOutput(
                dna_class="DMT2",
                clinical_prediction="Non-Diabetic",
                decision="hospital",
                transcriptomics_confirmed=True,
            ),
        ),
        EvalCase(
            id="case-3",
            name="Clinical Override — Avoid Unnecessary Treatment",
            description=(
                "Clinical positive + NONDM DNA → reconsider drugs, lifestyle first. "
                "DNA says no genetic risk — don't over-treat."
            ),
            dna_sequence=dna.get("NONDM"),
            clinical_features=clinical.get("diabetic"),
            expected=ExpectedOutput(
                dna_class="NONDM",
                clinical_prediction="Diabetic",
                decision="reconsider",
            ),
        ),
        EvalCase(
            id="case-4",
            name="Healthy — Prevention",
            description=(
                "Clinical negative + NONDM DNA → health trainer, prevention focus. "
                "No risk from either source."
            ),
            dna_sequence=dna.get("NONDM"),
            clinical_features=clinical.get("non_diabetic"),
            health_trainer_vitals=ht_vitals.get("case-4"),
            expected=ExpectedOutput(
                dna_class="NONDM",
                clinical_prediction="Non-Diabetic",
                decision="health_trainer",
                fitness_level="beginner",
                workout_type="Cardio",
            ),
        ),
    ]


def load_cases() -> list[EvalCase]:
    """Load built-in test cases with inputs."""
    return _build_cases()
