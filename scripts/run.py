"""E2E pipeline — run a patient case through all validation layers.

Usage:
    uv run python scripts/run.py --case 1              # run case 1 (real API)
    uv run python scripts/run.py --case 1 --mock       # run case 1 (pre-recorded)
    uv run python scripts/run.py --all --mock           # run all cases (pre-recorded)
    uv run python scripts/run.py --list                 # list available cases
"""

import argparse
import asyncio
import json
from pathlib import Path

from precision_health_agents.config import Settings
from precision_health_agents.eval.cases import EvalCase, load_cases
from precision_health_agents.models import (
    AgentResult,
    DoctorFindings,
    GenomicsFindings,
    PharmacologyFindings,
    TranscriptomicsFindings,
)

MOCK_DIR = Path("src/precision_health_agents/eval/data/mock_outputs")


# -- Agent runners -----------------------------------------------------------


async def run_genomics(case: EvalCase, settings: Settings) -> AgentResult:
    from precision_health_agents.agents.genomics import GenomicsAgent

    agent = GenomicsAgent()
    return await agent.analyze(case.dna_sequence or "")


def run_doctor(case: EvalCase, settings: Settings) -> AgentResult:
    from precision_health_agents.agents.doctor import DoctorAgent

    agent = DoctorAgent()
    if case.clinical_features:
        feature_str = ", ".join(f"{k}={v}" for k, v in case.clinical_features.items())
        agent.chat(f"My clinical values: {feature_str}")
    elif case.patient_description:
        agent.chat(case.patient_description)
    return agent.result(summary="Pipeline run")


async def run_transcriptomics(
    case: EvalCase,
    settings: Settings,
    context: dict,
) -> AgentResult:
    from precision_health_agents.agents.transcriptomics import TranscriptomicsAgent

    agent = TranscriptomicsAgent(settings=settings)
    gene_data = case.gene_expression or {}
    query = "Gene expression profile:\n" + "\n".join(
        f"  {gene}: {val}" for gene, val in gene_data.items()
    )
    return await agent.analyze(query, context=context)


def run_pharmacology(
    settings: Settings,
    context: dict,
    tx_findings: TranscriptomicsFindings,
) -> AgentResult:
    from precision_health_agents.agents.pharmacology import PharmacologyAgent

    agent = PharmacologyAgent(settings=settings, context=context)
    subtype = tx_findings.diabetes_subtype.get("primary", "Type 2")
    complications = ", ".join(
        c.get("complication", "") for c in tx_findings.complication_risks
    )
    msg = (
        f"Patient confirmed with {subtype} diabetes. "
        f"Complication risks: {complications or 'none identified'}. "
        f"Dominant pathway: {tx_findings.dominant_pathway}. "
        f"Please recommend a medication plan."
    )
    agent.chat(msg)
    return agent.result(summary="Pipeline run")


def run_health_trainer(
    case: EvalCase,
    settings: Settings,
    context: dict,
) -> AgentResult:
    from precision_health_agents.agents.health_trainer import HealthTrainerAgent

    agent = HealthTrainerAgent(settings=settings, context=context or None)
    vitals = case.health_trainer_vitals or {}
    msg = (
        f"I'm {vitals.get('age', 25)} years old, {vitals.get('gender', 'Male')}. "
        f"Height: {vitals.get('height_cm', 170)} cm, "
        f"Weight: {vitals.get('weight_kg', 70)} kg. "
        f"I exercise {vitals.get('workout_frequency_per_week', 0)} days per week, "
        f"about {vitals.get('session_duration_hours', 0)} hours per session. "
        f"I have basic home dumbbells. No specific body part focus. No injuries."
    )
    agent.chat(msg)
    return agent.result(summary="Pipeline run")


# -- Mock I/O ----------------------------------------------------------------


def load_mock(case: EvalCase) -> dict[str, AgentResult]:
    """Load pre-recorded agent outputs for a case."""
    case_dir = MOCK_DIR / case.id
    results = {}
    for path in case_dir.glob("*.json"):
        data = json.loads(path.read_text())
        results[path.stem] = AgentResult.model_validate(data)
    return results


# -- Routing logic -----------------------------------------------------------


def route_decision(
    genomics: GenomicsFindings, doctor: DoctorFindings
) -> str:
    """2-layer decision matrix: DNA risk × clinical prediction."""
    dna_risk = "low" if genomics.predicted_class == "NONDM" else "high"
    matrix = {
        ("high", "Diabetic"): "hospital",
        ("high", "Non-Diabetic"): "hospital",
        ("low", "Diabetic"): "reconsider",
        ("low", "Non-Diabetic"): "health_trainer",
    }
    return matrix.get((dna_risk, doctor.prediction), "reconsider")


# -- Pretty printing ---------------------------------------------------------


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60)


def print_layer(num: int, name: str, result: AgentResult) -> None:
    status = "OK" if result.status.value == "success" else "ERR"
    print(f"\n--- Layer {num}: {name} [{status}] ---")
    print(f"  {result.summary}")
    if result.findings:
        findings = result.findings
        if isinstance(findings, GenomicsFindings):
            print(f"  Class: {findings.predicted_class} (confidence: {findings.confidence:.0%})")
            print(f"  Risk: {findings.risk_level.value}")
        elif isinstance(findings, DoctorFindings):
            print(f"  Prediction: {findings.prediction} (probability: {findings.probability:.0%})")
            print(f"  Risk: {findings.risk_level.value}")
            print(f"  Reasoning: {findings.reasoning}")
        elif isinstance(findings, TranscriptomicsFindings):
            print(f"  Confirmed: {findings.diabetes_confirmed.get('confirmed', '?')}")
            print(f"  Active pathways: {', '.join(findings.active_pathways)}")
            print(f"  Dominant: {findings.dominant_pathway}")
            print(f"  Risk: {findings.risk_level.value}")
        elif isinstance(findings, PharmacologyFindings):
            print(f"  Subtype: {findings.diabetes_subtype}")
            print(f"  Medications: {findings.medication_summary}")
            print(f"  Monitoring: {findings.monitoring_plan}")


def print_decision(decision: str, override: bool = False) -> None:
    label = {
        "hospital": "HOSPITAL — proceed to molecular validation + pharmacology",
        "reconsider": "RECONSIDER — lifestyle intervention, avoid unnecessary drugs",
        "health_trainer": "HEALTH TRAINER — prevention-focused exercise plan",
    }
    print(f"\n>>> DECISION: {label.get(decision, decision)}")
    if override:
        print("    (transcriptomics override: molecular evidence does not confirm)")


# -- Main pipeline -----------------------------------------------------------


async def run_pipeline(
    case: EvalCase, settings: Settings, mock: bool = False
) -> None:
    """Run the full E2E pipeline on a single case."""
    print_header(f"{case.id}: {case.name}")
    print(f"  {case.description}")

    mock_outputs = load_mock(case) if mock else {}

    # Build context dict incrementally
    context: dict = {}

    # Layer 1: Genomics (DNA classification)
    genomics_result = mock_outputs.get("genomics") or await run_genomics(case, settings)
    print_layer(1, "Genomics (DNA)", genomics_result)
    context["genomics"] = genomics_result.model_dump()

    # Layer 2: Doctor (clinical prediction)
    doctor_result = mock_outputs.get("doctor") or run_doctor(case, settings)
    print_layer(2, "Doctor (Clinical)", doctor_result)
    context["doctor"] = doctor_result.model_dump()

    # Routing decision
    if not isinstance(genomics_result.findings, GenomicsFindings):
        print("\n  ERROR: Genomics findings missing, cannot route.")
        return
    if not isinstance(doctor_result.findings, DoctorFindings):
        print("\n  ERROR: Doctor findings missing, cannot route.")
        return

    decision = route_decision(genomics_result.findings, doctor_result.findings)
    print_decision(decision)

    # Layer 3+: Hospital path or Health Trainer path
    if decision == "hospital" and case.gene_expression:
        # Transcriptomics (molecular confirmation)
        tx_result = (
            mock_outputs.get("transcriptomics")
            or await run_transcriptomics(case, settings, context)
        )
        print_layer(3, "Transcriptomics (Molecular)", tx_result)
        context["transcriptomics"] = tx_result.model_dump()

        # Check if transcriptomics confirms
        if isinstance(tx_result.findings, TranscriptomicsFindings):
            confirmed = tx_result.findings.diabetes_confirmed.get("confirmed", False)
            if confirmed:
                # Pharmacology (drug recommendation)
                pharma_result = (
                    mock_outputs.get("pharmacology")
                    or run_pharmacology(settings, context, tx_result.findings)
                )
                print_layer(4, "Pharmacology (Treatment)", pharma_result)
                print(f"\n  FINAL: Hospital pathway complete — medication plan generated.")
            else:
                # False positive — override to health trainer
                print_decision("health_trainer", override=True)
                ht_result = (
                    mock_outputs.get("health_trainer")
                    or run_health_trainer(case, settings, context)
                )
                print_layer(4, "Health Trainer (Redirected)", ht_result)
        else:
            print("\n  WARNING: Transcriptomics findings missing, skipping pharmacology.")

    elif decision == "hospital":
        print("\n  NOTE: No gene expression data — transcriptomics skipped.")
        print("  Hospital referral based on 2-layer validation (DNA + Clinical).")

    elif decision in ("health_trainer", "reconsider"):
        ht_result = (
            mock_outputs.get("health_trainer")
            or run_health_trainer(case, settings, context)
        )
        print_layer(3, "Health Trainer (Prevention)", ht_result)
        print(f"\n  FINAL: {decision.replace('_', ' ').title()} pathway complete.")


async def main():
    parser = argparse.ArgumentParser(description="BioAI E2E Pipeline")
    parser.add_argument(
        "--case", type=int, help="Case number to run (1-4)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available cases"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all cases"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use pre-recorded outputs (no API calls)"
    )
    args = parser.parse_args()

    cases = load_cases()

    if args.list:
        print("Available cases:")
        for i, c in enumerate(cases, 1):
            print(f"  {i}. [{c.id}] {c.name} — {c.description[:60]}...")
        return

    settings = Settings.from_env()

    if args.mock:
        print("[MOCK MODE — using pre-recorded outputs]\n")

    if args.all:
        for case in cases:
            await run_pipeline(case, settings, mock=args.mock)
    elif args.case:
        if args.case < 1 or args.case > len(cases):
            print(f"Invalid case number. Use 1-{len(cases)}.")
            return
        await run_pipeline(cases[args.case - 1], settings, mock=args.mock)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
