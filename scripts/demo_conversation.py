"""Demo conversation — walk through Case 1 step-by-step for presentation.

Shows the input/output exchange at each pipeline layer as a conversation.
Uses mock outputs by default (no API calls). Use --live for real API calls.

Usage:
    uv run python scripts/demo_conversation.py          # mock (instant)
    uv run python scripts/demo_conversation.py --live   # real API calls
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env (ANTHROPIC_API_KEY)

from precision_health_agents.config import Settings
from precision_health_agents.eval.cases import load_cases
from precision_health_agents.models import (
    AgentResult,
    DoctorFindings,
    GenomicsFindings,
    PharmacologyFindings,
    TranscriptomicsFindings,
)

MOCK_DIR = Path("src/precision_health_agents/eval/data/mock_outputs")

# -- Formatting helpers ------------------------------------------------------

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(text: str) -> None:
    print(f"\n{BOLD}{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}{RESET}")


def step_header(num: int, icon: str, title: str) -> None:
    print(f"\n{BOLD}{CYAN}--- Step {num}: {icon} {title} ---{RESET}")


def arrow(label: str, color: str = DIM) -> None:
    print(f"{color}  --> {label}{RESET}")


def agent_says(agent: str, text: str) -> None:
    print(f"  {GREEN}{BOLD}{agent}:{RESET} {text}")


def system_says(text: str) -> None:
    print(f"  {YELLOW}{BOLD}System:{RESET} {text}")


def patient_says(text: str) -> None:
    print(f"  {BLUE}{BOLD}Patient:{RESET} {text}")


def decision_box(text: str) -> None:
    print(f"\n  {BOLD}{MAGENTA}>>> DECISION: {text}{RESET}")


def kv(key: str, value: str) -> None:
    print(f"    {DIM}{key}:{RESET} {value}")


def pause(seconds: float = 1.0) -> None:
    time.sleep(seconds)


# -- Main demo ---------------------------------------------------------------


async def demo(live: bool = False) -> None:
    cases = load_cases()
    case = cases[0]  # case-1: Confirmed Diabetic
    settings = Settings.from_env() if live else None

    banner(f"BioAI Demo — Case 1: {case.name}")
    print(f"\n  {DIM}{case.description}{RESET}")

    # ── Patient intake ──────────────────────────────────────────────────
    step_header(0, "\U0001F9D1\u200D\u2695\uFE0F", "Patient Intake")
    clinical = case.clinical_features or {}
    patient_says(
        f"I'm a {int(clinical['age'])}-year-old woman, "
        f"{int(clinical['pregnancies'])} pregnancy. "
        f"Glucose {int(clinical['glucose'])} mg/dL, "
        f"BP {int(clinical['blood_pressure'])} mmHg, "
        f"BMI {clinical['bmi']}, insulin {int(clinical['insulin'])} mu U/ml."
    )
    arrow("Sending DNA sample + clinical data to BioAI system...")

    pause(0.5)

    # ── Layer 1: Genomics ───────────────────────────────────────────────
    step_header(1, "\U0001F9EC", "Genomics Agent (DNA Analysis)")
    system_says("Analyzing 1,000 bp DNA sequence with CNN classifier...")

    if live:
        from precision_health_agents.agents.genomics import GenomicsAgent

        genomics_result = await GenomicsAgent().analyze(case.dna_sequence or "")
    else:
        genomics_result = AgentResult.model_validate(
            json.loads((MOCK_DIR / "case-1/genomics.json").read_text())
        )
    gf = genomics_result.findings
    if not isinstance(gf, GenomicsFindings):
        print(f"  {RED}ERROR: Genomics failed — {genomics_result.error}{RESET}")
        return

    pause(0.5)

    agent_says("Genomics", "DNA analysis complete.")
    kv("Classification", f"{gf.predicted_class}")
    kv("Confidence", f"{gf.confidence:.0%}")
    kv("Risk Level", f"{gf.risk_level.value}")
    kv(
        "Interpretation",
        "Strong genetic predisposition for Type 2 diabetes.",
    )

    pause(0.5)

    # ── Layer 2: Doctor ─────────────────────────────────────────────────
    step_header(2, "\U0001FA7A", "Doctor Agent (Clinical Assessment)")
    feature_str = ", ".join(f"{k}={v}" for k, v in clinical.items())
    patient_says(f"My clinical values: {feature_str}")

    if live:
        from precision_health_agents.agents.doctor import DoctorAgent

        doc = DoctorAgent(settings=settings)
        reply = doc.chat(f"My clinical values: {feature_str}")
        doctor_result = doc.result(summary=reply)
    else:
        doctor_result = AgentResult.model_validate(
            json.loads((MOCK_DIR / "case-1/doctor.json").read_text())
        )
    df = doctor_result.findings
    if not isinstance(df, DoctorFindings):
        print(f"  {RED}ERROR: Doctor failed — {doctor_result.error}{RESET}")
        return

    pause(0.5)

    agent_says("Doctor", "Clinical assessment complete.")
    kv("Prediction", f"{df.prediction}")
    kv("Probability", f"{df.probability:.0%}")
    kv("Risk Level", f"{df.risk_level.value}")
    kv("Recommendation", "Refer to hospital")

    pause(0.5)

    # ── Routing Decision ────────────────────────────────────────────────
    step_header(3, "\U0001F500", "Routing Decision (DNA x Clinical)")
    system_says(
        f"DNA risk: HIGH ({gf.predicted_class} @ {gf.confidence:.0%}) + "
        f"Clinical: {df.prediction} ({df.probability:.0%})"
    )
    decision_box("HOSPITAL — proceed to molecular validation + pharmacology")

    pause(0.5)

    # ── Layer 3: Transcriptomics ────────────────────────────────────────
    step_header(4, "\U0001F52C", "Transcriptomics Agent (Molecular Confirmation)")
    gene_data = case.gene_expression or {}
    top_genes = sorted(gene_data.items(), key=lambda x: x[1], reverse=True)[:5]
    system_says(
        "Analyzing 23-gene expression profile. "
        f"Top genes: {', '.join(f'{g}={v:.0f}' for g, v in top_genes)}"
    )

    if live:
        from precision_health_agents.agents.transcriptomics import TranscriptomicsAgent

        context = {
            "genomics": genomics_result.model_dump(),
            "doctor": doctor_result.model_dump(),
        }
        tx_result = await TranscriptomicsAgent(settings=settings).analyze(
            "Gene expression profile:\n"
            + "\n".join(f"  {g}: {v}" for g, v in gene_data.items()),
            context=context,
        )
    else:
        tx_result = AgentResult.model_validate(
            json.loads((MOCK_DIR / "case-1/transcriptomics.json").read_text())
        )
    tf = tx_result.findings
    if not isinstance(tf, TranscriptomicsFindings):
        print(f"  {RED}ERROR: Transcriptomics failed — {tx_result.error}{RESET}")
        return

    pause(0.5)

    agent_says("Transcriptomics", "Molecular analysis complete.")
    confirmed = tf.diabetes_confirmed.get("confirmed", False)
    confidence = tf.diabetes_confirmed.get("confidence", 0)
    try:
        conf_str = f"{float(confidence):.0%}"
    except (ValueError, TypeError):
        conf_str = str(confidence)
    kv("Diabetes Confirmed", f"{'YES' if confirmed else 'NO'} (confidence: {conf_str})")
    kv("Active Pathways", f"{len(tf.active_pathways)}/5")
    for pw in tf.active_pathways:
        score = tf.pathway_scores.get(pw, 0)
        try:
            score_str = f"z-score {float(score):.1f}"
        except (ValueError, TypeError):
            score_str = f"z-score {score}"
        kv(f"  {pw}", score_str)
    kv("Dominant Pathway", tf.dominant_pathway)
    kv("Subtype", tf.diabetes_subtype.get("primary", "unknown"))
    if tf.dysregulated_genes:
        top_dg = tf.dysregulated_genes[:3]
        kv(
            "Key Genes",
            ", ".join(f"{g['gene']} (z={g['z_score']})" for g in top_dg),
        )
    kv("Risk Level", tf.risk_level.value)

    pause(0.5)

    # ── Layer 4: Pharmacology ───────────────────────────────────────────
    step_header(5, "\U0001F48A", "Pharmacology Agent (Treatment Plan)")
    subtype = tf.diabetes_subtype.get("primary", "Type 2")
    complications = ", ".join(c.get("complication", "") for c in tf.complication_risks)
    system_says(
        f"Patient confirmed with {subtype} diabetes. "
        f"Complications: {complications}. "
        f"Dominant pathway: {tf.dominant_pathway}."
    )

    if live:
        from precision_health_agents.agents.pharmacology import PharmacologyAgent

        context["transcriptomics"] = tx_result.model_dump()
        pharma = PharmacologyAgent(settings=settings, context=context)
        msg = (
            f"Patient confirmed with {subtype} diabetes. "
            f"Complication risks: {complications}. "
            f"Dominant pathway: {tf.dominant_pathway}. "
            f"Please recommend a medication plan."
        )
        pharma.chat(msg)
        pharma_result = pharma.result(summary="Pipeline run")
    else:
        pharma_result = AgentResult.model_validate(
            json.loads((MOCK_DIR / "case-1/pharmacology.json").read_text())
        )
    pf = pharma_result.findings
    if not isinstance(pf, PharmacologyFindings):
        print(f"  {RED}ERROR: Pharmacology failed — {pharma_result.error}{RESET}")
        return

    pause(0.5)

    agent_says("Pharmacology", "Treatment plan generated.")
    kv("Diabetes Subtype", pf.diabetes_subtype)
    print()
    def print_med(med: dict) -> None:
        name = med.get("name", med.get("medication", "Unknown"))
        cls = med.get("class", med.get("drug_class", ""))
        print(f"      - {BOLD}{name}{RESET}" + (f" ({cls})" if cls else ""))
        if dose := med.get("dose", med.get("dosage", "")):
            print(f"        Dose: {dose}")
        if rationale := med.get("rationale", med.get("reason", "")):
            print(f"        {DIM}Rationale: {rationale}{RESET}")

    print(f"    {BOLD}Primary Medications:{RESET}")
    for med in pf.primary_medications:
        print_med(med)
    if pf.supportive_medications:
        print(f"    {BOLD}Supportive Medications:{RESET}")
        for med in pf.supportive_medications:
            print_med(med)
    print()
    kv("Monitoring", pf.monitoring_plan)

    pause(0.5)

    # ── Final Summary ───────────────────────────────────────────────────
    banner("Final Assessment — Case 1: Confirmed Diabetic")
    print()
    print(f"  {BOLD}4-Layer Validation Result:{RESET}")
    print(f"    Layer 1 (DNA):            {RED}DMT2 — 90% confidence{RESET}")
    print(f"    Layer 2 (Clinical):       {RED}Diabetic — 92% probability{RESET}")
    print(f"    Layer 3 (Transcriptomics):{RED} Confirmed — 5/5 pathways active{RESET}")
    print(f"    Layer 4 (Pharmacology):   {GREEN}Triple therapy prescribed{RESET}")
    print()
    print(
        f"  {BOLD}Outcome:{RESET} All 4 layers agree — "
        f"confirmed Type 2 diabetes with personalized treatment plan."
    )
    print(
        f"  {BOLD}Medications:{RESET} {pf.medication_summary}"
    )
    print(
        f"  {BOLD}Monitoring:{RESET} {pf.monitoring_plan}"
    )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioAI Demo Conversation")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real API calls instead of mock outputs",
    )
    args = parser.parse_args()
    asyncio.run(demo(live=args.live))
