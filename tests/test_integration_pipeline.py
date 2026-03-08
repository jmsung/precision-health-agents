"""End-to-end integration tests for the 5 pipeline scenarios.

Each test simulates a full patient journey through the multi-agent system,
with all agents mocked at the Anthropic API level while real tools execute.

Scenarios (based on whether diabetes is confirmed or not):
  1. DNA(DMT2) -> Doctor(Diabetic) -> Transcriptomics(NOT confirmed) -> HealthTrainer
     False positive: molecular evidence doesn't support diabetes -> lifestyle, not drugs.
  2. DNA(DMT2) -> Doctor(Diabetic) -> Transcriptomics(CONFIRMED) -> Pharmacology
     True positive: all 3 layers agree -> proceed to medication.
  3. DNA(NONDM) -> Doctor(Non-Diabetic) -> HealthTrainer
     No diabetes: genetics + clinical both clear -> prevention/lifestyle.
  4. DNA(DMT2) -> Doctor(Diabetic) -> Hospital
     Two-layer confirmed: awaiting transcriptomics (hospital pathway entry).
  5. DNA(DMT2) -> Doctor(Diabetic) -> Transcriptomics(CONFIRMED) -> Pharmacology (full pipeline)
     Complete 4-agent pipeline with medication recommendation.
"""

import asyncio
from unittest.mock import MagicMock, patch

from bioai.agents.doctor import DoctorAgent
from bioai.agents.genomics import GenomicsAgent
from bioai.agents.health_trainer import HealthTrainerAgent
from bioai.agents.pharmacology import PharmacologyAgent
from bioai.agents.transcriptomics import TranscriptomicsAgent
from bioai.models import AgentStatus, PharmacologyFindings, Recommendation, RiskLevel
from bioai.tools.gene_expression_analyzer import PATHWAY_GENES, _get_reference_stats


# ---------------------------------------------------------------------------
# Shared mock helpers
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
# Shared patient data
# ---------------------------------------------------------------------------

_DNA_SEQUENCE = "ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

_CLINICAL_DIABETIC = {
    "pregnancies": 2, "glucose": 160, "blood_pressure": 82,
    "skin_thickness": 30, "insulin": 0, "bmi": 31.5,
    "diabetes_pedigree_function": 0.5, "age": 42,
}

_CLINICAL_HEALTHY = {
    "pregnancies": 0, "glucose": 89, "blood_pressure": 70,
    "skin_thickness": 20, "insulin": 80, "bmi": 22.1,
    "diabetes_pedigree_function": 0.2, "age": 25,
}

_HT_VITALS = {
    "age": 42, "gender": "Female", "weight_kg": 89, "height_cm": 168,
    "workout_frequency_per_week": 1, "session_duration_hours": 0.5,
}

_HT_CLASSIFY_RESULT = {
    "suggested_type": "Cardio",
    "experience_level": "Beginner",
    "bmi": 31.5,
    "all_scores": {"Cardio": 1.8, "Strength": 1.6, "Flexibility": 1.2, "HIIT": 0.6},
    "reasoning": "ADA recommends aerobic exercise for diabetes management.",
}

_HT_EXERCISES = [
    {"Name": "Walking", "Type": "Cardio", "BodyPart": "Full Body",
     "Equipment": "None", "Level": "Beginner",
     "Description": "Brisk walking", "Benefits": "Cardio health",
     "CaloriesPerMinute": 5},
]


# ---------------------------------------------------------------------------
# Helper: build gene expression data
# ---------------------------------------------------------------------------

def _build_gene_expression(activated_pathways: list[str]) -> dict[str, float]:
    """Build gene expression dict with specified pathways activated."""
    ref_stats = _get_reference_stats()
    expr = {}
    for pathway in activated_pathways:
        for gene in PATHWAY_GENES[pathway]:
            if gene in ref_stats:
                mean, std = ref_stats[gene]
                expr[gene] = mean + 2.0 * std  # strong upregulation
    return expr


def _build_normal_gene_expression() -> dict[str, float]:
    """Build gene expression at normal/mean levels (no pathway activation)."""
    ref_stats = _get_reference_stats()
    return {gene: mean for gene, (mean, _) in ref_stats.items()}


# ---------------------------------------------------------------------------
# Helper: run genomics agent
# ---------------------------------------------------------------------------

def _run_genomics(predicted_class: str, confidence: float) -> MagicMock:
    """Run mocked GenomicsAgent, return result."""
    probs = {"DMT1": 0.05, "DMT2": 0.05, "NONDM": 0.05}
    probs[predicted_class] = confidence

    with (
        patch("bioai.agents.genomics.anthropic.Anthropic"),
        patch("bioai.agents.genomics.classify_dna") as mock_classify,
    ):
        mock_classify.return_value = {
            "predicted_class": predicted_class,
            "probabilities": probs,
            "confidence": confidence,
        }

        agent = GenomicsAgent()
        client = MagicMock()
        agent._client = client
        client.messages.create.side_effect = [
            _mock_tool_use("classify_dna", {"sequence": _DNA_SEQUENCE}),
            _mock_text(f"DNA analysis: {predicted_class} with {confidence:.0%} confidence."),
        ]

        result = asyncio.run(agent.analyze(
            f"Analyze this DNA sequence: {_DNA_SEQUENCE}"
        ))

    return result


# ---------------------------------------------------------------------------
# Helper: run doctor agent
# ---------------------------------------------------------------------------

def _run_doctor(clinical_values: dict, prediction: str, probability: float):
    """Run mocked DoctorAgent through conversation, return (agent, result).

    chat() calls _run() which loops until end_turn. So:
    - Turn 1: patient speaks -> Claude replies with text (end_turn)
    - Turn 2: patient speaks -> Claude calls tool (tool_use) -> tool result
              -> Claude replies with final text (end_turn)
    """
    risk = "high" if probability > 0.5 else "low"

    with (
        patch("bioai.agents.doctor.anthropic.Anthropic"),
        patch("bioai.agents.doctor.classify_diabetes") as mock_classify,
    ):
        mock_classify.return_value = {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk,
        }

        agent = DoctorAgent()
        client = MagicMock()
        agent._client = client

        # Turn 1: text reply. Turn 2: tool_use -> loops -> text reply.
        client.messages.create.side_effect = [
            _mock_text("Hello! Tell me about your health."),
            _mock_tool_use("classify_diabetes", clinical_values, "doc_tool_1"),
            _mock_text(f"Assessment complete: {prediction} ({probability:.0%})."),
        ]

        agent.chat("Hi doctor, I need a diabetes check.")
        agent.chat(f"I'm {clinical_values['age']}, glucose {clinical_values['glucose']}.")

        result = agent.result(f"Assessment: {prediction} ({probability:.0%})")

    return agent, result


# ---------------------------------------------------------------------------
# Helper: run transcriptomics agent
# ---------------------------------------------------------------------------

def _run_transcriptomics(gene_expression: dict, context: dict):
    """Run mocked TranscriptomicsAgent (real tool, mocked Claude)."""
    agent = TranscriptomicsAgent(settings=MagicMock(
        api_key="test-key", agent_model="test-model", max_tokens=1024,
    ))
    client = MagicMock()
    agent._client = client

    # Claude asks patient for data, then calls tool, then summarizes
    client.messages.create.side_effect = [
        _mock_text("I'll need your gene expression data to analyze diabetes pathways."),
        _mock_tool_use("analyze_gene_expression", {"gene_expression": gene_expression}),
        _mock_text("Analysis complete. Here are the pathway results."),
    ]

    # First call: agent asks for data (conversational)
    result1 = asyncio.run(agent.analyze(
        "I've been referred to hospital. What do you need?",
        context=context,
    ))

    # Second call: patient provides gene expression data, tool is called
    agent2 = TranscriptomicsAgent(settings=MagicMock(
        api_key="test-key", agent_model="test-model", max_tokens=1024,
    ))
    agent2._client = MagicMock()
    agent2._client.messages.create.side_effect = [
        _mock_tool_use("analyze_gene_expression", {"gene_expression": gene_expression}),
        _mock_text("Analysis complete. Here are the pathway results."),
    ]

    result = asyncio.run(agent2.analyze(
        f"Here is my gene expression data: {gene_expression}",
        context=context,
    ))

    return result


# ---------------------------------------------------------------------------
# Helper: run health trainer agent
# ---------------------------------------------------------------------------

def _run_health_trainer(context: dict):
    """Run mocked HealthTrainerAgent, return result."""
    with (
        patch("bioai.agents.health_trainer.anthropic.Anthropic"),
        patch("bioai.agents.health_trainer.classify_workout_type") as mock_classify,
        patch("bioai.agents.health_trainer.recommend_exercises") as mock_recommend,
    ):
        mock_classify.return_value = _HT_CLASSIFY_RESULT
        mock_recommend.return_value = {
            "exercises": _HT_EXERCISES, "total_found": 1, "filters_applied": {},
        }

        agent = HealthTrainerAgent(context=context)
        agent._client = MagicMock()

        ht_input = {**_HT_VITALS, "diabetes_type": "NONDM", "diabetes_probability": 0.0}
        agent._client.messages.create.side_effect = [
            _mock_tool_use("classify_workout_type", ht_input, "ht_1"),
            _mock_tool_use("recommend_exercises", {"exercise_type": "Cardio", "difficulty": "Beginner"}, "ht_2"),
            _mock_text("Here is your personalised exercise plan for healthy living."),
        ]

        agent.chat("I've been referred for exercise guidance.")

        return agent.result()


# ---------------------------------------------------------------------------
# Helper: run pharmacology agent
# ---------------------------------------------------------------------------

def _run_pharmacology(context: dict):
    """Run mocked PharmacologyAgent (real tool, mocked Claude)."""
    trans_findings = context.get("transcriptomics", {}).get("findings", {})
    subtype = trans_findings.get("diabetes_subtype", {}).get("subtype", "mixed")
    complications = trans_findings.get("complication_risks", [])

    with patch("bioai.agents.pharmacology.anthropic.Anthropic"):
        agent = PharmacologyAgent(context=context)
        agent._client = MagicMock()

        agent._client.messages.create.side_effect = [
            _mock_tool_use("recommend_medications", {
                "diabetes_subtype": subtype,
                "complication_risks": complications,
            }),
            _mock_text(
                "Based on your molecular profile, here is your personalized medication plan. "
                "I've selected medications that target your specific diabetes subtype and "
                "address your complication risks. Remember, this is a starting point — "
                "your care team will adjust as needed."
            ),
        ]

        agent.chat("What medications do you recommend for my diabetes?")
        return agent.result()


# ===========================================================================
# Test 1: Full pipeline — false positive (DNA+Doctor say diabetes,
#          Transcriptomics says NO) -> HealthTrainer
# ===========================================================================

def test_pipeline_false_positive_to_health_trainer():
    """DMT2 DNA + Diabetic clinical, but transcriptomics finds no pathway
    activation -> false positive -> route to Health Trainer, not drugs."""

    print("\n" + "=" * 70)
    print("TEST 1: False Positive — DNA+Doctor -> Transcriptomics(NO) -> Trainer")
    print("=" * 70)

    # Step 1: Genomics — DMT2 detected
    genomics_result = _run_genomics("DMT2", 0.88)
    assert genomics_result.status == AgentStatus.SUCCESS
    assert genomics_result.findings.predicted_class == "DMT2"
    assert genomics_result.findings.risk_level == RiskLevel.HIGH
    print(f"[Genomics] DMT2 detected ({genomics_result.findings.confidence:.0%})")

    # Step 2: Doctor — Diabetic confirmed clinically
    doctor_agent, doctor_result = _run_doctor(_CLINICAL_DIABETIC, "Diabetic", 0.76)
    assert doctor_result.findings.prediction == "Diabetic"
    assert doctor_result.findings.recommendation == Recommendation.HOSPITAL
    print(f"[Doctor] Diabetic ({doctor_result.findings.probability:.0%}) -> Hospital")

    # Routing: DNA=DMT2 + Clinical=Diabetic -> Hospital pathway
    context = {
        "genomics": {"predicted_class": "DMT2", "confidence": 0.88},
        "doctor": {"prediction": "Diabetic", "probability": 0.76},
    }

    # Step 3: Transcriptomics — normal gene expression (false positive!)
    normal_expr = _build_normal_gene_expression()
    trans_result = _run_transcriptomics(normal_expr, context)

    assert trans_result.status == AgentStatus.SUCCESS
    assert trans_result.findings is not None
    assert trans_result.findings.diabetes_confirmed["confirmed"] is False
    assert trans_result.findings.recommendation == "health_trainer"
    print(f"[Transcriptomics] NOT confirmed -> false positive -> Health Trainer")

    # Step 4: Health Trainer — lifestyle intervention (not drugs)
    ht_context = {
        **context,
        "transcriptomics": {
            "diabetes_confirmed": False,
            "recommendation": "health_trainer",
        },
    }
    ht_result = _run_health_trainer(ht_context)
    assert ht_result.status == AgentStatus.SUCCESS
    assert ht_result.findings is not None
    print(f"[HealthTrainer] Exercise plan delivered")

    print("[Decision] 3-layer validation caught false positive -> no drugs needed")
    print("=" * 70)


# ===========================================================================
# Test 2: Full pipeline — true positive (all 3 layers agree)
#          -> Pharmacology (routing verified)
# ===========================================================================

def test_pipeline_confirmed_to_pharmacology():
    """DMT2 DNA + Diabetic clinical + active pathways in transcriptomics
    -> confirmed diabetes -> route to Pharmacology."""

    print("\n" + "=" * 70)
    print("TEST 2: Confirmed — DNA+Doctor+Transcriptomics -> Pharmacology")
    print("=" * 70)

    # Step 1: Genomics
    genomics_result = _run_genomics("DMT2", 0.92)
    assert genomics_result.findings.predicted_class == "DMT2"
    print(f"[Genomics] DMT2 detected ({genomics_result.findings.confidence:.0%})")

    # Step 2: Doctor
    _, doctor_result = _run_doctor(_CLINICAL_DIABETIC, "Diabetic", 0.82)
    assert doctor_result.findings.recommendation == Recommendation.HOSPITAL
    print(f"[Doctor] Diabetic ({doctor_result.findings.probability:.0%}) -> Hospital")

    # Step 3: Transcriptomics — active pathways (confirmed!)
    context = {
        "genomics": {"predicted_class": "DMT2", "confidence": 0.92},
        "doctor": {"prediction": "Diabetic", "probability": 0.82},
    }
    activated_expr = _build_gene_expression(["inflammation_immune", "beta_cell_stress"])
    trans_result = _run_transcriptomics(activated_expr, context)

    assert trans_result.findings.diabetes_confirmed["confirmed"] is True
    assert trans_result.findings.recommendation == "pharmacology"
    assert len(trans_result.findings.active_pathways) >= 2
    print(f"[Transcriptomics] CONFIRMED — {trans_result.findings.diabetes_subtype['subtype']}")
    print(f"  Active pathways: {trans_result.findings.active_pathways}")
    print(f"  Complications: {[r['complication'] for r in trans_result.findings.complication_risks]}")

    # Step 4: Pharmacology — medication recommendation
    pharma_context = {
        "genomics": genomics_result.model_dump(),
        "doctor": doctor_result.model_dump(),
        "transcriptomics": trans_result.model_dump(),
    }
    pharma_result = _run_pharmacology(pharma_context)

    assert pharma_result.status == AgentStatus.SUCCESS
    assert pharma_result.findings is not None
    assert isinstance(pharma_result.findings, PharmacologyFindings)
    assert len(pharma_result.findings.primary_medications) + len(pharma_result.findings.supportive_medications) > 0
    print(f"[Pharmacology] Medications recommended:")
    for med in pharma_result.findings.primary_medications:
        print(f"  Primary: {med['name']} ({med['class']}) — {', '.join(med['reasons'])}")
    for med in pharma_result.findings.supportive_medications:
        print(f"  Supportive: {med['name']} ({med['class']})")

    print("[Decision] All 3 layers confirm -> Pharmacology delivered medication plan")
    print("=" * 70)


# ===========================================================================
# Test 3: No diabetes — NONDM + Non-Diabetic -> HealthTrainer directly
# ===========================================================================

def test_pipeline_nondm_to_health_trainer():
    """NONDM DNA + Non-Diabetic clinical -> skip hospital -> HealthTrainer."""

    print("\n" + "=" * 70)
    print("TEST 3: No Diabetes — NONDM + Non-Diabetic -> Health Trainer")
    print("=" * 70)

    # Step 1: Genomics — NONDM
    genomics_result = _run_genomics("NONDM", 0.91)
    assert genomics_result.findings.predicted_class == "NONDM"
    assert genomics_result.findings.risk_level == RiskLevel.LOW
    print(f"[Genomics] NONDM ({genomics_result.findings.confidence:.0%}) -> no genetic risk")

    # Step 2: Doctor — Non-Diabetic
    _, doctor_result = _run_doctor(_CLINICAL_HEALTHY, "Non-Diabetic", 0.25)
    assert doctor_result.findings.prediction == "Non-Diabetic"
    assert doctor_result.findings.recommendation == Recommendation.HEALTH_TRAINER
    print(f"[Doctor] Non-Diabetic -> Health Trainer")

    # No hospital pathway — skip transcriptomics, go straight to Health Trainer
    context = {
        "genomics": {"predicted_class": "NONDM", "confidence": 0.91},
        "doctor": {"prediction": "Non-Diabetic", "probability": 0.25},
    }
    ht_result = _run_health_trainer(context)
    assert ht_result.status == AgentStatus.SUCCESS
    assert ht_result.findings is not None
    print(f"[HealthTrainer] Prevention exercise plan delivered")

    print("[Decision] No diabetes risk -> lifestyle prevention")
    print("=" * 70)


# ===========================================================================
# Test 4: Two-layer confirmed — DNA + Doctor -> Hospital (entry point)
# ===========================================================================

def test_pipeline_dna_doctor_to_hospital():
    """DMT2 DNA + Diabetic clinical -> Hospital pathway entry.
    Transcriptomics would follow but is not run here."""

    print("\n" + "=" * 70)
    print("TEST 4: DNA + Doctor -> Hospital (awaiting Transcriptomics)")
    print("=" * 70)

    # Step 1: Genomics
    genomics_result = _run_genomics("DMT2", 0.85)
    assert genomics_result.findings.predicted_class == "DMT2"
    assert genomics_result.findings.risk_level == RiskLevel.HIGH
    print(f"[Genomics] DMT2 ({genomics_result.findings.confidence:.0%})")

    # Step 2: Doctor
    _, doctor_result = _run_doctor(_CLINICAL_DIABETIC, "Diabetic", 0.78)
    assert doctor_result.findings.prediction == "Diabetic"
    assert doctor_result.findings.recommendation == Recommendation.HOSPITAL
    print(f"[Doctor] Diabetic ({doctor_result.findings.probability:.0%})")

    # Routing decision
    dna_class = genomics_result.findings.predicted_class
    clinical = doctor_result.findings.prediction
    assert dna_class in ("DMT1", "DMT2")
    assert clinical == "Diabetic"

    print(f"[Decision] DNA={dna_class} + Clinical={clinical} -> HOSPITAL")
    print("  Next step: Transcriptomics Agent for molecular confirmation")
    print("=" * 70)


# ===========================================================================
# Test 5: Complete 4-agent pipeline — DNA -> Doctor -> Transcriptomics
#          -> Pharmacology (full end-to-end with medication output)
# ===========================================================================

def test_pipeline_full_four_agent_to_medication():
    """Complete pipeline: DMT2 DNA + Diabetic clinical + confirmed transcriptomics
    -> PharmacologyAgent delivers personalized medication plan."""

    print("\n" + "=" * 70)
    print("TEST 5: Full 4-Agent Pipeline -> Medication Plan")
    print("=" * 70)

    # Step 1: Genomics — DMT2
    genomics_result = _run_genomics("DMT2", 0.90)
    assert genomics_result.status == AgentStatus.SUCCESS
    assert genomics_result.findings.predicted_class == "DMT2"
    print(f"[Genomics] DMT2 detected ({genomics_result.findings.confidence:.0%})")

    # Step 2: Doctor — Diabetic
    _, doctor_result = _run_doctor(_CLINICAL_DIABETIC, "Diabetic", 0.80)
    assert doctor_result.findings.prediction == "Diabetic"
    assert doctor_result.findings.recommendation == Recommendation.HOSPITAL
    print(f"[Doctor] Diabetic ({doctor_result.findings.probability:.0%}) -> Hospital")

    # Step 3: Transcriptomics — confirmed with inflammation + insulin resistance
    context_for_trans = {
        "genomics": {"predicted_class": "DMT2", "confidence": 0.90},
        "doctor": {"prediction": "Diabetic", "probability": 0.80},
    }
    activated_expr = _build_gene_expression(["inflammation_immune", "insulin_resistance"])
    trans_result = _run_transcriptomics(activated_expr, context_for_trans)

    assert trans_result.findings.diabetes_confirmed["confirmed"] is True
    assert trans_result.findings.recommendation == "pharmacology"
    subtype = trans_result.findings.diabetes_subtype["subtype"]
    complications = trans_result.findings.complication_risks
    print(f"[Transcriptomics] CONFIRMED — subtype={subtype}")
    print(f"  Active pathways: {trans_result.findings.active_pathways}")
    print(f"  Complications: {[r['complication'] for r in complications]}")

    # Step 4: Pharmacology — medication plan
    pharma_context = {
        "genomics": genomics_result.model_dump(),
        "doctor": doctor_result.model_dump(),
        "transcriptomics": trans_result.model_dump(),
    }
    pharma_result = _run_pharmacology(pharma_context)

    assert pharma_result.status == AgentStatus.SUCCESS
    assert pharma_result.agent == "pharmacology"
    findings = pharma_result.findings
    assert findings is not None
    assert isinstance(findings, PharmacologyFindings)
    assert findings.diabetes_subtype != "unknown"

    total_meds = len(findings.primary_medications) + len(findings.supportive_medications)
    assert total_meds > 0
    assert findings.medication_summary  # non-empty plan text
    assert findings.monitoring_plan  # non-empty monitoring

    print(f"[Pharmacology] Subtype: {findings.diabetes_subtype}")
    print(f"  Primary medications ({len(findings.primary_medications)}):")
    for med in findings.primary_medications:
        print(f"    - {med['name']} ({med['class']}): {', '.join(med['reasons'])}")
    print(f"  Supportive medications ({len(findings.supportive_medications)}):")
    for med in findings.supportive_medications:
        print(f"    - {med['name']} ({med['class']}): {', '.join(med['reasons'])}")
    print(f"  Monitoring: {findings.monitoring_plan}")

    print("\n[Summary] Full pipeline complete:")
    print(f"  Layer 1 (DNA):            DMT2 -> genetic risk confirmed")
    print(f"  Layer 2 (Clinical):       Diabetic -> clinical risk confirmed")
    print(f"  Layer 3 (Transcriptomics): {subtype} -> molecular confirmation")
    print(f"  Layer 4 (Pharmacology):   {total_meds} medications recommended")
    print("=" * 70)
