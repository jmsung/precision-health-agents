"""Evaluation CLI — score agents and optionally run Ralph Loop.

Usage:
    uv run python scripts/evaluate.py              # real mode
    uv run python scripts/evaluate.py --mock       # mock mode (pre-recorded outputs)
    uv run python scripts/evaluate.py --save       # real mode + save outputs for mock
    uv run python scripts/evaluate.py --ralph --iter 3  # Ralph Loop, 3 iterations
"""

import argparse
import asyncio
import json
from pathlib import Path

from precision_health_agents.config import Settings
from precision_health_agents.eval.cases import EvalCase, load_cases
from precision_health_agents.eval.judge import JudgeScore, judge_agent
from precision_health_agents.eval.metrics import MetricResult, score_decision, score_tool_accuracy
from precision_health_agents.eval.ralph import FailureExample, RalphResult, ralph_iterate
from precision_health_agents.models import AgentResult

MOCK_DIR = Path("src/precision_health_agents/eval/data/mock_outputs")


# -- Agent runners (real mode) ------------------------------------------------


async def run_genomics(case: EvalCase, settings: Settings) -> AgentResult:
    """Run Genomics agent on a test case."""
    from precision_health_agents.agents.genomics import GenomicsAgent

    agent = GenomicsAgent()
    return await agent.analyze(case.dna_sequence or "")


async def run_doctor(case: EvalCase, settings: Settings) -> AgentResult:
    """Run Doctor agent on a test case."""
    from precision_health_agents.agents.doctor import DoctorAgent

    agent = DoctorAgent()
    if case.clinical_features:
        # Direct feature input — single turn
        feature_str = ", ".join(f"{k}={v}" for k, v in case.clinical_features.items())
        agent.chat(f"My clinical values: {feature_str}")
    elif case.patient_description:
        agent.chat(case.patient_description)
    return agent.result(summary="Evaluation run")


async def run_health_trainer(
    case: EvalCase,
    settings: Settings,
    genomics_result: AgentResult | None,
    doctor_result: AgentResult | None,
) -> AgentResult:
    """Run Health Trainer agent on a test case (only for health_trainer decisions)."""
    from precision_health_agents.agents.health_trainer import HealthTrainerAgent

    # Build context from prior agent results
    context: dict = {}
    if genomics_result:
        context["genomics"] = genomics_result.model_dump()
    if doctor_result:
        context["doctor"] = doctor_result.model_dump()

    agent = HealthTrainerAgent(settings=settings, context=context or None)

    # Single-turn: provide vitals + exercise history + equipment
    vitals = case.health_trainer_vitals or {}
    msg = (
        f"I'm {vitals.get('age', 25)} years old, {vitals.get('gender', 'Male')}. "
        f"Height: {vitals.get('height_cm', 170)} cm, Weight: {vitals.get('weight_kg', 70)} kg. "
        f"I exercise {vitals.get('workout_frequency_per_week', 0)} days per week, "
        f"about {vitals.get('session_duration_hours', 0)} hours per session. "
        f"I have basic home dumbbells. No specific body part focus. No injuries."
    )
    agent.chat(msg)
    return agent.result(summary="Evaluation run")


async def run_transcriptomics(
    case: EvalCase,
    settings: Settings,
    genomics_result: AgentResult | None,
    doctor_result: AgentResult | None,
) -> AgentResult:
    """Run Transcriptomics agent on a test case (only for hospital-path cases with gene data)."""
    from precision_health_agents.agents.transcriptomics import TranscriptomicsAgent

    # Build context from prior agent results
    context: dict = {}
    if genomics_result:
        context["genomics"] = genomics_result.model_dump()
    if doctor_result:
        context["doctor"] = doctor_result.model_dump()

    agent = TranscriptomicsAgent(settings=settings)
    # Format gene expression as query
    gene_data = case.gene_expression or {}
    query = "Gene expression profile:\n" + "\n".join(
        f"  {gene}: {val}" for gene, val in gene_data.items()
    )
    return await agent.analyze(query, context=context or None)


# -- Mock I/O ----------------------------------------------------------------


def save_outputs(
    case: EvalCase, results: dict[str, AgentResult], mock_dir: Path
) -> None:
    """Save agent outputs for mock mode."""
    case_dir = mock_dir / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    for agent_name, result in results.items():
        (case_dir / f"{agent_name}.json").write_text(result.model_dump_json(indent=2))


def load_outputs(case: EvalCase, mock_dir: Path) -> dict[str, AgentResult]:
    """Load pre-recorded agent outputs."""
    case_dir = mock_dir / case.id
    results = {}
    for path in case_dir.glob("*.json"):
        data = json.loads(path.read_text())
        results[path.stem] = AgentResult.model_validate(data)
    return results


# -- Eval runner --------------------------------------------------------------


async def evaluate_case(
    case: EvalCase,
    settings: Settings,
    mock: bool = False,
    save: bool = False,
) -> dict:
    """Evaluate a single test case. Returns dict of metric results."""
    # Get agent outputs
    if mock:
        outputs = load_outputs(case, MOCK_DIR)
        genomics_result = outputs.get("genomics")
        doctor_result = outputs.get("doctor")
    else:
        genomics_result = await run_genomics(case, settings)
        doctor_result = await run_doctor(case, settings)

    # Transcriptomics (only for hospital-path cases with gene expression data)
    transcriptomics_result = None
    if case.gene_expression:
        if mock:
            transcriptomics_result = outputs.get("transcriptomics")  # type: ignore[possibly-undefined]
        else:
            transcriptomics_result = await run_transcriptomics(
                case, settings, genomics_result, doctor_result
            )

    # Health trainer (only for health_trainer decision cases)
    health_trainer_result = None
    if case.expected.decision == "health_trainer":
        if mock:
            health_trainer_result = outputs.get("health_trainer")  # type: ignore[possibly-undefined]
        else:
            health_trainer_result = await run_health_trainer(
                case, settings, genomics_result, doctor_result
            )

    if save and not mock:
        results_dict = {}
        if genomics_result:
            results_dict["genomics"] = genomics_result
        if doctor_result:
            results_dict["doctor"] = doctor_result
        if transcriptomics_result:
            results_dict["transcriptomics"] = transcriptomics_result
        if health_trainer_result:
            results_dict["health_trainer"] = health_trainer_result
        save_outputs(case, results_dict, MOCK_DIR)

    # Layer 1: Tool accuracy
    metrics: list[MetricResult] = []
    if genomics_result:
        metrics.append(score_tool_accuracy(genomics_result, case))
    if doctor_result:
        metrics.append(score_tool_accuracy(doctor_result, case))
    if transcriptomics_result:
        metrics.append(score_tool_accuracy(transcriptomics_result, case))
    if health_trainer_result:
        metrics.append(score_tool_accuracy(health_trainer_result, case))

    # Layer 3: Decision correctness (with optional TX override)
    if genomics_result and doctor_result:
        metrics.append(
            score_decision(
                genomics_result, doctor_result, case,
                transcriptomics=transcriptomics_result,
            )
        )

    # Layer 2: LLM-as-judge (skip in mock unless judge is also mocked)
    judge_scores: dict[str, JudgeScore] = {}
    if not mock:
        for agent_result in [
            genomics_result, doctor_result, transcriptomics_result, health_trainer_result,
        ]:
            if agent_result:
                judge_scores[agent_result.agent] = await judge_agent(
                    agent_result, case, settings
                )

    return {
        "case": case.id,
        "metrics": metrics,
        "judge_scores": judge_scores,
    }


def print_report(results: list[dict]) -> None:
    """Print a summary report to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    for result in results:
        print(f"\n--- {result['case']} ---")
        for m in result["metrics"]:
            status = "PASS" if m.passed else "FAIL"
            print(f"  [{status}] {m.agent}/{m.metric}: {m.score:.1f} — {m.detail}")
        for agent, js in result.get("judge_scores", {}).items():
            print(
                f"  [JUDGE] {agent}: "
                f"rel={js.relevance} comp={js.completeness} "
                f"acc={js.accuracy} safe={js.safety}"
            )
            print(f"          {js.explanation}")

    # Summary
    all_metrics = [m for r in results for m in r["metrics"]]
    passed = sum(1 for m in all_metrics if m.passed)
    total = len(all_metrics)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed}/{total} metrics passed")
    print("=" * 60)


def collect_judge_averages(results: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate judge scores per agent for Ralph Loop input."""
    agent_scores: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for agent, js in result.get("judge_scores", {}).items():
            if agent not in agent_scores:
                agent_scores[agent] = {
                    "relevance": [],
                    "completeness": [],
                    "accuracy": [],
                    "safety": [],
                }
            agent_scores[agent]["relevance"].append(js.relevance)
            agent_scores[agent]["completeness"].append(js.completeness)
            agent_scores[agent]["accuracy"].append(js.accuracy)
            agent_scores[agent]["safety"].append(js.safety)

    return {
        agent: {m: sum(v) / len(v) for m, v in scores.items()}
        for agent, scores in agent_scores.items()
    }


async def main():
    parser = argparse.ArgumentParser(description="BioAI Evaluation")
    parser.add_argument("--mock", action="store_true", help="Use pre-recorded outputs")
    parser.add_argument(
        "--save", action="store_true", help="Save agent outputs for mock mode"
    )
    parser.add_argument("--ralph", action="store_true", help="Run Ralph Loop")
    parser.add_argument(
        "--iter", type=int, default=3, help="Ralph Loop iterations (default: 3)"
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    cases = load_cases()

    # Run evaluation
    results = []
    for case in cases:
        result = await evaluate_case(case, settings, mock=args.mock, save=args.save)
        results.append(result)

    print_report(results)

    # Ralph Loop (v2: failure context + rollback)
    if args.ralph:
        avg_scores = collect_judge_averages(results)
        if not avg_scores:
            print("\nNo judge scores available for Ralph Loop. Run without --mock.")
            return

        print(f"\nStarting Ralph Loop v2 ({args.iter} iterations)...")
        history: list[RalphResult] = []

        for i in range(args.iter):
            print(f"\n--- Ralph iteration {i + 1} ---")

            # Collect failure examples from judge scores
            failures: list[FailureExample] = []
            for result in results:
                for agent, js in result.get("judge_scores", {}).items():
                    if min(js.relevance, js.completeness, js.accuracy, js.safety) < 3.0:
                        failures.append(
                            FailureExample(
                                case_id=result["case"],
                                agent_output=f"{agent} summary",
                                judge_explanation=js.explanation,
                            )
                        )

            ralph_result = await ralph_iterate(
                avg_scores,
                settings=settings,
                failure_context=failures or None,
                history=history or None,
            )
            print(
                f"  Target: {ralph_result.agent}/{ralph_result.metric} "
                f"(score: {ralph_result.old_score:.1f})"
            )
            print(f"  Changed: {ralph_result.prompt_changed}")
            print(f"  Detail: {ralph_result.diff}")

            if ralph_result.prompt_changed:
                # Re-run eval to get new scores
                old_avg = avg_scores.copy()
                results = []
                for case in cases:
                    result = await evaluate_case(case, settings)
                    results.append(result)
                print_report(results)
                avg_scores = collect_judge_averages(results)

                # Check for regression — rollback if worse
                new_agent_scores = avg_scores.get(ralph_result.agent, {})
                old_agent_scores = old_avg.get(ralph_result.agent, {})
                new_target = new_agent_scores.get(ralph_result.metric, 0.0)
                old_target = old_agent_scores.get(ralph_result.metric, 0.0)
                ralph_result.new_score = new_target

                if new_target < old_target and ralph_result.backup_path:
                    backup = Path(ralph_result.backup_path)
                    if backup.exists():
                        prompt_file = backup.with_suffix("")  # remove .bak
                        prompt_file.write_text(backup.read_text())
                        backup.unlink()
                        print(
                            f"  ROLLBACK: {ralph_result.metric} regressed "
                            f"({old_target:.1f} → {new_target:.1f}), restored prompt"
                        )
                        avg_scores = old_avg  # revert scores

            history.append(ralph_result)


if __name__ == "__main__":
    asyncio.run(main())
