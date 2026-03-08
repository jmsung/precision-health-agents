"""BioAI Evaluation Dashboard — Streamlit app."""

import asyncio
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bioai.eval.cases import EvalCase, load_cases
from bioai.eval.judge import JudgeScore, judge_agent
from bioai.eval.metrics import score_decision, score_tool_accuracy
from bioai.models import AgentResult

MOCK_DIR = Path("src/bioai/eval/data/mock_outputs")


# -- Data loading -------------------------------------------------------------


def load_mock_outputs(case: EvalCase) -> dict[str, AgentResult]:
    """Load saved agent outputs for a case."""
    case_dir = MOCK_DIR / case.id
    results = {}
    for path in case_dir.glob("*.json"):
        data = json.loads(path.read_text())
        results[path.stem] = AgentResult.model_validate(data)
    return results


def run_deterministic_eval(
    cases: list[EvalCase],
) -> pd.DataFrame:
    """Run Layer 1 + 3 metrics on all cases, return DataFrame."""
    rows = []
    for case in cases:
        outputs = load_mock_outputs(case)
        genomics = outputs.get("genomics")
        doctor = outputs.get("doctor")

        if genomics:
            m = score_tool_accuracy(genomics, case)
            rows.append(
                {
                    "case": case.name,
                    "agent": "genomics",
                    "metric": "tool_accuracy",
                    "score": m.score,
                    "passed": m.passed,
                    "detail": m.detail,
                }
            )
        if doctor:
            m = score_tool_accuracy(doctor, case)
            rows.append(
                {
                    "case": case.name,
                    "agent": "doctor",
                    "metric": "tool_accuracy",
                    "score": m.score,
                    "passed": m.passed,
                    "detail": m.detail,
                }
            )
        if genomics and doctor:
            m = score_decision(genomics, doctor, case)
            rows.append(
                {
                    "case": case.name,
                    "agent": "combined",
                    "metric": "decision",
                    "score": m.score,
                    "passed": m.passed,
                    "detail": m.detail,
                }
            )
    return pd.DataFrame(rows)


def run_judge_eval(cases: list[EvalCase]) -> pd.DataFrame:
    """Run LLM-as-judge on saved outputs. Uses API."""
    rows = []
    for case in cases:
        outputs = load_mock_outputs(case)
        for agent_name, result in outputs.items():
            js = asyncio.run(judge_agent(result, case))
            rows.append(
                {
                    "case": case.name,
                    "agent": agent_name,
                    "relevance": js.relevance,
                    "completeness": js.completeness,
                    "accuracy": js.accuracy,
                    "safety": js.safety,
                    "explanation": js.explanation,
                }
            )
    return pd.DataFrame(rows)


# -- Page config --------------------------------------------------------------

st.set_page_config(page_title="BioAI Evaluation", layout="wide")
st.title("BioAI Evaluation Dashboard")

cases = load_cases()

# -- Tabs ----------------------------------------------------------------------

tab_overview, tab_details, tab_judge = st.tabs(
    ["Overview", "Case Details", "LLM-as-Judge"]
)

# -- Tab 1: Overview -----------------------------------------------------------

with tab_overview:
    st.header("Deterministic Metrics")

    if not MOCK_DIR.exists():
        st.warning("No saved outputs. Run `scripts/evaluate.py --save` first.")
    else:
        df = run_deterministic_eval(cases)

        # Summary metrics
        passed = df["passed"].sum()
        total = len(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Metrics", total)
        col2.metric("Passed", int(passed))
        col3.metric("Pass Rate", f"{passed / total:.0%}")

        # Heatmap: case x agent
        pivot = df.pivot_table(
            index="case", columns="agent", values="score", aggfunc="mean"
        )
        fig = px.imshow(
            pivot,
            color_continuous_scale=[[0, "#ff4444"], [0.5, "#555555"], [1, "#00cc66"]],
            zmin=0,
            zmax=1,
            text_auto=".1f",
            aspect="auto",
        )
        fig.update_layout(
            title="Tool Accuracy + Decision Correctness",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fafafa",
        )
        st.plotly_chart(fig, width="stretch")

        # Detail table
        st.subheader("Metric Details")
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color: #1a3a2a; color: #66ff99"
                    if row["passed"]
                    else "background-color: #3a1a1a; color: #ff6666"
                ]
                * len(row),
                axis=1,
            ),
            width="stretch",
        )

# -- Tab 2: Case Details ------------------------------------------------------

with tab_details:
    st.header("Case Details")

    for case in cases:
        with st.expander(f"{case.id}: {case.name}"):
            st.markdown(f"**Description**: {case.description}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Expected**")
                st.json(case.expected.model_dump())
            with col2:
                st.markdown("**Inputs**")
                if case.clinical_features:
                    st.json(case.clinical_features)
                if case.dna_sequence:
                    st.code(case.dna_sequence[:80] + "...", language="text")

            if (MOCK_DIR / case.id).exists():
                outputs = load_mock_outputs(case)
                for agent_name, result in outputs.items():
                    st.markdown(f"---\n**{agent_name.title()} Agent Output**")
                    if result.findings:
                        st.json(result.findings.model_dump())
                    st.markdown(result.summary)

# -- Tab 3: LLM-as-Judge ------------------------------------------------------

with tab_judge:
    st.header("LLM-as-Judge Scores")
    st.caption("Click 'Run Judge' to score saved outputs via Claude API.")

    if st.button("Run Judge"):
        with st.spinner("Scoring with Claude Sonnet..."):
            judge_df = run_judge_eval(cases)
            st.session_state["judge_df"] = judge_df

    if "judge_df" in st.session_state:
        judge_df = st.session_state["judge_df"]

        # Heatmap
        dims = ["relevance", "completeness", "accuracy", "safety"]
        for agent in judge_df["agent"].unique():
            agent_df = judge_df[judge_df["agent"] == agent]
            pivot = agent_df.set_index("case")[dims]
            fig = px.imshow(
                pivot,
                color_continuous_scale=[[0, "#ff4444"], [0.5, "#555555"], [1, "#00cc66"]],
                zmin=1,
                zmax=5,
                text_auto=True,
                aspect="auto",
            )
            fig.update_layout(
                title=f"{agent.title()} Agent — Judge Scores",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#fafafa",
            )
            st.plotly_chart(fig, width="stretch")

        # Explanations
        st.subheader("Judge Explanations")
        for _, row in judge_df.iterrows():
            with st.expander(f"{row['case']} — {row['agent']}"):
                st.markdown(row["explanation"])
    else:
        st.info("No judge scores yet. Click 'Run Judge' above.")
