"""Tests for Ralph Loop prompt optimizer."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from bioai.eval.ralph import FailureExample, RalphResult, ralph_iterate, _find_weakest


def _mock_rewrite_response(new_prompt: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = new_prompt

    response = MagicMock()
    response.content = [block]
    return response


def _mock_judge_response(scores: dict) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(scores)

    response = MagicMock()
    response.content = [block]
    return response


# -- Existing tests -----------------------------------------------------------


@patch("bioai.eval.ralph.anthropic.Anthropic")
def test_ralph_iterate_rewrites_prompt(mock_anthropic_cls, tmp_path):
    """Ralph should rewrite the weakest agent's prompt file."""
    prompt_file = tmp_path / "doctor.txt"
    prompt_file.write_text("You are a doctor. Ask questions.")

    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_rewrite_response(
        "You are an expert clinical doctor. Gather all 8 features carefully."
    )

    eval_scores = {
        "doctor": {"relevance": 4, "completeness": 2, "accuracy": 4, "safety": 5},
        "genomics": {"relevance": 4, "completeness": 4, "accuracy": 5, "safety": 5},
    }

    result = asyncio.run(
        ralph_iterate(eval_scores=eval_scores, prompt_dir=tmp_path)
    )

    assert isinstance(result, RalphResult)
    assert result.agent == "doctor"
    assert result.metric == "completeness"
    assert result.prompt_changed
    assert "expert clinical doctor" in prompt_file.read_text()


@patch("bioai.eval.ralph.anthropic.Anthropic")
def test_ralph_iterate_skips_missing_prompt(mock_anthropic_cls, tmp_path):
    """If the worst agent has no prompt file, ralph should report no change."""
    eval_scores = {
        "doctor": {"relevance": 5, "completeness": 5, "accuracy": 5, "safety": 5},
        "genomics": {"relevance": 2, "completeness": 2, "accuracy": 2, "safety": 5},
    }

    result = asyncio.run(
        ralph_iterate(eval_scores=eval_scores, prompt_dir=tmp_path)
    )

    assert not result.prompt_changed
    assert "no prompt file" in result.diff.lower()


# -- P1: Filter non-prompt-improvable metrics ---------------------------------


def test_find_weakest_skips_tool_accuracy():
    """_find_weakest should ignore tool_accuracy and decision metrics."""
    eval_scores = {
        "genomics": {
            "tool_accuracy": 0.0,  # worst but not prompt-improvable
            "relevance": 4,
            "completeness": 3,
            "accuracy": 4,
            "safety": 5,
        },
    }
    agent, metric, score = _find_weakest(eval_scores)
    assert metric == "completeness"
    assert score == 3


def test_find_weakest_skips_decision():
    """_find_weakest should ignore combined decision metric."""
    eval_scores = {
        "combined": {"decision": 0.0},
        "doctor": {"relevance": 4, "completeness": 3, "accuracy": 4, "safety": 5},
    }
    agent, metric, score = _find_weakest(eval_scores)
    assert agent == "doctor"
    assert metric == "completeness"


# -- P0: Failure context in rewrite prompt ------------------------------------


@patch("bioai.eval.ralph.anthropic.Anthropic")
def test_ralph_includes_failure_context(mock_anthropic_cls, tmp_path):
    """Rewrite prompt should include failure examples when provided."""
    prompt_file = tmp_path / "doctor.txt"
    prompt_file.write_text("You are a doctor.")

    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_rewrite_response(
        "You are an improved doctor."
    )

    eval_scores = {
        "doctor": {"relevance": 4, "completeness": 2, "accuracy": 4, "safety": 5},
    }
    failures = [
        FailureExample(
            case_id="case_1",
            agent_output="Patient has diabetes.",
            judge_explanation="Missing BMI discussion and family history analysis.",
        ),
    ]

    asyncio.run(
        ralph_iterate(
            eval_scores=eval_scores,
            prompt_dir=tmp_path,
            failure_context=failures,
        )
    )

    # Verify the API call included failure context
    call_args = mock_client.messages.create.call_args
    user_msg = call_args.kwargs["messages"][0]["content"]
    assert "case_1" in user_msg
    assert "Missing BMI discussion" in user_msg
    assert "Patient has diabetes." in user_msg


# -- P0: Backup prompt for rollback ------------------------------------------


@patch("bioai.eval.ralph.anthropic.Anthropic")
def test_ralph_saves_backup(mock_anthropic_cls, tmp_path):
    """Ralph should save backup of original prompt before rewriting."""
    prompt_file = tmp_path / "doctor.txt"
    original = "You are a doctor. Ask questions."
    prompt_file.write_text(original)

    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_rewrite_response(
        "You are an improved doctor."
    )

    eval_scores = {
        "doctor": {"relevance": 4, "completeness": 2, "accuracy": 4, "safety": 5},
    }

    result = asyncio.run(
        ralph_iterate(eval_scores=eval_scores, prompt_dir=tmp_path)
    )

    assert result.backup_path is not None
    backup = Path(result.backup_path)
    assert backup.exists()
    assert backup.read_text() == original


@patch("bioai.eval.ralph.anthropic.Anthropic")
def test_ralph_rollback_restores_prompt(mock_anthropic_cls, tmp_path):
    """Caller can restore prompt from backup_path on regression."""
    prompt_file = tmp_path / "doctor.txt"
    original = "You are a doctor. Ask questions."
    prompt_file.write_text(original)

    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_rewrite_response(
        "You are an improved doctor."
    )

    eval_scores = {
        "doctor": {"relevance": 4, "completeness": 2, "accuracy": 4, "safety": 5},
    }

    result = asyncio.run(
        ralph_iterate(eval_scores=eval_scores, prompt_dir=tmp_path)
    )

    # Simulate regression: caller restores from backup
    assert prompt_file.read_text().startswith("You are an improved doctor.")
    backup = Path(result.backup_path)
    prompt_file.write_text(backup.read_text())
    assert prompt_file.read_text() == original


# -- P2: History prevents oscillation -----------------------------------------


@patch("bioai.eval.ralph.anthropic.Anthropic")
def test_ralph_includes_history(mock_anthropic_cls, tmp_path):
    """Rewrite prompt should include past iteration history when provided."""
    prompt_file = tmp_path / "doctor.txt"
    prompt_file.write_text("You are a doctor.")

    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.return_value = _mock_rewrite_response(
        "You are an improved doctor."
    )

    eval_scores = {
        "doctor": {"relevance": 4, "completeness": 3, "accuracy": 4, "safety": 5},
    }
    history = [
        RalphResult(
            agent="doctor",
            metric="completeness",
            old_score=2.0,
            new_score=3.0,
            prompt_changed=True,
            diff="Rewrote doctor.txt to improve completeness",
        ),
    ]

    asyncio.run(
        ralph_iterate(
            eval_scores=eval_scores,
            prompt_dir=tmp_path,
            history=history,
        )
    )

    call_args = mock_client.messages.create.call_args
    user_msg = call_args.kwargs["messages"][0]["content"]
    assert "Previous iterations" in user_msg
    assert "completeness" in user_msg
    assert "2.0" in user_msg
    assert "3.0" in user_msg
