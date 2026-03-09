"""Tests for evaluation test cases."""

from precision_health_agents.eval.cases import ExpectedOutput, EvalCase, load_cases


def test_expected_output_validates():
    out = ExpectedOutput(
        dna_class="DMT2",
        clinical_prediction="Diabetic",
        decision="hospital",
    )
    assert out.dna_class == "DMT2"
    assert out.decision == "hospital"


def test_test_case_structure():
    case = EvalCase(
        id="test-1",
        name="Test Case",
        description="A test case",
        expected=ExpectedOutput(
            dna_class="NONDM",
            clinical_prediction="Non-Diabetic",
            decision="health_trainer",
        ),
    )
    assert case.id == "test-1"
    assert case.dna_sequence is None
    assert case.clinical_features is None
    assert case.expected.decision == "health_trainer"


def test_load_cases_returns_list():
    cases = load_cases()
    assert isinstance(cases, list)
    assert len(cases) == 4


def test_load_cases_all_have_expected():
    cases = load_cases()
    for case in cases:
        assert case.expected is not None
        assert case.expected.decision in ("hospital", "reconsider", "health_trainer")


def test_load_cases_cover_all_decisions():
    cases = load_cases()
    decisions = {c.expected.decision for c in cases}
    assert "hospital" in decisions
    assert "reconsider" in decisions
    assert "health_trainer" in decisions


def test_load_cases_have_inputs():
    """Cases should have DNA sequences and clinical features from case_inputs.json."""
    cases = load_cases()
    for case in cases:
        assert case.dna_sequence is not None, f"{case.id} missing dna_sequence"
        assert len(case.dna_sequence) > 100, f"{case.id} dna_sequence too short"
        assert case.clinical_features is not None, f"{case.id} missing clinical_features"
        assert len(case.clinical_features) == 8, f"{case.id} should have 8 features"
