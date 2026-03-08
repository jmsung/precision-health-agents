"""Tests for the DNA classifier tool."""

from bioai.tools.dna_classifier import CLASSES, _load_model, _load_tokenizer, classify_dna

# A short but valid DNA sequence for testing
_SAMPLE_SEQ = "ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG"


def test_load_model():
    model = _load_model()
    assert model is not None
    assert model.output_shape == (None, 3)


def test_load_tokenizer():
    tokenizer = _load_tokenizer()
    # Should have vocabulary from 3-mers (4^3 = 64 possible, but subset used)
    assert len(tokenizer.word_index) > 0
    # All keys should be 3-char lowercase strings
    sample_keys = list(tokenizer.word_index.keys())[:10]
    assert all(len(k) == 3 for k in sample_keys)


def test_classify_dna_output_structure():
    result = classify_dna(_SAMPLE_SEQ)
    assert set(result.keys()) == {"predicted_class", "probabilities", "confidence"}
    assert result["predicted_class"] in CLASSES
    assert set(result["probabilities"].keys()) == set(CLASSES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_dna_confidence_matches_predicted_class():
    result = classify_dna(_SAMPLE_SEQ)
    predicted = result["predicted_class"]
    assert result["confidence"] == result["probabilities"][predicted]


def test_classify_dna_short_sequence():
    # Sequences shorter than training data should still return a valid result
    short_seq = "ATGATG"
    result = classify_dna(short_seq)
    assert result["predicted_class"] in CLASSES
