"""DNA classification tool using a pre-trained 2-layer CNN model.

Classifies a DNA sequence as DMT1 (Type 1 Diabetes), DMT2 (Type 2 Diabetes),
or NONDM (Non-Diabetic) using 3-mer tokenization.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Paths relative to project root
_PROJECT_ROOT = Path(__file__).parents[3]
_DATA_DIR = _PROJECT_ROOT / "data" / "dna_classification"
_MODEL_PATH = _DATA_DIR / "models" / "CNN_2Layers_3mers.h5"
_DATASET_PATH = _DATA_DIR / "raw" / "Complete_DM_DNA_Sequence.csv"

# Fixed preprocessing constants (from training)
MAX_LENGTH = 9203
CLASSES = ["DMT1", "DMT2", "NONDM"]  # alphabetical order (OneHotEncoder)


def _kmers(seq: str, size: int = 3) -> list[str]:
    """Split a DNA sequence into overlapping k-mers."""
    return [seq[i : i + size].lower() for i in range(len(seq) - size + 1)]


@lru_cache(maxsize=1)
def _load_tokenizer():
    """Fit and cache the Keras Tokenizer on the training dataset."""
    from tensorflow.keras.preprocessing.text import Tokenizer

    data = pd.read_csv(_DATASET_PATH)
    sequences = data["sequence"].tolist()
    kmers_list = [" ".join(_kmers(seq)) for seq in sequences]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(kmers_list)
    return tokenizer


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the pre-trained CNN model."""
    import tensorflow as tf

    return tf.keras.models.load_model(str(_MODEL_PATH))


def classify_dna(sequence: str) -> dict:
    """Classify a DNA sequence for diabetes-associated genomic risk.

    Args:
        sequence: Raw DNA sequence string (A/T/G/C characters).

    Returns:
        dict with keys:
            - predicted_class: str, one of "DMT1", "DMT2", "NONDM"
            - probabilities: dict mapping class name to confidence score
            - confidence: float, confidence of the top prediction
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Preprocess
    kmers_str = " ".join(_kmers(sequence))
    tokenizer = _load_tokenizer()
    tokenized = tokenizer.texts_to_sequences([kmers_str])
    padded = pad_sequences(tokenized, maxlen=MAX_LENGTH, padding="post")

    # Predict
    model = _load_model()
    probs = model.predict(padded, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASSES[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
        "confidence": float(probs[predicted_idx]),
    }
