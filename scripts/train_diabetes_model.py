"""Train a simple MLP on the Pima Indians Diabetes dataset.

Saves the model to data/diabetes/models/mlp_diabetes.keras and a scaler to
data/diabetes/models/scaler.npy for use by the diabetes_classifier tool.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf

PROJECT_ROOT = Path(__file__).parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "diabetes"
RAW_PATH = DATA_DIR / "raw" / "diabetes.csv"
MODEL_PATH = DATA_DIR / "models" / "mlp_diabetes.keras"
SCALER_PATH = DATA_DIR / "models" / "scaler.npy"


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Class balance:\n{df['Outcome'].value_counts()}")

    X = df.drop("Outcome", axis=1).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.float32)

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler params (mean + scale) for inference
    np.save(str(SCALER_PATH), np.array([scaler.mean_, scaler.scale_]))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest  loss: {loss:.4f}  accuracy: {acc:.4f}")

    y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")


if __name__ == "__main__":
    main()
