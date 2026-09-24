"""Fit unsupervised Isolation Forest and optional sequence autoencoder models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_DATA_DIR, PROJECT_ROOT, RANDOM_SEED

LOGGER = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / "models"
ANOMALY_FEATURES = ["amount", "is_international", "account_age_days", "previous_transaction_count",
                    "average_transaction_amount", "failed_transaction_count", "hour", "day_of_week",
                    "distance_from_home", "new_device"]


def fit_isolation_forest(transactions: pd.DataFrame, model_dir: Path = MODEL_DIR) -> dict[str, object]:
    """Fit on unlabeled behavior and derive a threshold from held-out normal scores."""
    X = transactions[ANOMALY_FEATURES].replace([np.inf, -np.inf], np.nan)
    x_train, x_val = train_test_split(X, test_size=.2, random_state=RANDOM_SEED)
    medians = x_train.median()
    lower, upper = x_train.quantile(.005), x_train.quantile(.995)
    x_train = x_train.fillna(medians).clip(lower=lower, upper=upper, axis="columns")
    x_val = x_val.fillna(medians).clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    val_scaled = scaler.transform(x_val)
    model = IsolationForest(n_estimators=250, contamination="auto", n_jobs=-1, random_state=RANDOM_SEED)
    model.fit(train_scaled)
    # score_samples is higher for normal points; lower tail of validation defines the flag cutoff.
    val_scores = -model.score_samples(val_scaled)
    threshold = float(np.quantile(val_scores, .99))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": ANOMALY_FEATURES,
                 "threshold": threshold, "medians": medians.to_dict(),
                 "lower": lower.to_dict(), "upper": upper.to_dict()}, model_dir / "anomaly_model.pkl")
    return {"threshold": threshold, "validation_rows": len(x_val), "validation_anomaly_fraction": float((val_scores >= threshold).mean())}


def train_autoencoder(transactions: pd.DataFrame, model_dir: Path = MODEL_DIR) -> dict[str, float]:
    """Train a Keras dense autoencoder on normal transactions and validate its cutoff."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install TensorFlow to train the autoencoder") from exc
    X = transactions[ANOMALY_FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).to_numpy(dtype="float32")
    normal = X[transactions.fraud.to_numpy() == 0]
    train, val = train_test_split(normal, test_size=.2, random_state=RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    scaler = StandardScaler()
    train = scaler.fit_transform(train).astype("float32")
    val = scaler.transform(val).astype("float32")
    dims = train.shape[1]
    inputs = tf.keras.Input(shape=(dims,))
    encoded = tf.keras.layers.Dense(16, activation="relu")(inputs)
    encoded = tf.keras.layers.Dense(6, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(16, activation="relu")(encoded)
    outputs = tf.keras.layers.Dense(dims)(decoded)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(train, train, validation_data=(val, val), epochs=30, batch_size=256,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)], verbose=0)
    errors = np.mean(np.square(model.predict(val, verbose=0) - val), axis=1)
    threshold = float(np.quantile(errors, .99))
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "autoencoder.keras")
    joblib.dump({"scaler": scaler, "features": ANOMALY_FEATURES, "threshold": threshold},
                model_dir / "autoencoder_preprocessing.pkl")
    return {"validation_threshold": threshold, "validation_normal_rows": len(val)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "transactions.csv")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--skip-autoencoder", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    data = pd.read_csv(args.input)
    metrics = {"isolation_forest": fit_isolation_forest(data, args.model_dir)}
    print("Isolation Forest:", metrics["isolation_forest"])
    if not args.skip_autoencoder:
        metrics["autoencoder"] = train_autoencoder(data, args.model_dir)
        print("Autoencoder:", metrics["autoencoder"])
    (args.model_dir / "anomaly_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
