"""Train a customer-grouped LSTM over chronological transaction sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_DATA_DIR, PROJECT_ROOT, RANDOM_SEED

SEQUENCE_FEATURES = ["amount", "is_international", "account_age_days", "previous_transaction_count",
                     "average_transaction_amount", "failed_transaction_count", "hour", "day_of_week",
                     "distance_from_home", "new_device", "amount_to_average_ratio", "night_transaction"]


def make_sequences(frame: pd.DataFrame, sequence_length: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed windows, labeling each window by the fraud label of its final event."""
    data = frame.copy()
    typical = pd.to_numeric(data["average_transaction_amount"], errors="coerce").clip(lower=1)
    data["amount_to_average_ratio"] = (pd.to_numeric(data.amount, errors="coerce") / typical).clip(0, 100)
    timestamps = pd.to_datetime(data.timestamp, utc=True)
    data["night_transaction"] = timestamps.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype("int8")
    data = data.sort_values(["customer_id", "timestamp"])
    sequences, labels, groups, latest_rows = [], [], [], []
    for customer_id, group in data.groupby("customer_id", sort=False):
        values = group[SEQUENCE_FEATURES].to_numpy(dtype="float32")
        target = group.fraud.to_numpy(dtype="int8")
        for end in range(sequence_length - 1, len(group)):
            sequences.append(values[end - sequence_length + 1:end + 1])
            labels.append(target[end])
            groups.append(customer_id)
            latest_rows.append(group.index[end])
    return (np.asarray(sequences, dtype="float32"), np.asarray(labels, dtype="int8"),
            np.asarray(groups), np.asarray(latest_rows))


def train_lstm(frame: pd.DataFrame, model_dir: Path = PROJECT_ROOT / "models",
               sequence_length: int = 10) -> dict[str, float]:
    """Train, evaluate and save a Keras LSTM and its sequence scaler."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install TensorFlow to train the LSTM") from exc
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    data = data.reset_index(drop=True)
    X, y, groups, _ = make_sequences(data, sequence_length)
    if len(np.unique(y)) < 2:
        raise ValueError("Sequence dataset must include both fraud classes")
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=RANDOM_SEED)
    train_idx, val_idx = next(split.split(X, y, groups))
    scaler = StandardScaler()
    train_flat = X[train_idx].reshape(-1, X.shape[-1])
    scaler.fit(train_flat)
    X_train = scaler.transform(train_flat).reshape(X[train_idx].shape).astype("float32")
    X_val = scaler.transform(X[val_idx].reshape(-1, X.shape[-1])).reshape(X[val_idx].shape).astype("float32")
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    inputs = tf.keras.Input(shape=(sequence_length, len(SEQUENCE_FEATURES)))
    hidden = tf.keras.layers.LSTM(32, dropout=.2)(inputs)
    hidden = tf.keras.layers.Dense(16, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"), tf.keras.metrics.Recall(name="recall")])
    positives = int(y[train_idx].sum())
    negatives = len(train_idx) - positives
    class_weight = {0: 1.0, 1: max(1.0, negatives / max(positives, 1))}
    model.fit(X_train, y[train_idx], validation_data=(X_val, y[val_idx]), epochs=30, batch_size=256,
              class_weight=class_weight, callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,
              monitor="val_pr_auc", mode="max", restore_best_weights=True)], verbose=0)
    results = model.evaluate(X_val, y[val_idx], verbose=0, return_dict=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "lstm_model.keras")
    joblib.dump({"scaler": scaler, "features": SEQUENCE_FEATURES, "sequence_length": sequence_length},
                model_dir / "lstm_preprocessing.pkl")
    metrics = {key: float(value) for key, value in results.items()}
    (model_dir / "lstm_metrics.json").write_text(json.dumps({**metrics, "sequence_length": sequence_length,
        "validation_sequences": len(val_idx), "validation_customers": int(len(np.unique(groups[val_idx])))} , indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "transactions.csv")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--sequence-length", type=int, default=10)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(train_lstm(frame, args.model_dir, args.sequence_length))


if __name__ == "__main__":
    main()
