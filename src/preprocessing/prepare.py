"""Clean, split and transform transaction data without fitting on test rows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, FRAUD_TARGET,
                        NUMERIC_TRANSACTION_FEATURES, CATEGORICAL_TRANSACTION_FEATURES,
                        RANDOM_SEED, TRANSACTION_FEATURES)

LOGGER = logging.getLogger(__name__)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add bounded, interpretable behavioral features while retaining source fields."""
    data = frame.copy()
    if "timestamp" in data:
        timestamp = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
        data["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype("int8")
        data["night_transaction"] = timestamp.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype("int8")
    amount = pd.to_numeric(data["amount"], errors="coerce").clip(lower=0)
    typical = pd.to_numeric(data["average_transaction_amount"], errors="coerce").clip(lower=1)
    data["amount_to_average_ratio"] = (amount / typical).replace([np.inf, -np.inf], np.nan).clip(0, 100)
    data["log_amount"] = np.log1p(amount)
    # Fixed domain bounds cap impossible/extreme values without learning from held-out rows.
    bounds = {"amount": (0, 250_000), "distance_from_home": (0, 8_000),
              "monthly_income": (0, 500_000), "login_frequency": (0, 60)}
    for column, (low, high) in bounds.items():
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").clip(lower=low, upper=high)
    return data


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create imputing/scaling and imputation/one-hot pipelines."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric_features),
                              ("categorical", categorical_pipe, categorical_features)],
                             remainder="drop", verbose_feature_names_out=False)


def prepare_data(
    transactions: pd.DataFrame,
    test_size: float = .2,
    random_seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Return cleaned rows, leakage-safe split, fitted transformer and balanced train arrays."""
    if FRAUD_TARGET not in transactions:
        raise ValueError(f"Input must contain target column {FRAUD_TARGET!r}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    data = transactions.drop_duplicates(subset=["transaction_id"] if "transaction_id" in transactions else None).copy()
    data = engineer_features(data)
    feature_columns = TRANSACTION_FEATURES + ["amount_to_average_ratio", "log_amount", "is_weekend", "night_transaction"]
    absent = sorted(set(feature_columns) - set(data.columns))
    if absent:
        raise ValueError(f"Missing required feature columns: {', '.join(absent)}")
    y = data[FRAUD_TARGET].astype("int8")
    X = data[feature_columns]
    num_cols = NUMERIC_TRANSACTION_FEATURES + ["amount_to_average_ratio", "log_amount", "is_weekend", "night_transaction"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    transformer = build_preprocessor(num_cols, CATEGORICAL_TRANSACTION_FEATURES)
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)
    # Random oversampling happens after the split and only touches training arrays.
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Fraud target must contain both classes")
    majority_count = int(class_counts.max())
    rng = np.random.default_rng(random_seed)
    oversampled_parts = [np.flatnonzero(y_train.to_numpy() == label) for label in class_counts.index]
    sample_indices = np.concatenate([
        rng.choice(indices, size=majority_count, replace=len(indices) < majority_count)
        for indices in oversampled_parts
    ])
    rng.shuffle(sample_indices)
    return {
        "cleaned_data": data, "X_train": X_train_processed[sample_indices],
        "X_test": X_test_processed, "y_train": y_train.to_numpy()[sample_indices],
        "y_test": y_test.to_numpy(), "preprocessor": transformer,
        "feature_names": transformer.get_feature_names_out().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    transactions = pd.read_csv(args.input_dir / "transactions.csv", parse_dates=["timestamp"])
    result = prepare_data(transactions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "train_test.npz", X_train=result["X_train"], X_test=result["X_test"],
                        y_train=result["y_train"], y_test=result["y_test"])
    result["cleaned_data"].to_csv(args.output_dir / "transactions_cleaned.csv", index=False)
    joblib.dump(result["preprocessor"], args.artifact_dir / "transaction_preprocessor.joblib")
    pd.Series(result["feature_names"], name="feature_name").to_csv(args.artifact_dir / "feature_names.csv", index=False)
    LOGGER.info("Prepared rows: %d; balanced training rows: %d; held-out test rows: %d",
                len(result["cleaned_data"]), len(result["y_train"]), len(result["y_test"]))


if __name__ == "__main__":
    main()
