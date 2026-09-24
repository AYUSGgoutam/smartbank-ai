"""Train and compare fraud classifiers using the Phase 1 data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from src.config import DEFAULT_DATA_DIR, PROJECT_ROOT, RANDOM_SEED
from src.preprocessing.prepare import prepare_data

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def build_models(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Return comparable classifiers; class weighting complements train resampling."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_seed),
        "XGBoost": XGBClassifier(n_estimators=350, max_depth=5, learning_rate=.06,
            subsample=.85, colsample_bytree=.85, reg_lambda=2, min_child_weight=3,
            objective="binary:logistic", eval_metric="logloss", n_jobs=-1,
            random_state=random_seed),
    }


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    """Calculate threshold metrics and ranking metrics on held-out observations."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= .5).astype("int8")
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }


def train_and_compare(transactions: pd.DataFrame, model_dir: Path = DEFAULT_MODEL_DIR,
                      random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fit models, persist test metrics and save the highest PR-AUC model bundle."""
    prepared = prepare_data(transactions, random_seed=random_seed)
    models = build_models(random_seed)
    reports: list[dict[str, Any]] = []
    best_name: str | None = None
    best_score = float("-inf")
    best_model: Any = None
    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(prepared["X_train"], prepared["y_train"])
        metrics = evaluate_model(model, prepared["X_test"], prepared["y_test"])
        reports.append({"model": name, **metrics})
        LOGGER.info("%s | PR-AUC %.4f | ROC-AUC %.4f | Recall %.4f | F1 %.4f",
                    name, metrics["pr_auc"], metrics["roc_auc"], metrics["recall"], metrics["f1"])
        if metrics["pr_auc"] > best_score:
            best_name, best_score, best_model = name, metrics["pr_auc"], model

    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "preprocessor": prepared["preprocessor"],
        "feature_names": prepared["feature_names"],
        "model_name": best_name,
        "decision_threshold": .5,
        "selection_metric": "pr_auc",
        "selection_score": best_score,
    }
    joblib.dump(bundle, model_dir / "fraud_model.pkl")
    report = pd.DataFrame(reports).sort_values("pr_auc", ascending=False)
    report.to_csv(model_dir / "fraud_model_metrics.csv", index=False)
    (model_dir / "fraud_model_metrics.json").write_text(
        json.dumps({"selected_model": best_name, "selection_metric": "pr_auc",
                    "test_rows": len(prepared["y_test"]), "fraud_rate": float(pd.Series(prepared["y_test"]).mean()),
                    "models": reports}, indent=2), encoding="utf-8")
    LOGGER.info("Selected %s by held-out PR-AUC %.4f", best_name, best_score)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "transactions.csv")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frame = pd.read_csv(args.input, parse_dates=["timestamp"])
    report = train_and_compare(frame, args.model_dir, args.seed)
    print(report[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
