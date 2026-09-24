"""Train customer risk-score regression and churn-probability classifiers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DEFAULT_DATA_DIR, PROJECT_ROOT, RANDOM_SEED

LOGGER = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / "models"
NUMERIC = ["age", "account_age_days", "monthly_income", "credit_score", "total_transactions",
           "average_transaction_amount", "failed_transactions", "complaints", "login_frequency", "customer_service_calls"]


def fit_customer_models(customers: pd.DataFrame, model_dir: Path = MODEL_DIR) -> dict[str, dict[str, float]]:
    """Fit models with a shared leakage-safe held-out split and persist reports."""
    required = set(NUMERIC + ["risk_score", "churn"])
    if absent := required - set(customers):
        raise ValueError(f"Missing customer columns: {', '.join(sorted(absent))}")
    X = customers[NUMERIC]
    x_train, x_test, y_r_train, y_r_test = train_test_split(X, customers.risk_score, test_size=.2,
                                                             random_state=RANDOM_SEED)
    # Fit preprocessing on train only. Risk/churn labels never enter model features.
    risk_prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), NUMERIC)])
    xtr, xte = risk_prep.fit_transform(x_train), risk_prep.transform(x_test)
    risk_model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_SEED)
    risk_model.fit(xtr, y_r_train)
    risk_pred = risk_model.predict(xte).clip(0, 100)
    risk_metrics = {"mae": float(mean_absolute_error(y_r_test, risk_pred)), "r2": float(r2_score(y_r_test, risk_pred))}
    x_train_c, x_test_c, y_c_train, y_c_test = train_test_split(X, customers.churn, test_size=.2,
        random_state=RANDOM_SEED, stratify=customers.churn)
    churn_prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), NUMERIC)])
    xtrc, xtec = churn_prep.fit_transform(x_train_c), churn_prep.transform(x_test_c)
    churn_model = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced",
                                         n_jobs=-1, random_state=RANDOM_SEED)
    churn_model.fit(xtrc, y_c_train)
    churn_metrics = {"roc_auc": float(roc_auc_score(y_c_test, churn_model.predict_proba(xtec)[:, 1])),
                     "feature_importance": dict(sorted(zip(NUMERIC, churn_model.feature_importances_),
                                                        key=lambda item: item[1], reverse=True))}
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": risk_model, "preprocessor": risk_prep, "features": NUMERIC}, model_dir / "risk_model.pkl")
    joblib.dump({"model": churn_model, "preprocessor": churn_prep, "features": NUMERIC}, model_dir / "churn_model.pkl")
    metrics = {"risk": risk_metrics, "churn": churn_metrics}
    import json
    (model_dir / "customer_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "customers.csv")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(fit_customer_models(pd.read_csv(args.input), args.model_dir))


if __name__ == "__main__":
    main()
