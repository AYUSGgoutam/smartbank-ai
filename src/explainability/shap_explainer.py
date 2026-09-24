"""Explain fraud model predictions with SHAP feature attributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.prepare import engineer_features
from src.config import CATEGORICAL_TRANSACTION_FEATURES, NUMERIC_TRANSACTION_FEATURES


def explain_fraud(bundle: dict[str, Any], transaction: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Return local SHAP contributions, sorted by absolute impact on fraud score."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install SHAP to explain predictions") from exc
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    row = engineer_features(pd.DataFrame([transaction]))
    columns = NUMERIC_TRANSACTION_FEATURES + CATEGORICAL_TRANSACTION_FEATURES + [
        "amount_to_average_ratio", "log_amount", "is_weekend", "night_transaction"]
    transformed = preprocessor.transform(row[columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(transformed)
        base_value = np.asarray(explainer.expected_value).reshape(-1)[-1]
    except Exception:
        # Model agnostic permutation SHAP fallback supports the logistic baseline.
        explainer = shap.Explainer(lambda matrix: model.predict_proba(matrix)[:, 1], transformed)
        explanation = explainer(transformed, max_evals=2 * transformed.shape[1] + 1)
        raw_values = explanation.values
        base_value = np.asarray(explanation.base_values).reshape(-1)[0]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]
    names = bundle.get("feature_names", [f"feature_{i}" for i in range(len(values))])
    order = np.argsort(np.abs(values))[::-1][:top_k]
    return {"base_value": float(base_value),
            "contributions": [{"feature": names[i], "value": float(values[i]),
                               "direction": "increases" if values[i] > 0 else "decreases"}
                              for i in order]}
