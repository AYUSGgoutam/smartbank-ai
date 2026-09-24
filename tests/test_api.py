"""FastAPI route and input validation smoke tests."""

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

import src.api.main as api_module
from src.api.main import app

client = TestClient(app)


def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_fraud_payload_is_rejected():
    response = client.post("/predict/fraud", json={"amount": -1})
    assert response.status_code == 422


def test_docs_are_available():
    assert client.get("/docs").status_code == 200


def test_fraud_prediction_route_persists_and_returns_score(monkeypatch):
    class DummyPreprocessor:
        def transform(self, frame):
            return [[0.0]]

    class DummyClassifier:
        def predict_proba(self, features):
            return np.asarray([[0.28, 0.72]])

    monkeypatch.setattr(api_module, "load_bundle", lambda _name: {
        "model": DummyClassifier(), "preprocessor": DummyPreprocessor(),
        "decision_threshold": .5, "feature_names": ["example"],
    })
    payload = {
        "transaction_id": "TEST_TX_API_01", "customer_id": "C_TEST_API", "timestamp": "2025-01-04T02:00:00Z",
        "amount": 85000, "transaction_type": "transfer", "merchant_category": "electronics",
        "location": "Mumbai", "device_type": "new_mobile", "is_international": 1,
        "account_age_days": 400, "previous_transaction_count": 30, "average_transaction_amount": 2000,
        "failed_transaction_count": 0, "hour": 2, "day_of_week": 5, "distance_from_home": 900,
        "new_device": 1, "age": 35, "monthly_income": 60000, "credit_score": 680,
        "complaints": 0, "login_frequency": 12, "customer_service_calls": 0,
    }
    response = client.post("/predict/fraud?explain=true", json=payload)
    assert response.status_code == 200
    assert response.json()["score"] == .72
    assert response.json()["prediction"] == "FRAUD"
    assert response.json()["reasons"]


def test_saved_customer_models_score_a_profile_when_trained():
    models = Path(__file__).resolve().parents[1] / "models"
    if not (models / "risk_model.pkl").exists() or not (models / "churn_model.pkl").exists():
        pytest.skip("Train customer models to run this integration check")
    customer = {"customer_id": "C_TEST_MODEL", "age": 35, "account_age_days": 900,
        "monthly_income": 60000, "credit_score": 680, "total_transactions": 10,
        "average_transaction_amount": 2000, "failed_transactions": 1, "complaints": 0,
        "login_frequency": 12, "customer_service_calls": 0}
    risk = client.post("/predict/risk", json=customer)
    churn = client.post("/predict/churn", json=customer)
    assert risk.status_code == 200
    assert 0 <= risk.json()["risk_score"] <= 100
    assert churn.status_code == 200
    assert 0 <= churn.json()["churn_probability"] <= 1
    assert churn.json()["important_features"]
