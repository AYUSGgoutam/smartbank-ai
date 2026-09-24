"""FastAPI routes for fraud/risk scoring and persisted banking records."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from database.database import Base, engine, get_db
from database.models import Customer, FraudAlert, ModelMetric, Prediction, Transaction
from src.api.schemas import CustomerInput, PredictionResponse, TransactionInput
from src.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / "models"
app = FastAPI(title="SmartBank AI", description="Fraud and risk intelligence API", version="0.2.0")
Base.metadata.create_all(bind=engine)


def load_bundle(filename: str) -> dict[str, Any] | None:
    path = MODEL_DIR / filename
    return joblib.load(path) if path.exists() else None


def risk_band(score: float) -> str:
    if score <= 25: return "LOW"
    if score <= 50: return "MEDIUM"
    if score <= 75: return "HIGH"
    return "VERY HIGH"


def record_prediction(db: Session, entity_id: str, kind: str, score: float,
                      predicted_class: str, explanations: dict | None = None) -> None:
    db.add(Prediction(entity_id=entity_id, prediction_type=kind, score=score,
                      predicted_class=predicted_class, explanations=explanations))
    db.commit()


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "SmartBank AI", "docs": "/docs", "status": "ok"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "fraud_model_loaded": (MODEL_DIR / "fraud_model.pkl").exists()}


@app.post("/predict/fraud", response_model=PredictionResponse)
def predict_fraud(payload: TransactionInput, explain: bool = False, db: Session = Depends(get_db)):
    bundle = load_bundle("fraud_model.pkl")
    if bundle is None: raise HTTPException(503, "Fraud model not trained. Run src.models.train_fraud first.")
    row = pd.DataFrame([payload.model_dump()])
    from src.preprocessing.prepare import engineer_features
    from src.config import TRANSACTION_FEATURES
    columns = TRANSACTION_FEATURES + ["amount_to_average_ratio", "log_amount", "is_weekend", "night_transaction"]
    transformed = bundle["preprocessor"].transform(engineer_features(row)[columns])
    score = float(bundle["model"].predict_proba(transformed)[0, 1])
    reasons: list[dict] = []
    if explain:
        try:
            from src.explainability.shap_explainer import explain_fraud
            reasons = explain_fraud(bundle, payload.model_dump()) ["contributions"]
        except Exception as exc:
            LOGGER.exception("SHAP explanation failed")
            raise HTTPException(503, f"Explanation unavailable: {exc}") from exc
    band = risk_band(score * 100)
    prediction = "FRAUD" if score >= bundle.get("decision_threshold", .5) else "LEGITIMATE"
    transaction = Transaction(transaction_id=payload.transaction_id, customer_id=payload.customer_id,
        timestamp=payload.timestamp, amount=payload.amount, location=payload.location,
        transaction_type=payload.transaction_type, fraud=prediction == "FRAUD", payload_json=payload.model_dump(mode="json"))
    db.merge(transaction)
    record_prediction(db, payload.transaction_id, "fraud", score, prediction, {"reasons": reasons})
    if prediction == "FRAUD":
        db.add(FraudAlert(transaction_id=payload.transaction_id, risk_level=band,
                          probability=score, details={"location": payload.location, "amount": payload.amount}))
        db.commit()
    return PredictionResponse(score=score, prediction=prediction, risk_level=band, reasons=reasons)


@app.post("/predict/risk")
def predict_risk(payload: CustomerInput, db: Session = Depends(get_db)):
    bundle = load_bundle("risk_model.pkl")
    if bundle is None: raise HTTPException(503, "Risk model not trained. Run src.models.train_customer first.")
    row = pd.DataFrame([payload.model_dump()])[bundle["features"]]
    score = float(np.clip(bundle["model"].predict(bundle["preprocessor"].transform(row))[0], 0, 100))
    record_prediction(db, payload.customer_id, "risk", score, risk_band(score))
    return {"customer_id": payload.customer_id, "risk_score": score, "risk_category": risk_band(score)}


@app.post("/predict/churn")
def predict_churn(payload: CustomerInput, db: Session = Depends(get_db)):
    bundle = load_bundle("churn_model.pkl")
    if bundle is None: raise HTTPException(503, "Churn model not trained. Run src.models.train_customer first.")
    row = pd.DataFrame([payload.model_dump()])[bundle["features"]]
    probability = float(bundle["model"].predict_proba(bundle["preprocessor"].transform(row))[0, 1])
    record_prediction(db, payload.customer_id, "churn", probability, "CHURN" if probability >= .5 else "RETAIN")
    importance = bundle["model"].feature_importances_
    top_features = sorted(zip(bundle["features"], importance), key=lambda item: item[1], reverse=True)[:5]
    return {"customer_id": payload.customer_id, "churn_probability": probability,
            "important_features": [{"feature": name, "importance": float(value)} for name, value in top_features]}


@app.post("/predict/anomaly")
def predict_anomaly(payload: TransactionInput):
    bundle = load_bundle("anomaly_model.pkl")
    if bundle is None: raise HTTPException(503, "Anomaly model not trained. Run src.models.train_anomaly --skip-autoencoder.")
    row = pd.DataFrame([payload.model_dump()])[bundle["features"]].replace([np.inf, -np.inf], np.nan)
    row = row.fillna(pd.Series(bundle["medians"]))
    row = row.clip(lower=pd.Series(bundle["lower"]), upper=pd.Series(bundle["upper"]), axis="columns")
    score = float(-bundle["model"].score_samples(bundle["scaler"].transform(row))[0])
    return {"anomaly_score": score, "threshold": bundle["threshold"], "is_anomaly": score >= bundle["threshold"]}


@app.post("/predict/lstm")
def predict_lstm(payload: TransactionInput):
    model_path = MODEL_DIR / "lstm_model.keras"
    prep_path = MODEL_DIR / "lstm_preprocessing.pkl"
    if not model_path.exists() or not prep_path.exists():
        raise HTTPException(503, "LSTM not trained. Run src.models.train_lstm.")
    import tensorflow as tf
    prep = joblib.load(prep_path)
    features = prep["features"]
    row_frame = pd.DataFrame([payload.model_dump()])
    if "amount_to_average_ratio" in features:
        row_frame["amount_to_average_ratio"] = min(payload.amount / max(payload.average_transaction_amount, 1), 100)
    if "night_transaction" in features:
        row_frame["night_transaction"] = int(payload.hour in [0, 1, 2, 3, 4, 5])
    row = row_frame[features].to_numpy(dtype="float32")
    sequence = np.repeat(row, prep["sequence_length"], axis=0)
    sequence = prep["scaler"].transform(sequence).reshape(1, prep["sequence_length"], len(features))
    score = float(tf.keras.models.load_model(model_path).predict(sequence, verbose=0)[0, 0])
    return {"fraud_probability": score, "note": "Single-event request repeated to form a window; customer history improves sequence predictions."}


@app.get("/transactions")
def transactions(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    rows = db.scalars(select(Transaction).order_by(desc(Transaction.timestamp)).limit(limit)).all()
    return [{"transaction_id": r.transaction_id, "customer_id": r.customer_id, "timestamp": r.timestamp,
             "amount": r.amount, "location": r.location, "transaction_type": r.transaction_type, "fraud": r.fraud} for r in rows]


@app.get("/customers")
def customers(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    rows = db.scalars(select(Customer).limit(limit)).all()
    return [{"customer_id": r.customer_id, "age": r.age, "credit_score": r.credit_score,
             "risk_score": r.risk_score, "churn_probability": r.churn_probability} for r in rows]


@app.get("/alerts")
def alerts(level: str | None = None, limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    statement = select(FraudAlert).order_by(desc(FraudAlert.created_at)).limit(limit)
    if level: statement = statement.where(FraudAlert.risk_level == level.upper())
    return [{"id": r.id, "transaction_id": r.transaction_id, "risk_level": r.risk_level,
             "probability": r.probability, "created_at": r.created_at, "details": r.details}
            for r in db.scalars(statement).all()]


@app.get("/model-performance")
def model_performance(db: Session = Depends(get_db)):
    path = MODEL_DIR / "fraud_model_metrics.json"
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    rows = db.scalars(select(ModelMetric).order_by(desc(ModelMetric.recorded_at))).all()
    return [{"model": r.model_name, "type": r.model_type, "metric": r.metric_name, "value": r.metric_value} for r in rows]
