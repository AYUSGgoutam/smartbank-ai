"""Shared project configuration and feature definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessing"
RANDOM_SEED = 42
FRAUD_TARGET = "fraud"
CHURN_TARGET = "churn"

NUMERIC_TRANSACTION_FEATURES = [
    "amount", "is_international", "account_age_days", "previous_transaction_count",
    "average_transaction_amount", "failed_transaction_count", "hour", "day_of_week",
    "distance_from_home", "new_device", "age", "monthly_income", "credit_score",
    "complaints", "login_frequency", "customer_service_calls",
]
CATEGORICAL_TRANSACTION_FEATURES = ["transaction_type", "merchant_category", "location", "device_type"]
TRANSACTION_FEATURES = NUMERIC_TRANSACTION_FEATURES + CATEGORICAL_TRANSACTION_FEATURES
