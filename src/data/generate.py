"""Generate reproducible synthetic banking customers and transactions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_DIR, RANDOM_SEED

LOGGER = logging.getLogger(__name__)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform for converting risk to probabilities."""
    values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def generate_dataset(
    customer_count: int = 10_000,
    transaction_count: int = 100_000,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return customer and transaction frames with behavior-linked labels."""
    if customer_count < 1 or transaction_count < 1:
        raise ValueError("customer_count and transaction_count must both be positive")
    rng = np.random.default_rng(random_seed)
    customer_ids = np.array([f"C{i:07d}" for i in range(1, customer_count + 1)])
    age = np.clip(rng.normal(42, 14, customer_count).round(), 18, 90).astype(int)
    account_age = rng.integers(30, 6000, customer_count)
    income = np.clip(rng.lognormal(np.log(62_000), 0.55, customer_count), 12_000, 500_000).round(2)
    credit = np.clip(rng.normal(690, 75, customer_count), 300, 850).round().astype(int)
    customer_tx_count = rng.poisson(75, customer_count) + 1
    avg_amount = np.clip(rng.lognormal(np.log(2_200), 0.65, customer_count), 100, 30_000).round(2)
    failed = rng.poisson(1.4, customer_count)
    complaints = rng.poisson(0.7, customer_count)
    login_frequency = np.clip(rng.gamma(3.0, 3.5, customer_count), 0, 60).round(1)
    service_calls = rng.poisson(1.1, customer_count)
    churn_probability = _sigmoid(-3.4 + 0.42 * complaints + 0.30 * service_calls + 0.32 * failed - 0.035 * login_frequency)
    churn = rng.binomial(1, churn_probability)
    risk_score = np.clip(48 + (700 - credit) * .10 + complaints * 4.0 + failed * 3.5
                         + service_calls * 2.0 - login_frequency * .35
                         + rng.normal(0, 7, customer_count), 0, 100).round(1)
    customers = pd.DataFrame({
        "customer_id": customer_ids, "age": age, "account_age_days": account_age,
        "monthly_income": income, "credit_score": credit, "total_transactions": customer_tx_count,
        "average_transaction_amount": avg_amount, "failed_transactions": failed,
        "complaints": complaints, "login_frequency": login_frequency,
        "customer_service_calls": service_calls, "churn": churn, "risk_score": risk_score,
        "risk_category": pd.cut(risk_score, bins=[-1, 25, 50, 75, 100],
                                 labels=["Low", "Medium", "High", "Very High"]).astype(str),
    })

    customer_idx = rng.integers(0, customer_count, transaction_count)
    # Keep each customer's profile count consistent with assigned transaction rows.
    customers["total_transactions"] = np.bincount(customer_idx, minlength=customer_count)
    tx_ids = np.array([f"T{i:09d}" for i in range(1, transaction_count + 1)])
    timestamps = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 365 * 24 * 60, transaction_count), unit="m"
    )
    tx_hour = timestamps.hour.to_numpy()
    tx_dow = timestamps.dayofweek.to_numpy()
    types = rng.choice(["card", "online", "transfer", "atm", "wallet"], transaction_count, p=[.36, .25, .18, .12, .09])
    merchants = rng.choice(["grocery", "fuel", "travel", "electronics", "dining", "cash", "utilities", "retail"], transaction_count)
    locations = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"], transaction_count)
    devices = rng.choice(["known_mobile", "known_web", "new_mobile", "new_web", "pos_terminal"], transaction_count, p=[.36, .19, .12, .08, .25])
    new_device = np.isin(devices, ["new_mobile", "new_web"]).astype(int)
    international = rng.binomial(1, np.where(types == "travel", .22, .045))
    distance = np.clip(rng.gamma(2.0, 18.0, transaction_count) + international * rng.gamma(2, 700, transaction_count), 0, 8000).round(1)
    typical_amount = avg_amount[customer_idx]
    amount = np.clip(rng.lognormal(np.log(np.maximum(typical_amount, 1)), .72), 10, 250_000).round(2)
    tx_failed = rng.binomial(1, np.clip(.025 + failed[customer_idx] * .018, 0, .35))
    prev_count = rng.integers(0, 500, transaction_count)
    # Fraud propensity rises with combinations of behavioral signals, with noise.
    amount_ratio = np.log1p(amount / np.maximum(typical_amount, 1))
    risk_logit = (-5.1 + 1.05 * new_device + 1.4 * international + .9 * (distance > 500)
                  + .75 * ((tx_hour <= 4) | (tx_hour >= 23)) + .65 * (amount_ratio > 1.3)
                  + .5 * tx_failed + .35 * (credit[customer_idx] < 560)
                  + .25 * (types == "transfer") + rng.normal(0, .8, transaction_count))
    fraud = rng.binomial(1, _sigmoid(risk_logit))

    transactions = pd.DataFrame({
        "transaction_id": tx_ids, "customer_id": customer_ids[customer_idx], "timestamp": timestamps,
        "amount": amount, "transaction_type": types, "merchant_category": merchants,
        "location": locations, "device_type": devices, "is_international": international,
        "account_age_days": account_age[customer_idx], "previous_transaction_count": prev_count,
        "average_transaction_amount": typical_amount, "failed_transaction_count": tx_failed,
        "hour": tx_hour, "day_of_week": tx_dow, "distance_from_home": distance,
        "new_device": new_device, "fraud": fraud,
    })
    # Attach customer descriptors for model training and customer-centric EDA.
    for column in ("age", "monthly_income", "credit_score", "complaints", "login_frequency", "customer_service_calls"):
        transactions[column] = customers[column].to_numpy()[customer_idx]
    # Small realistic missingness is introduced only in descriptive fields, never IDs or targets.
    for column in ("monthly_income", "distance_from_home", "merchant_category"):
        mask = rng.random(transaction_count) < .002
        transactions.loc[mask, column] = np.nan
    LOGGER.info("Generated %s customers and %s transactions (fraud rate %.3f%%)", customer_count, transaction_count, fraud.mean() * 100)
    return customers, transactions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=10_000)
    parser.add_argument("--transactions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    customers, transactions = generate_dataset(args.customers, args.transactions, args.seed)
    customers.to_csv(args.output_dir / "customers.csv", index=False)
    transactions.to_csv(args.output_dir / "transactions.csv", index=False)
    LOGGER.info("Wrote dataset files to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
