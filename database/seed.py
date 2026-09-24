"""Load generated customer and transaction CSVs into the local SQL database."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from database.database import Base, SessionLocal, engine
from database.models import Customer, Transaction
from src.config import DEFAULT_DATA_DIR


def seed_database(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[int, int]:
    """Upsert generated profiles and transactions into their SQL tables."""
    customers = pd.read_csv(data_dir / "customers.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=["timestamp"])
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        for row in customers.to_dict(orient="records"):
            profile = {key: (None if pd.isna(value) else value) for key, value in row.items()}
            session.merge(Customer(customer_id=str(profile["customer_id"]), age=int(profile["age"]),
                credit_score=int(profile["credit_score"]), total_transactions=int(profile["total_transactions"]),
                average_transaction_amount=float(profile["average_transaction_amount"]),
                risk_score=float(profile["risk_score"]), profile_json=profile))
        for row in transactions.to_dict(orient="records"):
            record = {key: (None if pd.isna(value) else value) for key, value in row.items()}
            record["timestamp"] = pd.Timestamp(record["timestamp"]).isoformat()
            session.merge(Transaction(transaction_id=str(record["transaction_id"]),
                customer_id=str(record["customer_id"]), timestamp=pd.Timestamp(record["timestamp"]).to_pydatetime(),
                amount=float(record["amount"]), location=str(record["location"] or "Unknown"),
                transaction_type=str(record["transaction_type"]), fraud=bool(record["fraud"]), payload_json=record))
        session.commit()
        return len(customers), len(transactions)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    print("Imported customer and transaction rows:", seed_database(args.data_dir))


if __name__ == "__main__":
    main()
