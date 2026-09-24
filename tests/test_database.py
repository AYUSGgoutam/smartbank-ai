"""SQLAlchemy schema and local database operation tests."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from database.database import Base
from database.models import Customer


def test_customer_database_round_trip():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Customer(customer_id="C_TEST", age=30, credit_score=700,
                             total_transactions=12, average_transaction_amount=100.0,
                             risk_score=20.0, churn_probability=.1, profile_json={"tier": "silver"}))
        session.commit()
        saved = session.scalar(select(Customer).where(Customer.customer_id == "C_TEST"))
        assert saved is not None
        assert saved.profile_json["tier"] == "silver"
