"""Validated API input and output models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=40)
    customer_id: str = Field(min_length=1, max_length=32)
    timestamp: datetime
    amount: float = Field(gt=0, le=10_000_000)
    transaction_type: str
    merchant_category: str
    location: str
    device_type: str
    is_international: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    previous_transaction_count: int = Field(ge=0)
    average_transaction_amount: float = Field(gt=0)
    failed_transaction_count: int = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    distance_from_home: float = Field(ge=0)
    new_device: int = Field(ge=0, le=1)
    age: int = Field(ge=18, le=120)
    monthly_income: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=850)
    complaints: int = Field(ge=0)
    login_frequency: float = Field(ge=0)
    customer_service_calls: int = Field(ge=0)


class CustomerInput(BaseModel):
    customer_id: str = Field(min_length=1)
    age: int = Field(ge=18, le=120)
    account_age_days: int = Field(ge=0)
    monthly_income: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=850)
    total_transactions: int = Field(ge=0)
    average_transaction_amount: float = Field(ge=0)
    failed_transactions: int = Field(ge=0)
    complaints: int = Field(ge=0)
    login_frequency: float = Field(ge=0)
    customer_service_calls: int = Field(ge=0)


class PredictionResponse(BaseModel):
    score: float
    prediction: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "VERY HIGH"]
    reasons: list[dict] = Field(default_factory=list)


class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)
