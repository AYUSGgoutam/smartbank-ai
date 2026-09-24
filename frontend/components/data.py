"""Data and API access shared by Streamlit components."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
API_URL = os.getenv("SMARTBANK_API_URL", "http://localhost:8000").rstrip("/")


@st.cache_data(ttl=60)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load locally generated source CSVs; absent data is represented by empty frames."""
    transaction_path = DATA_DIR / "transactions.csv"
    customer_path = DATA_DIR / "customers.csv"
    transactions = pd.read_csv(transaction_path, parse_dates=["timestamp"]) if transaction_path.exists() else pd.DataFrame()
    customers = pd.read_csv(customer_path) if customer_path.exists() else pd.DataFrame()
    if not transactions.empty:
        transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True, errors="coerce")
        transactions["fraud"] = transactions["fraud"].astype(bool)
    return transactions, customers


def api_request(path: str, payload: dict[str, Any] | None = None) -> Any | None:
    """Call the existing API and return JSON, or None with a compact UI message."""
    try:
        if payload is None:
            response = requests.get(f"{API_URL}{path}", timeout=8)
        else:
            response = requests.post(f"{API_URL}{path}", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.warning(f"API is unavailable: {exc}")
        return None


def read_json_artifact(name: str) -> dict[str, Any] | None:
    path = MODEL_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def filter_data(transactions: pd.DataFrame, customers: pd.DataFrame,
                date_range: tuple | None, search: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply dashboard date and global ID/location search filters."""
    tx = transactions.copy()
    cust = customers.copy()
    if date_range and len(date_range) == 2 and not tx.empty:
        start = pd.Timestamp(date_range[0], tz="UTC")
        end = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        tx = tx[tx.timestamp.ge(start) & tx.timestamp.lt(end)]
    query = search.strip().casefold()
    if query:
        if not tx.empty:
            mask = (tx.transaction_id.astype(str).str.casefold().str.contains(query, na=False)
                    | tx.customer_id.astype(str).str.casefold().str.contains(query, na=False)
                    | tx.location.astype(str).str.casefold().str.contains(query, na=False))
            tx = tx[mask]
        if not cust.empty:
            cust = cust[cust.customer_id.astype(str).str.casefold().str.contains(query, na=False)]
    return tx, cust


def fmt_money(value: float) -> str:
    return f"₹{value:,.0f}"
