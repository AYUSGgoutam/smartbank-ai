"""Main full-width fraud and risk intelligence overview."""

import pandas as pd
import streamlit as st

from frontend.components.alerts import render_alerts
from frontend.components.charts import (render_fraud_types, render_model_performance_chart,
    render_risk_distribution, render_transaction_trend)
from frontend.components.metric_cards import render_metric_cards
from frontend.components.prediction_card import render_ai_prediction, render_risk_prediction
from frontend.components.tables import render_recent_transactions


def _period_change(current: int | float, previous: int | float, suffix: str = "%") -> tuple[str, str]:
    if not previous:
        return ("No prior window", "neutral")
    change = (current - previous) / previous * 100
    return (f"{'+' if change > 0 else ''}{change:.1f}{suffix} vs prior 7d", "good" if change < 0 else "bad" if change > 0 else "neutral")


def _summary(transactions: pd.DataFrame, customers: pd.DataFrame) -> list[dict]:
    if transactions.empty:
        return [
            {"title": "Total transactions", "value": "0", "icon": "⇄", "caption": "No rows in selected range", "accent": "blue"},
            {"title": "Fraud detected", "value": "0", "icon": "⌁", "caption": "No positive labels", "accent": "red"},
            {"title": "High risk customers", "value": "0", "icon": "◈", "caption": "Current customer profiles", "accent": "orange"},
            {"title": "Fraud rate", "value": "—", "icon": "%", "caption": "No rows to calculate rate", "accent": "green"},
        ]
    latest = transactions.timestamp.max().normalize() + pd.Timedelta(days=1)
    current_start, previous_start = latest - pd.Timedelta(days=7), latest - pd.Timedelta(days=14)
    current = transactions[transactions.timestamp.ge(current_start)]
    previous = transactions[transactions.timestamp.ge(previous_start) & transactions.timestamp.lt(current_start)]
    current_fraud = float(current.fraud.sum())
    previous_fraud = float(previous.fraud.sum())
    current_rate = float(current.fraud.mean() * 100) if len(current) else 0.0
    previous_rate = float(previous.fraud.mean() * 100) if len(previous) else 0.0
    if len(previous):
        rate_delta = current_rate - previous_rate
        rate_change = (f"{'+' if rate_delta > 0 else ''}{rate_delta:.2f} pp vs prior 7d",
                       "bad" if rate_delta > 0 else "good" if rate_delta < 0 else "neutral")
    else:
        rate_change = ("No prior window", "neutral")
    high_risk = int(customers.risk_score.ge(51).sum()) if not customers.empty and "risk_score" in customers else 0
    tx_delta = _period_change(len(current), len(previous))
    fraud_delta = _period_change(current_fraud, previous_fraud)
    return [
        {"title": "Total transactions", "value": f"{len(transactions):,}", "icon": "⇄", "caption": "In selected date range", "delta": tx_delta[0], "tone": tx_delta[1], "accent": "blue"},
        {"title": "Fraud detected", "value": f"{int(transactions.fraud.sum()):,}", "icon": "⌁", "caption": "Synthetic dataset positive labels", "delta": fraud_delta[0], "tone": fraud_delta[1], "accent": "red"},
        {"title": "High risk customers", "value": f"{high_risk:,}", "icon": "◈", "caption": "Profile score above 50 / 100", "accent": "orange"},
        {"title": "Fraud rate", "value": f"{transactions.fraud.mean():.2%}", "icon": "%", "caption": "Fraud labels in selected range", "delta": rate_change[0], "tone": rate_change[1], "accent": "green"},
    ]


def render(transactions: pd.DataFrame, customers: pd.DataFrame) -> None:
    render_metric_cards(_summary(transactions, customers))
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    trend, fraud_mix, alert_col = st.columns([1.45, 1.05, .92], gap="medium")
    with trend:
        with st.container(border=True):
            render_transaction_trend(transactions, 265)
    with fraud_mix:
        with st.container(border=True):
            render_fraud_types(transactions, 265)
    with alert_col:
        with st.container(border=True):
            render_alerts(transactions, 5)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    risk, performance, prediction = st.columns([1.05, 1.35, .92], gap="medium")
    with risk:
        with st.container(border=True):
            render_risk_distribution(customers, 248)
    with performance:
        with st.container(border=True):
            render_model_performance_chart(248)
    with prediction:
        with st.container(border=True):
            render_risk_prediction(customers)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    recent, ai = st.columns([1.65, .92], gap="medium")
    with recent:
        with st.container(border=True):
            render_recent_transactions(transactions, customers, 8)
    with ai:
        with st.container(border=True):
            render_ai_prediction(transactions, customers)
