"""Stateful, model-backed prediction panels for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components.charts import render_risk_gauge
from frontend.components.data import api_request


def render_risk_prediction(customers: pd.DataFrame) -> None:
    st.markdown("<div class='card-title-row'><div><h3>Customer risk prediction</h3><p>Scored by the trained customer model</p></div><span class='ai-pill'>AI MODEL</span></div>", unsafe_allow_html=True)
    if customers.empty:
        st.info("Generate customer data to choose a profile.")
        return
    if "_risk_prediction" not in st.session_state:
        st.session_state["_risk_prediction"] = None
    ids = customers.customer_id.astype(str).tolist()
    selected = st.selectbox("Customer ID", ids, key="dashboard_customer_select", label_visibility="collapsed")
    record = customers[customers.customer_id.astype(str) == selected].iloc[0]
    payload = {name: (record[name].item() if hasattr(record[name], "item") else record[name]) for name in [
        "customer_id", "age", "account_age_days", "monthly_income", "credit_score", "total_transactions",
        "average_transaction_amount", "failed_transactions", "complaints", "login_frequency", "customer_service_calls"]}
    if st.button("Score customer profile", key="score_dashboard_customer", width="stretch"):
        risk = api_request("/predict/risk", payload)
        churn = api_request("/predict/churn", payload)
        if risk and churn:
            st.session_state["_risk_prediction"] = {"customer_id": selected, **risk, **churn}
    result = st.session_state.get("_risk_prediction")
    if not result or result.get("customer_id") != selected:
        st.markdown("<div class='gauge-empty'>Select a customer and run a score to view model predictions.</div>", unsafe_allow_html=True)
        st.caption(f"Profile credit score: {int(record.credit_score)} · {int(record.total_transactions):,} transactions")
        return
    score = float(result["risk_score"])
    render_risk_gauge(score, height=190)
    st.markdown(f"<div class='risk-band band-{result['risk_category'].lower().replace(' ', '-')}'>{result['risk_category']} RISK</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric("Credit score", f"{int(record.credit_score)}")
    b.metric("Churn probability", f"{result['churn_probability']:.1%}")


def render_ai_prediction(transactions: pd.DataFrame, customers: pd.DataFrame) -> None:
    st.markdown("<div class='card-title-row'><div><h3>AI prediction</h3><p>Current transaction · fraud and anomaly models</p></div><span class='ai-pill'>SMARTBANK AI</span></div>", unsafe_allow_html=True)
    if transactions.empty:
        st.info("Choose a transaction after generating data.")
        return
    choices = transactions.sort_values("timestamp", ascending=False).head(300)
    identifiers = choices.transaction_id.astype(str).tolist()
    selected = st.selectbox("Transaction", identifiers, key="dashboard_transaction_select", label_visibility="collapsed")
    record = choices[choices.transaction_id.astype(str) == selected].iloc[0]
    st.markdown(f"<div class='prediction-facts'><div><small>AMOUNT</small><b>₹{record.amount:,.0f}</b></div><div><small>LOCATION</small><b>{record.location}</b></div><div><small>DEVICE</small><b>{str(record.device_type).replace('_', ' ').title()}</b></div><div><small>TIME</small><b>{pd.Timestamp(record.timestamp).strftime('%I:%M %p')}</b></div></div>", unsafe_allow_html=True)
    if st.button("Analyze with AI", key="analyze_dashboard_tx", type="primary", width="stretch"):
        payload = record.drop(labels=["fraud"], errors="ignore").to_dict()
        payload["timestamp"] = pd.Timestamp(payload["timestamp"]).isoformat()
        customer_rows = customers[customers.customer_id.astype(str) == str(payload["customer_id"])] if not customers.empty else pd.DataFrame()
        if not customer_rows.empty:
            customer = customer_rows.iloc[0]
            customer_to_transaction = {"account_age_days": "account_age_days", "previous_transaction_count": "total_transactions",
                "average_transaction_amount": "average_transaction_amount", "age": "age", "monthly_income": "monthly_income",
                "credit_score": "credit_score", "complaints": "complaints", "login_frequency": "login_frequency",
                "customer_service_calls": "customer_service_calls"}
            for transaction_key, customer_key in customer_to_transaction.items():
                if pd.isna(payload.get(transaction_key)) and customer_key in customer:
                    payload[transaction_key] = customer[customer_key]
        if pd.isna(payload.get("merchant_category")):
            payload["merchant_category"] = "unknown"
        for key in ["distance_from_home", "monthly_income"]:
            if pd.isna(payload.get(key)):
                payload[key] = float(transactions[key].median())
        payload = {key: value.item() if hasattr(value, "item") else value for key, value in payload.items()}
        fraud = api_request("/predict/fraud?explain=true", payload)
        anomaly = api_request("/predict/anomaly", payload)
        if fraud:
            st.session_state["_dashboard_ai_result"] = {"transaction_id": selected, "fraud": fraud, "anomaly": anomaly}
    result = st.session_state.get("_dashboard_ai_result")
    if result and result.get("transaction_id") == selected:
        fraud = result["fraud"]
        st.markdown(f"<div class='prediction-score'><small>FRAUD PROBABILITY</small><strong>{fraud['score']:.1%}</strong><span class='risk-chip'>{fraud['risk_level']} RISK</span></div>", unsafe_allow_html=True)
        if fraud.get("reasons"):
            st.markdown("<div class='reason-title'>Top model explanation</div>", unsafe_allow_html=True)
            for reason in fraud["reasons"][:4]:
                arrow = "↑" if reason.get("direction") == "increases" else "↓"
                st.markdown(f"<div class='reason-row'><span>{arrow}</span><b>{reason['feature'].replace('_', ' ').title()}</b><small>{reason['direction']} score</small></div>", unsafe_allow_html=True)
        anomaly = result.get("anomaly")
        if anomaly:
            st.caption(f"Isolation Forest anomaly score: {anomaly['anomaly_score']:.3f} · threshold {anomaly['threshold']:.3f}")
    else:
        st.markdown("<div class='prediction-placeholder'>Run the model to see its fraud probability, anomaly score and explanation.</div>", unsafe_allow_html=True)
