"""Transaction input form using the existing FastAPI fraud/anomaly endpoints."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from frontend.components.data import api_request


def render() -> None:
    st.markdown("<div class='surface-heading'><h2>Fraud investigation</h2><p>Score a transaction with the saved fraud, SHAP and anomaly models.</p></div>", unsafe_allow_html=True)
    with st.container(border=True):
        with st.form("redesign_fraud_form"):
            c1, c2, c3 = st.columns(3)
            customer_id = c1.text_input("Customer ID", "C0000001")
            amount = c2.number_input("Transaction amount (₹)", min_value=1.0, value=85000.0)
            location = c3.selectbox("Location", ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"])
            transaction_type = c1.selectbox("Transaction type", ["card", "online", "transfer", "atm", "wallet"])
            merchant = c2.selectbox("Merchant category", ["grocery", "fuel", "travel", "electronics", "dining", "cash", "utilities", "retail"])
            device = c3.selectbox("Device type", ["known_mobile", "known_web", "new_mobile", "new_web", "pos_terminal"])
            international = c1.checkbox("International transaction")
            credit = c2.number_input("Credit score", 300, 850, 680)
            age = c3.number_input("Customer age", 18, 100, 35)
            submitted = st.form_submit_button("Analyze transaction", type="primary", width="stretch")
    if submitted:
        now = datetime.now(timezone.utc)
        payload = {"transaction_id": f"UI-{now.strftime('%Y%m%d%H%M%S%f')}", "customer_id": customer_id,
            "timestamp": now.isoformat(), "amount": amount, "transaction_type": transaction_type,
            "merchant_category": merchant, "location": location, "device_type": device,
            "is_international": int(international), "account_age_days": 600, "previous_transaction_count": 40,
            "average_transaction_amount": 2200, "failed_transaction_count": 0, "hour": now.hour,
            "day_of_week": now.weekday(), "distance_from_home": 850 if international else 25,
            "new_device": int(device.startswith("new_")), "age": age, "monthly_income": 60000,
            "credit_score": credit, "complaints": 0, "login_frequency": 12, "customer_service_calls": 0}
        fraud = api_request("/predict/fraud?explain=true", payload)
        anomaly = api_request("/predict/anomaly", payload)
        if fraud:
            st.markdown("### Model result")
            cols = st.columns(3)
            cols[0].metric("Fraud probability", f"{fraud['score']:.1%}")
            cols[1].metric("Risk level", fraud["risk_level"])
            cols[2].metric("Anomaly score", f"{anomaly['anomaly_score']:.3f}" if anomaly else "Unavailable")
            st.markdown(f"<div class='prediction-score'><small>PREDICTION</small><strong>{fraud['prediction']}</strong><span class='risk-chip'>{fraud['risk_level']}</span></div>", unsafe_allow_html=True)
            st.markdown("#### SHAP contributing features")
            st.dataframe(pd.DataFrame(fraud.get("reasons", [])), width="stretch", hide_index=True)
