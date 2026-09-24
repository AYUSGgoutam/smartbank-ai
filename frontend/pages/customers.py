"""Customer lookup with live risk and churn model scoring."""

import pandas as pd
import streamlit as st

from frontend.components.charts import render_risk_gauge
from frontend.components.data import api_request


def _payload(row: pd.Series) -> dict:
    keys = ["customer_id", "age", "account_age_days", "monthly_income", "credit_score", "total_transactions",
            "average_transaction_amount", "failed_transactions", "complaints", "login_frequency", "customer_service_calls"]
    return {key: row[key].item() if hasattr(row[key], "item") else row[key] for key in keys}


def render(customers: pd.DataFrame, transactions: pd.DataFrame) -> None:
    st.markdown("<div class='surface-heading'><h2>Customer intelligence</h2><p>Profile data with model-backed risk and churn scores.</p></div>", unsafe_allow_html=True)
    if customers.empty:
        st.info("Generate the dataset to browse customer profiles.")
        return
    customer_id = st.selectbox("Customer ID", customers.customer_id.astype(str).tolist(), key="customers_page_id")
    row = customers[customers.customer_id.astype(str) == customer_id].iloc[0]
    profile, score_panel = st.columns([1.2, 1], gap="large")
    with profile:
        with st.container(border=True):
            st.markdown(f"### Customer `{customer_id}`")
            cols = st.columns(3)
            cols[0].metric("Age", int(row.age))
            cols[1].metric("Credit score", int(row.credit_score))
            cols[2].metric("Transactions", f"{int(row.total_transactions):,}")
            cols[0].metric("Monthly income", f"₹{row.monthly_income:,.0f}")
            cols[1].metric("Average transaction", f"₹{row.average_transaction_amount:,.0f}")
            cols[2].metric("Complaints", int(row.complaints))
    with score_panel:
        with st.container(border=True):
            if st.button("Run customer risk and churn models", type="primary", key="customer_run_models"):
                risk = api_request("/predict/risk", _payload(row))
                churn = api_request("/predict/churn", _payload(row))
                if risk and churn:
                    st.session_state["customer_page_scores"] = {"id": customer_id, "risk": risk, "churn": churn}
            scores = st.session_state.get("customer_page_scores")
            if scores and scores["id"] == customer_id:
                render_risk_gauge(float(scores["risk"]["risk_score"]), 220)
                band = scores["risk"]["risk_category"]
                st.markdown(f"<div class='risk-band band-{band.lower().replace(' ', '-')}'>{band.upper()} RISK</div>", unsafe_allow_html=True)
                st.metric("Churn probability", f"{scores['churn']['churn_probability']:.1%}")
                if scores["churn"].get("important_features"):
                    st.caption("Most influential churn model features")
                    st.dataframe(pd.DataFrame(scores["churn"]["important_features"]), hide_index=True, width="stretch")
            else:
                st.info("Run the models to calculate this customer's scores.")
    factors = []
    if row.credit_score < 600: factors.append("Credit score below 600")
    if row.failed_transactions >= 3: factors.append("Several failed transactions")
    if row.complaints >= 2: factors.append("Multiple complaints")
    if row.customer_service_calls >= 3: factors.append("Frequent service calls")
    st.markdown("#### Profile signals")
    st.write(" · ".join(factors) if factors else "No elevated profile signals in the generated record.")
    recent = transactions[transactions.customer_id.astype(str) == customer_id].sort_values("timestamp", ascending=False).head(20)
    st.markdown("#### Recent transactions")
    st.dataframe(recent, width="stretch", hide_index=True)
