"""Service and artifact availability page."""

import streamlit as st

from frontend.components.data import API_URL, MODEL_DIR, api_request


def render() -> None:
    st.markdown("<div class='surface-heading'><h2>Workspace settings</h2><p>Runtime connectivity and model artifacts.</p></div>", unsafe_allow_html=True)
    health = api_request("/health")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### API service")
            st.write("URL:", API_URL)
            st.write("Health:", health or "Unavailable")
    with c2:
        with st.container(border=True):
            st.markdown("### Model artifacts")
            for name in ["fraud_model.pkl", "risk_model.pkl", "churn_model.pkl", "anomaly_model.pkl", "lstm_model.keras", "autoencoder.keras"]:
                st.write(("●" if (MODEL_DIR / name).exists() else "○"), name)
    st.caption("Configure the API target with SMARTBANK_API_URL and database settings with DATABASE_URL in .env.")
