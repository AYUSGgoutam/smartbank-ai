"""Honest comparison table from persisted evaluation artifacts."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.data import api_request, read_json_artifact


def render() -> None:
    st.markdown("<div class='surface-heading'><h2>Model performance</h2><p>Only metrics written by real evaluation runs are shown.</p></div>", unsafe_allow_html=True)
    fraud = api_request("/model-performance") or read_json_artifact("fraud_model_metrics.json")
    rows = []
    if fraud and fraud.get("models"):
        for item in fraud["models"]:
            rows.append({"Model": item["model"], "Type": "Fraud · supervised",
                **{name: item.get(name) for name in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}})
    lstm = read_json_artifact("lstm_metrics.json")
    if lstm:
        rows.append({"Model": "LSTM", "Type": "Fraud · sequence", "accuracy": None,
            "precision": None, "recall": lstm.get("recall"), "f1": None,
            "roc_auc": lstm.get("roc_auc"), "pr_auc": lstm.get("pr_auc")})
    anomaly = read_json_artifact("anomaly_model_metrics.json")
    if anomaly and anomaly.get("autoencoder"):
        rows.append({"Model": "Autoencoder", "Type": "Anomaly · reconstruction", "accuracy": None,
            "precision": None, "recall": None, "f1": None, "roc_auc": None, "pr_auc": None})
    if not rows:
        st.info("No evaluation artifacts found. Train the models from the project root.")
        return
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={name: st.column_config.NumberColumn(name.upper().replace("_", "-"), format="%.3f")
                                for name in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]})
    st.caption(f"Selected fraud model by PR-AUC: {fraud.get('selected_model', '—') if fraud else '—'}. Missing values mean the metric was not evaluated or saved for that model type.")
    customer = read_json_artifact("customer_model_metrics.json")
    if customer:
        c1, c2 = st.columns(2)
        c1.metric("Risk model · validation MAE", f"{customer['risk']['mae']:.2f}")
        c2.metric("Churn model · validation ROC-AUC", f"{customer['churn']['roc_auc']:.3f}")
        importance = customer["churn"].get("feature_importance", {})
        if importance:
            st.markdown("#### Churn feature importance")
            features = pd.DataFrame({"Feature": list(importance), "Importance": list(importance.values())})
            fig = px.bar(features, x="Importance", y="Feature", orientation="h", template="plotly_dark",
                         color="Importance", color_continuous_scale=["#253957", "#73a0ff"])
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=8, r=8, t=12, b=8), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
