"""Readable Plotly charts used across dashboard pages."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components.data import read_json_artifact

PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   margin=dict(l=8, r=8, t=42, b=8), font=dict(family="Inter, sans-serif", color="#aebbd0", size=12))


def render_transaction_trend(transactions: pd.DataFrame, height: int = 275) -> None:
    st.markdown("<div class='card-title-row'><div><h3>Transaction trend</h3><p>Daily volume and dataset fraud labels</p></div><span class='live-pill'>●&nbsp; Selected period</span></div>", unsafe_allow_html=True)
    if transactions.empty:
        st.info("No transaction rows in this date/search selection.")
        return
    daily = transactions.assign(date=transactions.timestamp.dt.date).groupby("date").agg(
        transactions=("transaction_id", "count"), fraud=("fraud", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.date, y=daily.transactions, mode="lines", name="All transactions",
        line=dict(color="#5b8cff", width=2.5), fill="tozeroy", fillcolor="rgba(73,125,255,.10)"))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.fraud, mode="lines", name="Fraud-labeled",
        line=dict(color="#ff647c", width=2.2)))
    fig.update_layout(**PLOT_LAYOUT, height=height, hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
                      xaxis_title=None, yaxis_title="Transactions", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#202c40"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _fraud_signal_types(fraud: pd.DataFrame) -> pd.Series:
    """Assign a single dominant, explicitly heuristic signal to each positive label."""
    if fraud.empty:
        return pd.Series(dtype="object")
    ratio = fraud.amount / fraud.average_transaction_amount.clip(lower=1)
    labels = np.select(
        [fraud.new_device.astype(bool), fraud.is_international.astype(bool),
         fraud.distance_from_home.gt(500), fraud.hour.isin([0, 1, 2, 3, 4, 5]),
         ratio.gt(3), fraud.transaction_type.eq("online")],
        ["New device", "International", "Location anomaly", "Unusual hour", "Amount anomaly", "Online transaction"],
        default="Other signals")
    return pd.Series(labels, index=fraud.index)


def render_fraud_types(transactions: pd.DataFrame, height: int = 275) -> None:
    st.markdown("<div class='card-title-row'><div><h3>Fraud signal mix</h3><p>Dominant behavior in fraud-labeled rows</p></div></div>", unsafe_allow_html=True)
    fraud = transactions[transactions.fraud.astype(bool)] if not transactions.empty else transactions
    if fraud.empty:
        st.info("No fraud-labeled rows in this selection.")
        return
    counts = _fraud_signal_types(fraud).value_counts().rename_axis("Signal").reset_index(name="Count")
    colors = ["#f06279", "#f4a75b", "#8b79f7", "#51b8c4", "#5b8cff", "#60738f", "#aebbd0"]
    fig = px.pie(counts, names="Signal", values="Count", hole=.68, color="Signal",
                 color_discrete_sequence=colors)
    fig.update_traces(textposition="inside", textinfo="percent", insidetextfont=dict(size=9, color="white"),
                      hovertemplate="%{label}<br>%{value:,} rows · %{percent}<extra></extra>")
    fig.update_layout(**PLOT_LAYOUT, height=height, showlegend=True,
        legend=dict(orientation="v", x=1.02, y=.5, font=dict(size=10)),
        annotations=[dict(text=f"{len(fraud):,}<br><span style='font-size:10px'>flagged</span>", x=.5, y=.5,
                          showarrow=False, font=dict(size=18, color="#eef4ff"))])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Rule-derived dominant signals, not ground-truth fraud typologies.")


def render_risk_distribution(customers: pd.DataFrame, height: int = 255) -> None:
    st.markdown("<div class='card-title-row'><div><h3>Customer risk distribution</h3><p>Portfolio profile scores</p></div></div>", unsafe_allow_html=True)
    if customers.empty or "risk_category" not in customers:
        st.info("Customer risk profile data is not available.")
        return
    order = ["Low", "Medium", "High", "Very High"]
    counts = customers.risk_category.value_counts().reindex(order, fill_value=0).rename_axis("Risk band").reset_index(name="Customers")
    fig = px.bar(counts, x="Risk band", y="Customers", color="Risk band", category_orders={"Risk band": order},
        color_discrete_map={"Low": "#42c89a", "Medium": "#e5b253", "High": "#f18b54", "Very High": "#ef5d73"})
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:,} customers<extra></extra>")
    fig.update_layout(**PLOT_LAYOUT, height=height, showlegend=False, xaxis_title=None, yaxis_title="Customers",
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#202c40"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_model_performance_chart(height: int = 255) -> None:
    metrics = read_json_artifact("fraud_model_metrics.json")
    st.markdown("<div class='card-title-row'><div><h3>Top ML/DL model performance</h3><p>Held-out ROC-AUC and PR-AUC</p></div></div>", unsafe_allow_html=True)
    if not metrics or not metrics.get("models"):
        st.info("Train the fraud models to populate this panel.")
        return
    frame = pd.DataFrame(metrics["models"])
    lstm = read_json_artifact("lstm_metrics.json")
    if lstm:
        frame = pd.concat([frame, pd.DataFrame([{"model": "LSTM", "roc_auc": lstm.get("roc_auc"),
                                                 "pr_auc": lstm.get("pr_auc")}])], ignore_index=True)
    plot = frame.melt(id_vars="model", value_vars=["roc_auc", "pr_auc"],
                      var_name="Metric", value_name="Score")
    fig = px.bar(plot, x="Metric", y="Score", color="model", barmode="group",
                 color_discrete_sequence=["#598bff", "#a17cf7", "#43bea3"])
    fig.update_layout(**PLOT_LAYOUT, height=height, yaxis=dict(range=[0, 1], gridcolor="#202c40"),
                      xaxis=dict(showgrid=False), legend=dict(orientation="h", y=1.18, x=0),
                      xaxis_title=None, yaxis_title="Score")
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f}<extra></extra>")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"Selected by PR-AUC: {metrics.get('selected_model', '—')}")


def render_risk_gauge(score: float | None, height: int = 200) -> None:
    if score is None:
        st.markdown("<div class='gauge-empty'>Run a customer risk prediction to see its score.</div>", unsafe_allow_html=True)
        return
    color = "#ef6476" if score > 75 else "#f19a5b" if score > 50 else "#e6b44f" if score > 25 else "#45c69a"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix": "%", "font": {"size": 30, "color": "#f3f6fc"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#6c7a91"},
          "bar": {"color": color, "thickness": .22}, "bgcolor": "#121c2b", "borderwidth": 0,
          "steps": [{"range": [0, 25], "color": "#16332f"}, {"range": [25, 50], "color": "#353223"},
                    {"range": [50, 75], "color": "#3b2b25"}, {"range": [75, 100], "color": "#3b232d"}]},
        domain={"x": [0, 1], "y": [0, 1]}))
    fig.update_layout(**{**PLOT_LAYOUT, "height": height, "margin": dict(l=20, r=20, t=15, b=5)})
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
