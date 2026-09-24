"""Recent fraud alert list with database data and honest dataset fallback."""

import html

import pandas as pd
import streamlit as st

from frontend.components.data import api_request, fmt_money


def _time(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%I:%M %p")
    except (TypeError, ValueError):
        return "—"


def render_alerts(transactions: pd.DataFrame, limit: int = 5) -> None:
    st.markdown("<div class='card-title-row'><div><h3>Recent fraud alerts</h3><p>Latest flagged activity</p></div><span class='alert-count'>LIVE</span></div>", unsafe_allow_html=True)
    stored = api_request(f"/alerts?limit={limit}")
    items = []
    if stored:
        for alert in stored[:limit]:
            details = alert.get("details") or {}
            items.append({"level": alert.get("risk_level", "HIGH"), "title": "Model fraud alert",
                "amount": details.get("amount", 0), "location": details.get("location", "Unknown"),
                "time": _time(alert.get("created_at")), "id": alert.get("transaction_id", "")})
    elif not transactions.empty:
        flagged = transactions[transactions.fraud.astype(bool)].sort_values("timestamp", ascending=False).head(limit)
        for _, row in flagged.iterrows():
            signal = "New device" if row.get("new_device", 0) else "International activity" if row.get("is_international", 0) else "Unusual transaction"
            items.append({"level": "HIGH", "title": f"Dataset flagged · {signal}", "amount": row.amount,
                "location": row.location, "time": _time(row.timestamp), "id": row.transaction_id})
    if not items:
        st.markdown("<div class='empty-state'>No fraud alerts in this selected period.</div>", unsafe_allow_html=True)
        return
    for item in items:
        level = html.escape(str(item["level"]).upper())
        css = "alert-high" if level in {"HIGH", "VERY HIGH"} else "alert-medium"
        st.markdown(f"""
        <div class="alert-item"><span class="alert-pip {css}"></span><div class="alert-copy">
        <div class="alert-line"><b>{html.escape(item['title'])}</b><span class="alert-level {css}">{level}</span></div>
        <small>{fmt_money(float(item['amount'] or 0))} &nbsp;·&nbsp; {html.escape(str(item['location']))} &nbsp;·&nbsp; {html.escape(str(item['time']))}</small>
        </div></div>""", unsafe_allow_html=True)
