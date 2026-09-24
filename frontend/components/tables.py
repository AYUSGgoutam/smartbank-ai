"""Compact modern transaction table and status labels."""

import html

import pandas as pd
import streamlit as st

from frontend.components.data import fmt_money


def render_recent_transactions(transactions: pd.DataFrame, customers: pd.DataFrame, limit: int = 8) -> None:
    st.markdown("<div class='card-title-row'><div><h3>Recent transactions</h3><p>Latest activity in the selected data range</p></div></div>", unsafe_allow_html=True)
    if transactions.empty:
        st.info("No transactions match the current filters.")
        return
    recent = transactions.sort_values("timestamp", ascending=False).head(limit).copy()
    if not customers.empty and {"customer_id", "risk_score"}.issubset(customers.columns):
        recent = recent.merge(customers[["customer_id", "risk_score"]], on="customer_id", how="left")
    else:
        recent["risk_score"] = pd.NA
    rows = []
    for _, row in recent.iterrows():
        fraud = bool(row.fraud)
        status, status_css = ("FRAUD", "status-fraud") if fraud else ("NORMAL", "status-normal")
        risk = "—" if pd.isna(row.risk_score) else f"{float(row.risk_score):.0f}/100"
        rows.append(f"""<tr><td>{pd.Timestamp(row.timestamp).strftime('%d %b · %H:%M')}</td>
          <td class='mono'>{html.escape(str(row.customer_id))}</td><td class='amount'>{fmt_money(float(row.amount))}</td>
          <td>{html.escape(str(row.location))}</td><td>{html.escape(str(row.transaction_type).title())}</td>
          <td>{html.escape(str(row.device_type).replace('_', ' ').title())}</td>
          <td><span class='status-badge {status_css}'>{status}</span></td><td>{risk}</td></tr>""")
    st.markdown("""<div class='table-wrap'><table class='transactions-table'><thead><tr>
      <th>Date &amp; time</th><th>Customer ID</th><th>Amount</th><th>Location</th><th>Type</th>
      <th>Device</th><th>Dataset label</th><th>Customer risk</th></tr></thead><tbody>"""
      + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)
    st.caption("Customer risk is the generated profile score. Transaction status is the synthetic dataset label, not a live prediction.")
