"""Searchable and filterable transaction register."""

import pandas as pd
import streamlit as st


def render(transactions: pd.DataFrame) -> None:
    st.markdown("<div class='surface-heading'><h2>Transaction register</h2><p>Filter synthetic transaction records by date, amount, location, type and status.</p></div>", unsafe_allow_html=True)
    if transactions.empty:
        st.info("No transactions are available in the selected date and search filters.")
        return
    c1, c2, c3, c4 = st.columns(4)
    locations = c1.multiselect("Location", sorted(transactions.location.dropna().unique()))
    kinds = c2.multiselect("Transaction type", sorted(transactions.transaction_type.dropna().unique()))
    status = c3.selectbox("Dataset label", ["All", "Fraud", "Normal"])
    min_amount = float(transactions.amount.min())
    amount = c4.number_input("Minimum amount (₹)", min_value=0.0, value=min_amount)
    result = transactions[transactions.amount.ge(amount)]
    if locations: result = result[result.location.isin(locations)]
    if kinds: result = result[result.transaction_type.isin(kinds)]
    if status != "All": result = result[result.fraud == (status == "Fraud")]
    st.caption(f"{len(result):,} records · dataset labels are synthetic ground truth, not live model output.")
    visible = [column for column in ["timestamp", "transaction_id", "customer_id", "amount", "location",
        "transaction_type", "device_type", "fraud", "new_device", "is_international"] if column in result]
    st.dataframe(result.sort_values("timestamp", ascending=False)[visible], width="stretch",
                 hide_index=True, height=620)
