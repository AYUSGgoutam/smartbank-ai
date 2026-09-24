"""Entry point for the redesigned SmartBank AI Streamlit experience."""

from __future__ import annotations

import streamlit as st

from frontend.components.data import filter_data, load_data
from frontend.components.header import render_header
from frontend.components.sidebar import render_sidebar
from frontend.components.style import inject_styles
from frontend.pages import (customers, dashboard, fraud_detection, model_performance,
                             risk_analysis, settings, transactions as transactions_page)


def main() -> None:
    st.set_page_config(page_title="SmartBank AI", page_icon="🛡️", layout="wide",
                       initial_sidebar_state="expanded")
    inject_styles()
    page = render_sidebar()
    all_transactions, all_customers = load_data()
    search, date_range = render_header(page, all_transactions)
    st.markdown("<div class='header-rule'></div>", unsafe_allow_html=True)
    filtered_transactions, filtered_customers = filter_data(all_transactions, all_customers, date_range, search)
    if all_transactions.empty:
        st.warning("No transaction data found. Generate the local synthetic dataset with `py -3 -m src.data.generate`.")
    elif search.strip():
        st.caption(f"Search results · {len(filtered_transactions):,} transactions · {len(filtered_customers):,} customers")
    if page == "Dashboard":
        dashboard.render(filtered_transactions, filtered_customers)
    elif page == "Transactions":
        transactions_page.render(filtered_transactions)
    elif page == "Customers":
        customers.render(filtered_customers, filtered_transactions)
    elif page == "Fraud Detection":
        fraud_detection.render()
    elif page == "Risk Analysis":
        risk_analysis.render(filtered_customers)
    elif page == "Model Performance":
        model_performance.render()
    else:
        settings.render()


if __name__ == "__main__":
    main()
