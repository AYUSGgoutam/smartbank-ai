"""Global page heading, search, date range and notification controls."""

from datetime import date

import pandas as pd
import streamlit as st


def render_header(page: str, transactions: pd.DataFrame) -> tuple[str, tuple[date, date] | None]:
    if transactions.empty:
        minimum, maximum = date.today(), date.today()
    else:
        minimum = transactions.timestamp.min().date()
        maximum = transactions.timestamp.max().date()
    h1, h2, h3, h4 = st.columns([1.1, 1.4, 1.15, .65], vertical_alignment="center")
    titles = {
        "Dashboard": ("Dashboard", "Welcome back, Ayush. Here's your banking risk overview."),
        "Transactions": ("Transactions", "Search and review activity across the selected data window."),
        "Customers": ("Customers", "Review customer profiles, risk scores and churn signals."),
        "Fraud Detection": ("Fraud detection", "Score a transaction with the current fraud and anomaly models."),
        "Risk Analysis": ("Risk analysis", "Explore customer risk levels and model-scored profiles."),
        "Model Performance": ("Model performance", "Review metrics calculated from held-out evaluation data."),
        "Settings": ("Settings", "Service connections and workspace configuration."),
    }
    title, subtitle = titles[page]
    h1.markdown(f"<div class='page-heading'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)
    search = h2.text_input("Search", placeholder="⌕  Search transaction, customer ID, or location…", label_visibility="collapsed", key="global_search")
    if minimum == maximum:
        date_range = (minimum, maximum)
        h3.markdown(f"<div class='date-chip'>◷ &nbsp;{maximum.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
    else:
        selected = h3.date_input("Date range", value=(minimum, maximum), min_value=minimum,
                                 max_value=maximum, label_visibility="collapsed", key="global_date_range")
        date_range = tuple(selected) if isinstance(selected, (tuple, list)) and len(selected) == 2 else None
    h4.markdown("<div class='header-user'><span class='notification-icon'>🔔</span><span class='avatar'>A</span><b>Ayush</b></div>", unsafe_allow_html=True)
    return search, date_range
