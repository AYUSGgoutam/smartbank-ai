"""Compact branded navigation sidebar."""

from datetime import datetime

import requests
import streamlit as st

from frontend.components.data import API_URL
PAGES = [
    ("Dashboard", "▦"), ("Transactions", "⇄"), ("Customers", "♙"),
    ("Fraud Detection", "⌕"), ("Risk Analysis", "◈"),
    ("Model Performance", "⌁"), ("Settings", "⚙"),
]


def render_sidebar() -> str:
    try:
        online = requests.get(f"{API_URL}/health", timeout=1).ok
    except requests.RequestException:
        online = False
    status = "System online" if online else "API offline"
    status_class = "online-dot" if online else "offline-dot"
    with st.sidebar:
        st.markdown("""
        <div class="brand-mark"><span class="brand-icon">S</span><div><b>SmartBank <em>AI</em></b>
        <small>Fraud &amp; Risk Intelligence</small></div></div>
        <div class="side-section-label">WORKSPACE</div>
        """, unsafe_allow_html=True)
        labels = [f"{icon}   {name}" for name, icon in PAGES]
        chosen = st.radio("Navigation", labels, label_visibility="collapsed", key="main_navigation")
        page = PAGES[labels.index(chosen)][0]
        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="system-status"><span class="{status_class}"></span><b>{status}</b>
        <small>Updated {datetime.now().strftime('%H:%M')}</small></div>
        <div class="sidebar-user"><div class="avatar">A</div><div><b>Ayush</b><small>Risk analyst</small></div><span class="user-menu">···</span></div>
        """, unsafe_allow_html=True)
    return page
