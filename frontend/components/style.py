"""Shared premium dark theme for the Streamlit frontend."""

from pathlib import Path

import streamlit as st


def inject_styles() -> None:
    css = Path(__file__).resolve().parents[1] / "assets" / "style.css"
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
