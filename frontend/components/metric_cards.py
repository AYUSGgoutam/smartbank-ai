"""Custom HTML statistic cards with real data and compact comparisons."""

import html

import streamlit as st


def render_metric_cards(items: list[dict]) -> None:
    columns = st.columns(4, gap="medium")
    for column, item in zip(columns, items):
        delta = item.get("delta", "")
        delta_html = f"<span class='metric-delta {html.escape(item.get('tone', 'neutral'))}'>{html.escape(delta)}</span>" if delta else ""
        column.markdown(f"""
        <div class="metric-card accent-{html.escape(item['accent'])}">
          <div class="metric-top"><span>{html.escape(item['title'])}</span><i>{html.escape(item['icon'])}</i></div>
          <div class="metric-value">{html.escape(str(item['value']))}</div>
          <div class="metric-bottom">{delta_html}<small>{html.escape(item['caption'])}</small></div>
        </div>""", unsafe_allow_html=True)
