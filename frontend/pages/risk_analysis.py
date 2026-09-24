"""Portfolio risk analysis with generated labels and selectable model scoring."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.data import api_request
from frontend.components.charts import PLOT_LAYOUT


def render(customers: pd.DataFrame) -> None:
    st.markdown("<div class='surface-heading'><h2>Customer risk analysis</h2><p>Profile distribution and individual model scoring.</p></div>", unsafe_allow_html=True)
    if customers.empty or "risk_score" not in customers:
        st.info("Customer risk data is unavailable. Regenerate the dataset with `python -m src.data.generate`.")
        return
    order = ["Low", "Medium", "High", "Very High"]
    counts = customers.risk_category.value_counts().reindex(order, fill_value=0).rename_axis("Risk band").reset_index(name="Customers")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        fig = px.bar(counts, x="Risk band", y="Customers", color="Risk band", category_orders={"Risk band": order},
            color_discrete_map={"Low": "#42c89a", "Medium": "#e5b253", "High": "#f18b54", "Very High": "#ef5d73"})
        fig.update_layout(**PLOT_LAYOUT, height=330, showlegend=False, xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(fig, width="stretch")
    with right:
        with st.container(border=True):
            selected = st.selectbox("Customer profile", customers.customer_id.astype(str).tolist(), key="risk_page_customer")
            row = customers[customers.customer_id.astype(str) == selected].iloc[0]
            st.write(f"Credit score: **{int(row.credit_score)}** · Complaints: **{int(row.complaints)}**")
            if st.button("Predict risk score", type="primary", key="risk_page_score"):
                payload = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in ["customer_id", "age", "account_age_days",
                    "monthly_income", "credit_score", "total_transactions", "average_transaction_amount", "failed_transactions",
                    "complaints", "login_frequency", "customer_service_calls"]}
                result = api_request("/predict/risk", payload)
                if result: st.session_state["risk_page_result"] = {"customer": selected, **result}
            result = st.session_state.get("risk_page_result")
            if result and result["customer"] == selected:
                st.metric("Predicted risk score", f"{result['risk_score']:.1f} / 100", result["risk_category"])
            else:
                st.caption("Scores are requested from the trained FastAPI model when you press the button.")
    st.markdown("#### Highest profile risk scores")
    st.dataframe(customers.sort_values("risk_score", ascending=False).head(100), width="stretch", hide_index=True)
