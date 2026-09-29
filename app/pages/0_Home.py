"""Home page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, so `app.utils` imports work
import streamlit as st

from app.utils.app_state import ai_is_configured, footer, get_data, get_model, setup_page

setup_page("Home")
df = get_data()
model = get_model()

st.title("🏠 Rental Price Estimator")
st.markdown(
    "Get a fair weekly rent estimate for a Victorian suburb, see which factors drove it, "
    "and compare it with similar suburbs."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Suburbs covered", df["suburb"].nunique())
c2.metric("Rent records", f"{len(df):,}")
c3.metric("Typical error", f"±${model.test_mae:.0f}/wk" if model.test_mae else "n/a")
c4.metric("Accuracy (R²)", f"{model.test_r2:.2f}" if model.test_r2 else "n/a")

st.markdown("### How it works")
st.markdown(
    "1. **Predict**: pick a suburb, bedrooms and property type to get an estimate.\n"
    "2. **Why this price**: SHAP shows how much each factor pushed the estimate up or down, "
    "and an AI model turns that into a short plain English explanation.\n"
    "3. **Comparables**: see suburbs with a similar income and socio economic profile on a map."
)
if st.button("Start estimating →", type="primary"):
    st.switch_page("pages/1_Predict.py")

if not ai_is_configured():
    st.caption("AI explanations are off (no OPENROUTER_API_KEY), so the app uses a template explanation.")
st.caption(f"Model: {model.version} ({model.source}).")
footer()
