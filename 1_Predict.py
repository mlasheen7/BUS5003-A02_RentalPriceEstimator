"""Predict page: estimate, why this price, and comparable suburbs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, so `app.utils` imports work
import plotly.graph_objects as go
import streamlit as st

from app.utils.app_state import (
    PROPERTY_LABELS, comparables_table, find_comparables, footer, get_client,
    get_data, get_explainer, get_model, setup_page,
)
from app.utils.shap_explainer import SUPPORTED_COMBINATIONS, find_suburb_row

setup_page("Predict")
df = get_data()
model = get_model()

st.title("📊 Estimate your rent")

suburbs = sorted(df["suburb"].unique())
c1, c2, c3 = st.columns([2, 1, 1])
suburb = c1.selectbox("Suburb", suburbs, index=suburbs.index("footscray") if "footscray" in suburbs else 0,
                      format_func=str.title)
bedrooms = c2.selectbox("Bedrooms", [1, 2, 3, 4], index=1)
# The source report publishes 1 bed only as a flat and 4 bed only as a house.
types = [t for t in ("flat", "house") if (bedrooms, t) in SUPPORTED_COMBINATIONS]
ptype = c3.radio("Property type", types, format_func=PROPERTY_LABELS.get, horizontal=True)

if not st.button("Estimate rent", type="primary"):
    st.caption("1 bedroom homes are only published as flats and 4 bedroom homes only as houses, "
               "so the property type options follow the bedrooms you pick.")
    footer()
    st.stop()

try:
    row = find_suburb_row(df, suburb, bedrooms, ptype)
except LookupError as e:
    st.warning(str(e))
    footer()
    st.stop()

explainer = get_explainer()
result = explainer.explain(explainer.feature_row(row))
pred = result.prediction
mae = model.test_mae or 0

m1, m2, m3 = st.columns(3)
m1.metric("Estimated weekly rent", f"${pred:,.0f}")
m2.metric("Likely range", f"${pred - mae:,.0f} to ${pred + mae:,.0f}", help="Estimate ± the model's average error on unseen data.")
gap = pred - row["median_weekly_rent"]
m3.metric("Recorded median rent", f"${row['median_weekly_rent']:,.0f}",
          delta=f"estimate is ${abs(gap):,.0f} {'above' if gap > 0 else 'below'}", delta_color="off")

# ---- Why this price ----
st.markdown("### 💡 Why this price?")
with st.spinner("Writing the explanation..."):
    response = get_client().explain(row, result)
st.info(response["explanation"])
if response["used_fallback"]:
    st.caption("Template explanation (AI explanation not available right now).")
else:
    st.caption(f"Written by {response['model']} from the model's factors below.")

drivers = result.top_drivers(6)[::-1]
fig = go.Figure(go.Bar(
    x=[d.effect for d in drivers],
    y=[f"{d.label} ({d.value})" for d in drivers],
    orientation="h",
    marker_color=["#1a7f5a" if d.effect >= 0 else "#c0392b" for d in drivers],
    text=[f"{'+' if d.effect >= 0 else '−'}${abs(d.effect):,.0f}" for d in drivers],
    textposition="outside",
))
fig.update_layout(
    title=f"How each factor moved the estimate from the ${result.baseline:,.0f} starting point",
    xaxis_title="Effect on weekly estimate ($)", height=360, margin=dict(l=10, r=40, t=50, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Comparables ----
st.markdown("### 🏘️ Similar suburbs")
comps = find_comparables(df, row)
st.dataframe(comparables_table(comps), hide_index=True, use_container_width=True)
st.caption("Same bedrooms and property type, closest in SEIFA score and household income.")

footer()
