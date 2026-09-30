"""Comparables page: similar suburbs on a map and a chart."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, so `app.utils` imports work
import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.app_state import PROPERTY_LABELS, comparables_table, find_comparables, footer, get_data, setup_page
from app.utils.shap_explainer import SUPPORTED_COMBINATIONS, find_suburb_row

setup_page("Comparables")
df = get_data()

st.title("🏘️ Compare similar suburbs")
st.write("Find suburbs with a similar income and socio economic profile, and see how their rents compare.")

suburbs = sorted(df["suburb"].unique())
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
suburb = c1.selectbox("Suburb", suburbs, index=suburbs.index("footscray") if "footscray" in suburbs else 0,
                      format_func=str.title)
bedrooms = c2.selectbox("Bedrooms", [1, 2, 3, 4], index=1)
types = [t for t in ("flat", "house") if (bedrooms, t) in SUPPORTED_COMBINATIONS]
ptype = c3.radio("Property type", types, format_func=PROPERTY_LABELS.get, horizontal=True)
n = c4.slider("How many", 3, 15, 8)

try:
    row = find_suburb_row(df, suburb, bedrooms, ptype)
except LookupError as e:
    st.warning(str(e))
    footer()
    st.stop()

comps = find_comparables(df, row, n=n)
both = pd.concat([row.to_frame().T, comps])
both["Suburb"] = both["suburb"].str.title()
both["Type"] = ["Selected"] + ["Similar"] * len(comps)
both["median_weekly_rent"] = both["median_weekly_rent"].astype(float)

left, right = st.columns(2)
with left:
    chart = both.sort_values("median_weekly_rent")
    fig = px.bar(chart, x="median_weekly_rent", y="Suburb", orientation="h", color="Type",
                 color_discrete_map={"Selected": "#0066cc", "Similar": "#9bb7d4"},
                 labels={"median_weekly_rent": "Median weekly rent ($)", "Suburb": ""})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=40), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.scatter_mapbox(both, lat="Latitude", lon="Longitude", color="Type", hover_name="Suburb",
                            hover_data={"median_weekly_rent": ":$.0f", "Latitude": False, "Longitude": False, "Type": False},
                            color_discrete_map={"Selected": "#0066cc", "Similar": "#e67e22"}, zoom=8, height=420)
    fig.update_traces(marker=dict(size=13))
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(comparables_table(comps), hide_index=True, use_container_width=True)
footer()
