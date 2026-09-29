"""Rental Price Estimator, entry point and page menu.

Run from the project folder:  streamlit run app/main.py
"""

import streamlit as st

pages = [
    st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_Predict.py", title="Predict", icon="📊"),
    st.Page("pages/2_Comparables.py", title="Comparables", icon="🏘️"),
    st.Page("pages/3_FAQ.py", title="FAQ", icon="❓"),
    st.Page("pages/4_About.py", title="About", icon="ℹ️"),
]
st.navigation(pages).run()
