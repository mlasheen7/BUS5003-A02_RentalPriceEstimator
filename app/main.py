"""Rental Price Estimator, entry point and page menu.

Run from the project folder:  streamlit run app/main.py
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Rental Price Estimator",
    page_icon="🏠",
    layout="wide"
)



st.title("🏠 Rental Price Estimator")
st.write("Select a page from the sidebar to get started.")