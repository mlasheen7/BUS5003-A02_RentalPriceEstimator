"""Rental Price Estimator, entry point and page menu.

Run from the project folder:  streamlit run app/main.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import streamlit as st

# Helper modules live in app/pages/, so point the app.utils names at them
sys.modules["app.utils.model_loader"] = importlib.import_module("app.pages.model_loader")
sys.modules["app.utils.app_state"] = importlib.import_module("app.pages.app_state")

pages = [
    st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_Predict.py", title="Predict", icon="📊"),
    st.Page("pages/2_Comparables.py", title="Comparables", icon="🏘️"),
    st.Page("pages/3_FAQ.py", title="FAQ", icon="❓"),
    st.Page("pages/4_About.py", title="About", icon="ℹ️"),
]
st.navigation(pages).run()
