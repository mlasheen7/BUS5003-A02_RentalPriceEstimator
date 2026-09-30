"""About page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, so `app.utils` imports work
import streamlit as st

from app.utils.app_state import footer, setup_page

setup_page("About")
st.title("ℹ️ About this project")
st.markdown("""
**Rental Price Estimator** helps renters understand fair market rent across Victorian suburbs.

#### Data sources
- **Victorian Rental Bond Board / DFFH Rental Report**: moving annual median weekly rent by suburb, bedrooms and property type
- **ABS Census 2021**: population, household income, Year 12 completion, unemployment, labour force participation, SEIFA (IRSAD) scores
- **Victorian property valuations**: house and unit price change by locality

#### How it is built
- **Model:** XGBoost regression, tuned with Optuna, checked with a fairness audit across affluence groups
- **Explainability:** SHAP factor breakdown plus a plain English summary generated through OpenRouter
- **App:** Streamlit, deployed on Streamlit Community Cloud

#### Privacy
Only public, suburb level data is used. No personal information is collected.

#### Team
Group 4, BUS5003 Operationalising Business Analytics, La Trobe University, 2026
""")
footer()
