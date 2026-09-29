"""Everything the pages share, loaded once and cached by Streamlit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.utils.api_client import DEFAULT_MODEL, ExplanationClient
from app.utils.model_loader import load_model
from app.utils.shap_explainer import ShapExplainer

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "merged_rental_data.csv"

PROPERTY_LABELS = {"flat": "Flat / Unit", "house": "House"}


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading the rent model...")
def get_model():
    return load_model()


@st.cache_resource
def get_explainer() -> ShapExplainer:
    return ShapExplainer(get_model().model)


def setting(name: str) -> Optional[str]:
    """Read a setting from the environment or .env, then Streamlit secrets.

    Only touches st.secrets when a secrets.toml exists, so running without one
    (e.g. locally before the key is added) shows no "No secrets found" error box.
    """
    load_dotenv()
    value = os.getenv(name)
    if value:
        return value
    try:
        if st.secrets.load_if_toml_exists():
            return st.secrets.get(name)
    except Exception:
        pass
    return None


@st.cache_resource
def get_client() -> ExplanationClient:
    # Pass the settings in, so the client never has to look them up itself.
    return ExplanationClient(
        api_key=setting("OPENROUTER_API_KEY") or "",
        model=setting("EXPLAIN_MODEL") or DEFAULT_MODEL,
    )


def ai_is_configured() -> bool:
    return bool(setting("OPENROUTER_API_KEY"))


def find_comparables(df: pd.DataFrame, row: pd.Series, n: int = 5) -> pd.DataFrame:
    """Same bedrooms and property type, closest on SEIFA score and household income."""
    pool = df[
        (df["bedrooms"] == row["bedrooms"])
        & (df["property_type"] == row["property_type"])
        & (df["suburb"] != row["suburb"])
    ].copy()
    seifa_sd = df["IRSAD_Score"].std() or 1
    income_sd = df["Median_Household_Income_Weekly_AUD"].std() or 1
    pool["gap"] = (
        ((pool["IRSAD_Score"] - row["IRSAD_Score"]) / seifa_sd) ** 2
        + ((pool["Median_Household_Income_Weekly_AUD"] - row["Median_Household_Income_Weekly_AUD"]) / income_sd) ** 2
    ) ** 0.5
    return pool.nsmallest(n, "gap")


def comparables_table(comps: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Suburb": comps["suburb"].str.title(),
        "Region": comps["region"],
        "Median rent ($/wk)": comps["median_weekly_rent"].round(0).astype(int),
        "SEIFA score": comps["IRSAD_Score"].round(0).astype(int),
        "Household income ($/wk)": comps["Median_Household_Income_Weekly_AUD"].round(0).astype(int),
        "Distance to CBD (km)": comps["distance_to_cbd_km"].round(1),
    }).reset_index(drop=True)


def setup_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} | Rental Price Estimator", page_icon="🏠", layout="wide")
    st.markdown(
        "<style>.block-container{padding-top:2rem;max-width:1150px}"
        "[data-testid='stMetricValue']{font-size:1.9rem}</style>",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.divider()
    st.caption("Rental Price Estimator · BUS5003 Operationalising Business Analytics · La Trobe University · Group 4 · "
               "A university prototype, not financial or leasing advice.")
