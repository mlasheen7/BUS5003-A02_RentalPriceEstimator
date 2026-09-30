"""Turn one rent prediction into the factors that drove it.

The model is a tree ensemble, so SHAP's TreeExplainer gives each feature's
exact contribution in AUD/week: the contributions plus the model's baseline add
up to the prediction. One-hot columns are summed back into the feature they
encode (property_flat + property_house -> property type) so an explanation
talks about "property type" rather than two half-effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import shap


# One-hot prefix -> the single driver the dummies are folded into.
ONE_HOT_GROUPS = {
    "property_": "property_type",
    "affluence_": "area_affluence",
    "growth_": "price_growth",
}

FEATURE_LABELS = {
    "bedrooms": "number of bedrooms",
    "Median_Household_Income_Weekly_AUD": "median household income",
    "IRSAD_Score": "socio-economic advantage score (SEIFA)",
    "Labour_Force_Participation_Rate_Pct": "labour force participation rate",
    "Unemployment_Rate_Pct": "unemployment rate",
    "Year12_Completion_Rate_Pct": "share of residents who finished Year 12",
    "employment_rate": "employment rate",
    "distance_to_cbd_km": "distance to the Melbourne CBD",
    "income_x_seifa": "income and advantage score combined",
    "house_change_perc_24-25": "house price growth in 2024-25",
    "Population_2021": "suburb population",
    "rental_count": "number of rentals recorded in the suburb",
    "property_type": "property type",
    "area_affluence": "area affluence band",
    "price_growth": "house price growth band",
}

# Drivers that come from the valuation data. Suburbs missing from that file had
# their growth filled with 0 by the pipeline, so for them these are not real.
GROWTH_DRIVERS = frozenset({"house_change_perc_24-25", "price_growth"})

# The source rental report publishes 1-bedroom only as a flat and 4-bedroom
# only as a house, so these are the only combinations the model has seen.
SUPPORTED_COMBINATIONS = frozenset(
    {(1, "flat"), (2, "flat"), (2, "house"), (3, "flat"), (3, "house"), (4, "house")}
)


@dataclass(frozen=True)
class Driver:
    """One factor behind a prediction."""

    feature: str
    label: str
    value: str
    effect: float  # AUD/week; positive raises the estimate


@dataclass(frozen=True)
class PredictionExplanation:
    """A prediction, the model's baseline, and every driver, largest first."""

    prediction: float
    baseline: float
    drivers: List[Driver]

    def top_drivers(self, n: int = 5, include_growth: bool = True) -> List[Driver]:
        drivers = [
            d for d in self.drivers if include_growth or d.feature not in GROWTH_DRIVERS
        ]
        return drivers[:n]


def find_suburb_row(
    merged: pd.DataFrame, suburb: str, bedrooms: int, property_type: str
) -> pd.Series:
    """Return the processed-data row for one suburb/bedrooms/property type.

    Raises LookupError with a message fit to show a user when the combination
    does not exist in the data.
    """
    property_type = property_type.strip().lower()
    if property_type in ("unit", "flat/unit"):
        property_type = "flat"
    bedrooms = int(bedrooms)

    if (bedrooms, property_type) not in SUPPORTED_COMBINATIONS:
        raise LookupError(
            f"{bedrooms}-bedroom {property_type}s are not in the source rental data, "
            "so the model cannot estimate them."
        )

    match = merged[
        (merged["suburb"] == suburb.strip().lower())
        & (merged["bedrooms"] == bedrooms)
        & (merged["property_type"] == property_type)
    ]
    if match.empty:
        raise LookupError(
            f"No rental data for a {bedrooms}-bedroom {property_type} in {suburb.title()}."
        )
    return match.iloc[0]


def growth_is_known(suburb_row: Mapping[str, Any]) -> bool:
    """False when the pipeline filled a missing valuation with 0."""
    value = suburb_row.get("house_change_perc_24-25")
    return value is not None and not pd.isna(value) and float(value) != 0.0


class ShapExplainer:
    """Wraps a trained tree model; build once and reuse for every prediction."""

    def __init__(self, model: Any):
        if not hasattr(model, "feature_names_in_"):
            raise ValueError("Model does not record its feature names (feature_names_in_).")
        self.model = model
        self.feature_names = [str(name) for name in model.feature_names_in_]
        self._explainer = shap.TreeExplainer(model)

    def feature_row(self, suburb_row: Mapping[str, Any]) -> pd.DataFrame:
        """Build the one-row model input from a processed-data row."""
        missing = [name for name in self.feature_names if name not in suburb_row]
        if missing:
            raise ValueError("Row is missing model features: " + ", ".join(missing))
        values = {name: [float(suburb_row[name])] for name in self.feature_names}
        return pd.DataFrame(values, columns=self.feature_names)

    def explain(self, X_row: pd.DataFrame) -> PredictionExplanation:
        """Predict one row and break the prediction into drivers."""
        if len(X_row) != 1:
            raise ValueError(f"Expected exactly one row, got {len(X_row)}.")
        if list(X_row.columns) != self.feature_names:
            raise ValueError("Row columns do not match the model's feature order.")

        contributions = np.asarray(self._explainer.shap_values(X_row), dtype=float).reshape(-1)
        baseline = float(np.ravel(self._explainer.expected_value)[0])
        prediction = float(self.model.predict(X_row)[0])

        row = X_row.iloc[0]
        drivers = _group_drivers(dict(zip(self.feature_names, contributions)), row)
        drivers.sort(key=lambda d: abs(d.effect), reverse=True)
        return PredictionExplanation(prediction=prediction, baseline=baseline, drivers=drivers)


def _group_drivers(contributions: Dict[str, float], row: pd.Series) -> List[Driver]:
    grouped: Dict[str, float] = {}
    active_category: Dict[str, Optional[str]] = {}

    for feature, effect in contributions.items():
        group = _group_of(feature)
        if group is None:
            grouped[feature] = effect
            continue
        grouped[group] = grouped.get(group, 0.0) + effect
        active_category.setdefault(group, None)
        if float(row[feature]) == 1.0:
            prefix = next(p for p, g in ONE_HOT_GROUPS.items() if g == group)
            active_category[group] = feature[len(prefix):].replace("_", " ")

    drivers = []
    for feature, effect in grouped.items():
        if feature in active_category:
            value = active_category[feature] or "unknown"
        else:
            value = _format_value(feature, float(row[feature]))
        drivers.append(
            Driver(
                feature=feature,
                label=FEATURE_LABELS.get(feature, feature.replace("_", " ")),
                value=value,
                effect=float(effect),
            )
        )
    return drivers


def _group_of(feature: str) -> Optional[str]:
    for prefix, group in ONE_HOT_GROUPS.items():
        if feature.startswith(prefix):
            return group
    return None


def _format_value(feature: str, value: float) -> str:
    if feature == "bedrooms":
        return f"{value:.0f}"
    if feature == "distance_to_cbd_km":
        return f"{value:.1f} km"
    if feature == "Median_Household_Income_Weekly_AUD":
        return f"${value:,.0f} per week"
    if feature == "house_change_perc_24-25":
        return f"{value:+.1f}%"
    if feature.endswith("_Pct") or feature == "employment_rate":
        return f"{value:.1f}%"
    if feature in ("IRSAD_Score", "Population_2021", "rental_count"):
        return f"{value:,.0f}"
    return f"{value:,.2f}"
