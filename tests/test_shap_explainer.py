"""Tests for app/utils/shap_explainer.py.

A small XGBoost model is trained on the committed processed data, so these run
in CI without models/*.pkl (which is not in Git).
"""
import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from app.utils.shap_explainer import (
    GROWTH_DRIVERS,
    ShapExplainer,
    find_suburb_row,
    growth_is_known,
)
from src.models.data_utils import load_and_validate_training_data

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def merged():
    return pd.read_csv(os.path.join(REPO_ROOT, "data", "processed", "merged_rental_data.csv"))


@pytest.fixture(scope="module")
def explainer():
    data = load_and_validate_training_data()
    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=1)
    model.fit(data.X, data.y)
    return ShapExplainer(model)


@pytest.fixture(scope="module")
def footscray(merged, explainer):
    row = find_suburb_row(merged, "Footscray", 2, "flat")
    return row, explainer.explain(explainer.feature_row(row))


def test_drivers_add_up_to_the_prediction(footscray):
    _, result = footscray
    total = result.baseline + sum(d.effect for d in result.drivers)
    assert total == pytest.approx(result.prediction, abs=0.05)


def test_prediction_matches_the_model(footscray, explainer):
    row, result = footscray
    expected = float(explainer.model.predict(explainer.feature_row(row))[0])
    assert result.prediction == pytest.approx(expected)


def test_one_hot_columns_are_folded_into_one_driver(footscray):
    _, result = footscray
    features = {d.feature for d in result.drivers}
    assert {"property_type", "area_affluence", "price_growth"} <= features
    assert not features & {"property_flat", "property_house", "affluence_high", "growth_declining"}
    assert len(features) == len(result.drivers)


def test_grouped_driver_reports_the_active_category(footscray):
    _, result = footscray
    property_driver = next(d for d in result.drivers if d.feature == "property_type")
    assert property_driver.value == "flat"


def test_drivers_are_sorted_by_size(footscray):
    _, result = footscray
    sizes = [abs(d.effect) for d in result.drivers]
    assert sizes == sorted(sizes, reverse=True)


def test_every_driver_has_a_readable_label(footscray):
    _, result = footscray
    for driver in result.drivers:
        assert "_" not in driver.label, driver.label


def test_top_drivers_can_leave_out_growth(footscray):
    _, result = footscray
    top = result.top_drivers(len(result.drivers), include_growth=False)
    assert not any(d.feature in GROWTH_DRIVERS for d in top)
    assert len(result.top_drivers(3)) == 3


def test_wrong_feature_order_is_rejected(footscray, explainer):
    row, _ = footscray
    X = explainer.feature_row(row)
    with pytest.raises(ValueError, match="feature order"):
        explainer.explain(X[list(reversed(X.columns))])


def test_more_than_one_row_is_rejected(footscray, explainer):
    row, _ = footscray
    X = explainer.feature_row(row)
    with pytest.raises(ValueError, match="one row"):
        explainer.explain(pd.concat([X, X]))


def test_model_without_feature_names_is_rejected():
    model = xgb.XGBRegressor(n_estimators=2)
    model.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="feature names"):
        ShapExplainer(model)


def test_find_suburb_row_is_case_insensitive_and_accepts_unit(merged):
    row = find_suburb_row(merged, "  FOOTSCRAY ", 2, "Unit")
    assert row["suburb"] == "footscray"
    assert row["property_type"] == "flat"


@pytest.mark.parametrize("bedrooms, property_type", [(1, "house"), (4, "flat"), (5, "house")])
def test_combinations_missing_from_the_source_are_explained(merged, bedrooms, property_type):
    with pytest.raises(LookupError, match="not in the source rental data"):
        find_suburb_row(merged, "footscray", bedrooms, property_type)


def test_unknown_suburb_is_reported(merged):
    with pytest.raises(LookupError, match="No rental data"):
        find_suburb_row(merged, "atlantis", 2, "flat")


def test_growth_filled_with_zero_counts_as_unknown(merged):
    # Richmond has no valuation data; the pipeline filled its growth with 0.
    assert not growth_is_known(find_suburb_row(merged, "richmond", 2, "flat"))
    assert growth_is_known(find_suburb_row(merged, "footscray", 2, "flat"))
