"""Generate held-out SHAP explanations for the tuned XGBoost rental-price model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from .data_utils import RANDOM_STATE, get_project_paths, load_and_validate_training_data
except ImportError:  # Supports direct execution: python src/models/shap_analysis.py
    from data_utils import RANDOM_STATE, get_project_paths, load_and_validate_training_data  # type: ignore


EXPECTED_TEST_SIZE = 0.20


def load_persisted_test_positions(
    split_manifest_path: Path,
    total_rows: int,
) -> tuple[list[int], dict[str, Any]]:
    """Load and validate the baseline's persisted held-out row positions."""
    if not split_manifest_path.exists():
        raise FileNotFoundError(
            "Persisted split manifest is missing. Run baseline training before SHAP analysis: "
            f"{split_manifest_path}"
        )

    with split_manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_keys = {
        "random_state",
        "test_size",
        "total_rows",
        "train_rows",
        "test_rows",
        "train_row_positions",
        "test_row_positions",
    }
    missing_keys = sorted(required_keys.difference(manifest))
    if missing_keys:
        raise ValueError("Split manifest is missing required keys: " + ", ".join(missing_keys))
    if int(manifest["total_rows"]) != total_rows:
        raise ValueError("Split manifest total_rows does not match the validated feature data.")
    if int(manifest["random_state"]) != RANDOM_STATE:
        raise ValueError(
            f"Split manifest random_state must be {RANDOM_STATE}, "
            f"found {manifest['random_state']}."
        )
    if not np.isclose(float(manifest["test_size"]), EXPECTED_TEST_SIZE):
        raise ValueError(
            f"Split manifest test_size must be {EXPECTED_TEST_SIZE}, "
            f"found {manifest['test_size']}."
        )

    train_positions = [int(position) for position in manifest["train_row_positions"]]
    test_positions = [int(position) for position in manifest["test_row_positions"]]
    train_set = set(train_positions)
    test_set = set(test_positions)
    expected_positions = set(range(total_rows))

    if len(train_positions) != int(manifest["train_rows"]):
        raise ValueError("Split manifest train_rows does not match train_row_positions.")
    if len(test_positions) != int(manifest["test_rows"]):
        raise ValueError("Split manifest test_rows does not match test_row_positions.")
    if len(train_set) != len(train_positions) or len(test_set) != len(test_positions):
        raise ValueError("Split manifest contains duplicate row positions.")
    if train_set.intersection(test_set):
        raise ValueError("Split manifest has overlapping training and held-out positions.")
    if train_set.union(test_set) != expected_positions:
        raise ValueError("Split manifest does not cover every feature-data row exactly once.")

    return test_positions, manifest


def validate_shap_matrix(
    shap_values: Any,
    expected_rows: int,
    expected_features: int,
) -> np.ndarray:
    """Validate and normalize TreeExplainer output to a 2D SHAP matrix."""
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError(
                "Expected one SHAP-value matrix for regression, "
                f"but received {len(shap_values)} matrices."
            )
        shap_values = shap_values[0]

    matrix = np.asarray(shap_values)
    expected_shape = (expected_rows, expected_features)
    if matrix.ndim != 2 or matrix.shape != expected_shape:
        raise ValueError(
            "SHAP matrix dimensions do not match held-out data. "
            f"Expected {expected_shape}, received {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("SHAP matrix contains non-finite values.")
    return matrix


def main() -> None:
    """Generate SHAP plots and mean absolute SHAP importance on held-out observations."""
    paths = get_project_paths()
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    model_path = paths["models_dir"] / "xgboost_tuned.pkl"

    print("=" * 80)
    print("RENTAL PRICE ESTIMATOR - TUNED MODEL SHAP ANALYSIS")
    print("=" * 80)

    print("\n[1/5] Loading validated data and persisted held-out test positions...")
    data = load_and_validate_training_data()
    test_positions, split_manifest = load_persisted_test_positions(
        paths["split_manifest_path"],
        total_rows=len(data.X),
    )
    X_test = data.X.iloc[test_positions].copy()
    if len(X_test) != int(split_manifest["test_rows"]):
        raise ValueError("Held-out SHAP feature matrix row count does not match the split manifest.")
    print(f"  Held-out observations for interpretation: {len(X_test)}")
    print(f"  Training feature count: {len(data.selected_feature_names)}")

    print("\n[2/5] Loading tuned XGBoost model without retraining...")
    if not model_path.exists():
        raise FileNotFoundError(
            "Tuned model is missing. Run hyperparameter tuning before SHAP analysis: "
            f"{model_path}"
        )
    with model_path.open("rb") as file:
        model = pickle.load(file)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Tuned model does not expose feature_names_in_ for feature-order validation.")
    if list(model.feature_names_in_) != data.selected_feature_names:
        raise ValueError("Tuned model feature order does not match the validated 23 selected features.")

    print("\n[3/5] Computing TreeExplainer SHAP values on held-out observations...")
    explainer = shap.TreeExplainer(model)
    shap_matrix = validate_shap_matrix(
        explainer.shap_values(X_test),
        expected_rows=len(X_test),
        expected_features=len(data.selected_feature_names),
    )
    print(f"  Verified SHAP matrix dimensions: {shap_matrix.shape}")

    print("\n[4/5] Calculating mean absolute SHAP importance...")
    shap_importance = pd.DataFrame(
        {
            "feature": data.selected_feature_names,
            "mean_absolute_shap_value": np.mean(np.abs(shap_matrix), axis=0),
        }
    ).sort_values("mean_absolute_shap_value", ascending=False, ignore_index=True)
    csv_path = paths["reports_dir"] / "shap_feature_importance.csv"
    shap_importance.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.relative_to(paths['root'])}")

    print("\n[5/5] Generating SHAP bar-summary and beeswarm plots...")
    bar_plot_path = paths["reports_dir"] / "shap_feature_importance.png"
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_matrix, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary_plot_path = paths["reports_dir"] / "shap_summary.png"
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_matrix, X_test, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {bar_plot_path.relative_to(paths['root'])}")
    print(f"  Saved: {summary_plot_path.relative_to(paths['root'])}")

    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print("SHAP values were calculated on the persisted held-out test partition only.")


if __name__ == "__main__":
    main()