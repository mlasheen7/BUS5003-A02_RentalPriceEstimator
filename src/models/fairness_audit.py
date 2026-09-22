"""Audit tuned-model held-out prediction errors across existing affluence groups."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from .data_utils import RANDOM_STATE, get_project_paths, load_and_validate_training_data
except ImportError:  # Supports direct execution: python src/models/fairness_audit.py
    from data_utils import RANDOM_STATE, get_project_paths, load_and_validate_training_data  # type: ignore


AFFLUENCE_GROUPS = ("very_low", "low", "medium", "high", "very_high")
FAIRNESS_VARIANCE_TARGET_PERCENT = 15.0


def load_and_validate_split_positions(
    split_manifest_path: Path,
    total_rows: int,
) -> tuple[list[int], dict[str, Any]]:
    """Load persisted positions and ensure they define a complete, disjoint split."""
    if not split_manifest_path.exists():
        raise FileNotFoundError(
            "Persisted split manifest is missing. Run baseline training before the fairness audit: "
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
        raise ValueError("Split manifest total_rows does not match the processed training data.")
    if int(manifest["random_state"]) != RANDOM_STATE:
        raise ValueError(
            f"Split manifest random_state must be {RANDOM_STATE}, "
            f"found {manifest['random_state']}."
        )
    if not np.isclose(float(manifest["test_size"]), 0.20):
        raise ValueError(
            f"Split manifest test_size must be 0.20, found {manifest['test_size']}."
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
        raise ValueError("Split manifest has overlapping training and held-out test positions.")
    if train_set.union(test_set) != expected_positions:
        raise ValueError("Split manifest does not cover every processed-data row exactly once.")

    return test_positions, manifest


def load_and_validate_aligned_audit_data(
    selected_features: pd.DataFrame,
    target: pd.Series,
    selected_feature_names: list[str],
    merged_data_path: Path,
) -> pd.DataFrame:
    """Verify merged rows safely match processed features and target by position.

    Positional alignment is accepted only when the merged dataset has the same
    number of rows, matching target values, all selected model columns, and
    matching numeric/boolean feature values in the same row order.
    """
    if not merged_data_path.exists():
        raise FileNotFoundError(f"Merged audit dataset is missing: {merged_data_path}")

    merged = pd.read_csv(merged_data_path)
    required_columns = {"median_weekly_rent", "affluence_bin", *selected_feature_names}
    missing_columns = sorted(required_columns.difference(merged.columns))
    if missing_columns:
        raise ValueError(
            "Merged audit dataset is missing required columns: " + ", ".join(missing_columns)
        )
    if len(merged) != len(selected_features) or len(target) != len(selected_features):
        raise ValueError(
            "Cannot safely align rows: merged data, feature matrix, and target vector have different row counts."
        )
    if merged["affluence_bin"].isna().any():
        raise ValueError("Merged audit dataset contains missing affluence_bin values.")

    observed_groups = set(merged["affluence_bin"].astype(str))
    expected_groups = set(AFFLUENCE_GROUPS)
    if observed_groups != expected_groups:
        raise ValueError(
            "Merged audit dataset affluence_bin groups do not match the required groups. "
            f"Expected {sorted(expected_groups)}, found {sorted(observed_groups)}."
        )

    merged_target = pd.to_numeric(merged["median_weekly_rent"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.allclose(target.to_numpy(dtype=float), merged_target, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Cannot safely align rows: target values in merged_rental_data.csv do not match y_train.pkl by position."
        )

    for feature in selected_feature_names:
        merged_feature = pd.to_numeric(merged[feature], errors="raise").to_numpy(dtype=float)
        processed_feature = selected_features[feature].to_numpy(dtype=float)
        if not np.allclose(processed_feature, merged_feature, rtol=0.0, atol=1e-9):
            raise ValueError(
                "Cannot safely align rows: selected feature values differ by position for "
                f"'{feature}'."
            )

    return merged


def build_fairness_report(
    generated_at: str,
    model_name: str,
    split_manifest_path: Path,
    held_out_rows: int,
    overall_rmse: float,
    group_metrics: pd.DataFrame,
    fairness_variance_percent: float,
    fairness_passed: bool,
) -> str:
    """Create a readable Markdown record of the held-out fairness audit."""
    lines = [
        "# Rental Price Estimator — Fairness Audit",
        "",
        f"Generated (UTC): {generated_at}",
        "",
        "## Audit Scope",
        "",
        f"- Model: `{model_name}`",
        f"- Held-out rows evaluated: {held_out_rows}",
        "- Training rows were not used for fairness evaluation.",
        "- Grouping field: existing `affluence_bin` from `merged_rental_data.csv`.",
        "- Affluence groups: very_low, low, medium, high, very_high.",
        f"- Persisted split manifest: `{split_manifest_path.as_posix()}`",
        "",
        "## Overall Held-out Performance",
        "",
        f"- RMSE: ${overall_rmse:.2f}/week",
        "",
        "## Group Metrics",
        "",
        "| Affluence group | Sample count | RMSE (AUD/week) | MAE (AUD/week) | Mean residual / bias (AUD/week) | Median absolute error (AUD/week) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_metrics.itertuples(index=False):
        lines.append(
            f"| {row.affluence_bin} | {row.sample_count} | ${row.rmse:.2f} | ${row.mae:.2f} "
            f"| ${row.mean_residual_bias:.2f} | ${row.median_absolute_error:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Fairness Variance Check",
            "",
            "Formula: `std(group RMSE) / mean(group RMSE) * 100`",
            f"- Fairness variance: {fairness_variance_percent:.2f}%",
            f"- Guide target: < {FAIRNESS_VARIANCE_TARGET_PERCENT:.0f}%",
            f"- Result: {'PASS' if fairness_passed else 'FAIL'}",
            "",
            "## Residual Definition",
            "",
            "Residual / bias is calculated as `actual_rent - predicted_rent`. A positive mean indicates under-prediction on average; a negative mean indicates over-prediction on average.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the tuned-model fairness audit on the persisted held-out test rows."""
    paths = get_project_paths()
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    model_path = paths["models_dir"] / "xgboost_tuned.pkl"
    merged_data_path = paths["processed_dir"] / "merged_rental_data.csv"

    print("=" * 80)
    print("RENTAL PRICE ESTIMATOR - TUNED MODEL FAIRNESS AUDIT")
    print("=" * 80)

    print("\n[1/5] Loading and validating processed data...")
    data = load_and_validate_training_data()
    merged = load_and_validate_aligned_audit_data(
        data.X,
        data.y,
        data.selected_feature_names,
        merged_data_path,
    )
    print("  Positional alignment with merged audit data verified.")

    print("\n[2/5] Loading persisted held-out test positions...")
    test_positions, split_manifest = load_and_validate_split_positions(
        paths["split_manifest_path"],
        total_rows=len(data.X),
    )
    X_test = data.X.iloc[test_positions].copy()
    y_test = data.y.iloc[test_positions].copy()
    held_out_audit_data = merged.iloc[test_positions].copy()
    print(f"  Held-out rows: {len(test_positions)}")

    print("\n[3/5] Loading tuned model and generating held-out predictions...")
    if not model_path.exists():
        raise FileNotFoundError(
            "Tuned model is missing. Run hyperparameter tuning before the fairness audit: "
            f"{model_path}"
        )
    with model_path.open("rb") as file:
        model = pickle.load(file)
    if list(model.feature_names_in_) != data.selected_feature_names:
        raise ValueError("Tuned model feature order does not match the validated selected feature list.")

    predictions = model.predict(X_test)
    if not np.isfinite(predictions).all():
        raise ValueError("Tuned model returned non-finite held-out predictions.")
    residuals = y_test.to_numpy(dtype=float) - predictions
    absolute_errors = np.abs(residuals)
    overall_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    print("\n[4/5] Calculating held-out metrics by existing affluence_bin group...")
    audit_frame = pd.DataFrame(
        {
            "affluence_bin": held_out_audit_data["affluence_bin"].astype(str).to_numpy(),
            "actual_rent": y_test.to_numpy(dtype=float),
            "predicted_rent": predictions,
            "residual": residuals,
            "absolute_error": absolute_errors,
        }
    )
    group_rows: list[dict[str, float | int | str]] = []
    for group in AFFLUENCE_GROUPS:
        group_data = audit_frame.loc[audit_frame["affluence_bin"] == group]
        if group_data.empty:
            raise ValueError(f"Held-out test split contains no rows for required group '{group}'.")
        group_rows.append(
            {
                "affluence_bin": group,
                "sample_count": int(len(group_data)),
                "rmse": float(
                    np.sqrt(mean_squared_error(group_data["actual_rent"], group_data["predicted_rent"]))
                ),
                "mae": float(mean_absolute_error(group_data["actual_rent"], group_data["predicted_rent"])),
                "mean_residual_bias": float(group_data["residual"].mean()),
                "median_absolute_error": float(group_data["absolute_error"].median()),
            }
        )

    group_metrics = pd.DataFrame(group_rows)
    fairness_variance_percent = float(
        group_metrics["rmse"].std(ddof=0) / group_metrics["rmse"].mean() * 100
    )
    fairness_passed = fairness_variance_percent < FAIRNESS_VARIANCE_TARGET_PERCENT
    print(f"  Fairness variance: {fairness_variance_percent:.2f}%")
    print(f"  Target: < {FAIRNESS_VARIANCE_TARGET_PERCENT:.0f}%")
    print(f"  Result: {'PASS' if fairness_passed else 'FAIL'}")

    print("\n[5/5] Saving fairness-audit outputs...")
    generated_at = datetime.now(timezone.utc).isoformat()
    group_metrics.insert(0, "model_name", "xgboost_tuned")
    group_metrics.insert(1, "split_name", str(split_manifest.get("split_name", "baseline_xgboost_v1.0")))
    group_metrics.insert(2, "held_out_rows", len(test_positions))
    group_metrics["overall_held_out_rmse"] = overall_rmse
    group_metrics["fairness_variance_percent"] = fairness_variance_percent
    group_metrics["fairness_target_percent"] = FAIRNESS_VARIANCE_TARGET_PERCENT
    group_metrics["fairness_result"] = "PASS" if fairness_passed else "FAIL"
    group_metrics["generated_at_utc"] = generated_at

    csv_path = paths["reports_dir"] / "fairness_audit.csv"
    group_metrics.to_csv(csv_path, index=False)
    report_path = paths["reports_dir"] / "fairness_audit.md"
    report_path.write_text(
        build_fairness_report(
            generated_at=generated_at,
            model_name="xgboost_tuned",
            split_manifest_path=paths["split_manifest_path"].relative_to(paths["root"]),
            held_out_rows=len(test_positions),
            overall_rmse=overall_rmse,
            group_metrics=group_metrics,
            fairness_variance_percent=fairness_variance_percent,
            fairness_passed=fairness_passed,
        ),
        encoding="utf-8",
    )
    print(f"  Saved: {csv_path.relative_to(paths['root'])}")
    print(f"  Saved: {report_path.relative_to(paths['root'])}")

    print("\n" + "=" * 80)
    print("FAIRNESS AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()