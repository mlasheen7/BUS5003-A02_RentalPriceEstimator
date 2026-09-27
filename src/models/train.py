"""Train, validate, and serialize the baseline XGBoost rental-price model."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

try:
    from .data_utils import (
        RANDOM_STATE,
        create_reproducible_split,
        get_project_paths,
        load_and_validate_training_data,
        persist_split_manifest,
    )
except ImportError:  # Supports direct execution: python src/models/train.py
    from data_utils import (  # type: ignore
        RANDOM_STATE,
        create_reproducible_split,
        get_project_paths,
        load_and_validate_training_data,
        persist_split_manifest,
    )


BASELINE_HYPERPARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "n_jobs": 1,
}


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics with JSON-serializable float values."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_performance_report(
    generated_at: str,
    train_rows: int,
    test_rows: int,
    selected_feature_names: list[str],
    excluded_feature_names: list[str],
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    cv_rmse: np.ndarray,
    feature_importance: pd.DataFrame,
    model_reload_verified: bool,
    split_manifest_path: Path,
) -> str:
    """Build the Markdown baseline performance report."""
    cv_mean = float(cv_rmse.mean())
    cv_std = float(cv_rmse.std())
    initial_target = 70.0
    final_target = 50.0
    cv_gap_percent = abs(cv_mean - test_metrics["rmse"]) / test_metrics["rmse"] * 100
    train_test_gap = test_metrics["rmse"] - train_metrics["rmse"]

    report_lines = [
        "# Rental Price Estimator — Baseline Model Performance",
        "",
        f"Generated (UTC): {generated_at}",
        "",
        "## Data and Split",
        "",
        f"- Training rows: {train_rows}",
        f"- Held-out test rows: {test_rows}",
        "- Train/test split: 80/20",
        f"- Random state: {RANDOM_STATE}",
        "- Cross-validation: 5-fold KFold on the training partition only",
        f"- Final selected model features: {len(selected_feature_names)}",
        f"- Excluded raw categorical features: {', '.join(excluded_feature_names)}",
        f"- Persisted split manifest: `{split_manifest_path.relative_to(get_project_paths()['root']).as_posix()}`",
        "",
        "## Baseline XGBoost Configuration",
        "",
        "```json",
        json.dumps(BASELINE_HYPERPARAMETERS, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| Partition | RMSE (AUD/week) | MAE (AUD/week) | R² |",
        "|---|---:|---:|---:|",
        (
            f"| Train | ${train_metrics['rmse']:.2f} | ${train_metrics['mae']:.2f} "
            f"| {train_metrics['r2']:.4f} |"
        ),
        (
            f"| Held-out test | ${test_metrics['rmse']:.2f} | ${test_metrics['mae']:.2f} "
            f"| {test_metrics['r2']:.4f} |"
        ),
        (
            f"| 5-fold CV (training only) | ${cv_mean:.2f} ± ${cv_std:.2f} "
            "| — |"
        ),
        "",
        "## Validation Checks",
        "",
        f"- Initial guide target (test RMSE < $70/week): {'PASS' if test_metrics['rmse'] < initial_target else 'NOT YET ACHIEVED'}.",
        f"- Final guide target (test RMSE ≤ $50/week): {'PASS' if test_metrics['rmse'] <= final_target else 'NOT YET ACHIEVED'}.",
        f"- CV versus held-out test RMSE difference: {cv_gap_percent:.1f}% (guide threshold: within 10%).",
        f"- Train-to-test RMSE gap: ${train_test_gap:.2f}/week.",
        f"- Serialized model reload and held-out prediction check: {'PASS' if model_reload_verified else 'FAILED'}.",
        "",
        "## Final Selected Features",
        "",
    ]
    report_lines.extend(f"- `{feature}`" for feature in selected_feature_names)
    report_lines.extend(["", "## Top 10 Built-In Feature Importances", ""])
    report_lines.extend(
        f"{rank}. `{row.feature}` — {row.importance:.6f}"
        for rank, row in enumerate(feature_importance.head(10).itertuples(index=False), start=1)
    )
    report_lines.extend(
        [
            "",
            "## Scope",
            "",
            "This report covers the baseline training stage only. Hyperparameter tuning, fairness auditing, and SHAP analysis have not been run.",
            "",
        ]
    )

    return "\n".join(report_lines)


def main() -> None:
    """Run the complete baseline training workflow."""
    paths = get_project_paths()
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RENTAL PRICE ESTIMATOR - BASELINE MODEL TRAINING")
    print("=" * 80)

    print("\n[1/8] Loading and validating processed training data...")
    data = load_and_validate_training_data()
    print(f"  X shape: {data.X.shape}")
    print(f"  y shape: {data.y.shape}")
    print(f"  Selected numeric/encoded features: {len(data.selected_feature_names)}")
    print(f"  Excluded raw categorical features: {', '.join(data.excluded_feature_names)}")

    print("\n[2/8] Creating reproducible 80/20 train-test split...")
    X_train, X_test, y_train, y_test, split_manifest = create_reproducible_split(data)
    split_manifest_path = persist_split_manifest(split_manifest)
    print(f"  Train rows: {len(X_train)}")
    print(f"  Test rows: {len(X_test)}")
    print(f"  Saved split manifest: {split_manifest_path.relative_to(paths['root'])}")

    print("\n[3/8] Training baseline XGBoost regressor...")
    baseline_model = xgb.XGBRegressor(**BASELINE_HYPERPARAMETERS)
    baseline_model.fit(X_train, y_train)

    print("\n[4/8] Calculating training and held-out test metrics...")
    train_metrics = calculate_metrics(y_train, baseline_model.predict(X_train))
    test_predictions = baseline_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)
    print(
        f"  Train - RMSE: ${train_metrics['rmse']:.2f}, "
        f"MAE: ${train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.4f}"
    )
    print(
        f"  Test  - RMSE: ${test_metrics['rmse']:.2f}, "
        f"MAE: ${test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.4f}"
    )

    print("\n[5/8] Running 5-fold cross-validation on the training partition...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        xgb.XGBRegressor(**BASELINE_HYPERPARAMETERS),
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"  CV RMSE: ${cv_rmse.mean():.2f} ± ${cv_rmse.std():.2f}/week")

    print("\n[6/8] Saving built-in feature importance...")
    feature_importance = pd.DataFrame(
        {
            "feature": data.selected_feature_names,
            "importance": baseline_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    feature_importance_path = paths["reports_dir"] / "feature_importance_baseline.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"  Saved: {feature_importance_path.relative_to(paths['root'])}")

    print("\n[7/8] Saving and reloading the baseline model...")
    model_path = paths["models_dir"] / "xgboost_v1.0.pkl"
    with model_path.open("wb") as file:
        pickle.dump(baseline_model, file)

    with model_path.open("rb") as file:
        reloaded_model = pickle.load(file)
    reloaded_predictions = reloaded_model.predict(X_test)
    model_reload_verified = bool(
        reloaded_predictions.shape == test_predictions.shape
        and np.all(np.isfinite(reloaded_predictions))
        and np.allclose(reloaded_predictions, test_predictions)
    )
    if not model_reload_verified:
        raise RuntimeError("Reloaded model did not reproduce valid held-out predictions.")
    print(f"  Saved and verified: {model_path.relative_to(paths['root'])}")

    print("\n[8/8] Saving metadata and performance report...")
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_type": "XGBoost Regressor (Baseline)",
        "model_version": "xgboost_v1.0",
        "trained_at_utc": generated_at,
        "data": {
            "total_rows": int(len(data.X)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "source_feature_count": int(len(data.source_feature_names)),
            "selected_feature_count": int(len(data.selected_feature_names)),
            "selected_features": data.selected_feature_names,
            "excluded_raw_categorical_features": data.excluded_feature_names,
        },
        "split": {
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "split_manifest": str(split_manifest_path.relative_to(paths["root"])),
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "cross_validation": {
                "folds": 5,
                "rmse_scores": [float(score) for score in cv_rmse],
                "rmse_mean": float(cv_rmse.mean()),
                "rmse_std": float(cv_rmse.std()),
            },
        },
        "hyperparameters": BASELINE_HYPERPARAMETERS,
        "serialization_verified": model_reload_verified,
        "library_versions": {
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    metadata_path = paths["models_dir"] / "xgboost_v1.0.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    performance_report = build_performance_report(
        generated_at=generated_at,
        train_rows=len(X_train),
        test_rows=len(X_test),
        selected_feature_names=data.selected_feature_names,
        excluded_feature_names=data.excluded_feature_names,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_rmse=cv_rmse,
        feature_importance=feature_importance,
        model_reload_verified=model_reload_verified,
        split_manifest_path=split_manifest_path,
    )
    performance_report_path = paths["reports_dir"] / "model_performance.md"
    performance_report_path.write_text(performance_report, encoding="utf-8")
    print(f"  Saved: {metadata_path.relative_to(paths['root'])}")
    print(f"  Saved: {performance_report_path.relative_to(paths['root'])}")

    print("\n" + "=" * 80)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Held-out test RMSE: ${test_metrics['rmse']:.2f}/week")
    print("No hyperparameter tuning, fairness audit, or SHAP analysis was run.")


if __name__ == "__main__":
    main()