"""Tune XGBoost hyperparameters without using the held-out test partition."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

try:
    from .data_utils import RANDOM_STATE, get_project_paths, load_and_validate_training_data
except ImportError:  # Supports direct execution: python src/models/tune_hyperparameters.py
    from data_utils import RANDOM_STATE, get_project_paths, load_and_validate_training_data  # type: ignore


N_TRIALS = 50
CV_FOLDS = 5
BASELINE_TEST_RMSE = 49.176523914534826


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate JSON-serializable regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_persisted_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Load and validate the baseline split manifest without creating a new split."""
    if not split_manifest_path.exists():
        raise FileNotFoundError(
            "Persisted split manifest is missing. Run the baseline training stage first: "
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
    if int(manifest["total_rows"]) != len(X) or len(X) != len(y):
        raise ValueError(
            "Split manifest row count does not match the currently validated training data."
        )
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
    expected_positions = set(range(len(X)))
    train_set = set(train_positions)
    test_set = set(test_positions)

    if len(train_positions) != int(manifest["train_rows"]):
        raise ValueError("Split manifest training row count does not match train_row_positions.")
    if len(test_positions) != int(manifest["test_rows"]):
        raise ValueError("Split manifest test row count does not match test_row_positions.")
    if len(train_set) != len(train_positions) or len(test_set) != len(test_positions):
        raise ValueError("Split manifest contains duplicate row positions.")
    if train_set.intersection(test_set):
        raise ValueError("Split manifest has overlapping training and held-out test positions.")
    if train_set.union(test_set) != expected_positions:
        raise ValueError("Split manifest positions do not cover the full training dataset exactly once.")

    return (
        X.iloc[train_positions].copy(),
        X.iloc[test_positions].copy(),
        y.iloc[train_positions].copy(),
        y.iloc[test_positions].copy(),
        manifest,
    )


def suggest_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Return guide-specified search parameters for one Optuna trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_jobs": 1,
    }


def load_baseline_test_rmse(baseline_metadata_path: Path) -> float:
    """Read the baseline test RMSE used for the final tuned-model comparison."""
    if not baseline_metadata_path.exists():
        return BASELINE_TEST_RMSE

    with baseline_metadata_path.open("r", encoding="utf-8") as file:
        baseline_metadata = json.load(file)
    return float(baseline_metadata["metrics"]["test"]["rmse"])


def main() -> None:
    """Run 50-trial Optuna tuning, then evaluate the selected model once on test data."""
    paths = get_project_paths()
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RENTAL PRICE ESTIMATOR - HYPERPARAMETER TUNING")
    print("=" * 80)

    print("\n[1/7] Loading validated data and persisted baseline split...")
    data = load_and_validate_training_data()
    X_train, X_test, y_train, y_test, split_manifest = load_persisted_split(
        data.X,
        data.y,
        paths["split_manifest_path"],
    )
    print(f"  Training rows for tuning: {len(X_train)}")
    print(f"  Held-out test rows reserved for final evaluation: {len(X_test)}")
    print(f"  Selected feature count: {len(data.selected_feature_names)}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        """Minimize mean 5-fold CV RMSE on the training partition only."""
        model = xgb.XGBRegressor(**suggest_hyperparameters(trial))
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        cv_rmse_scores = np.sqrt(-cv_scores)
        return float(cv_rmse_scores.mean())

    print(f"\n[2/7] Running Optuna optimization ({N_TRIALS} trials)...")
    print("  Objective: mean 5-fold CV RMSE on persisted training rows only.")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n[3/7] Collecting all Optuna trial results...")
    trials = study.trials_dataframe(
        attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "state")
    )
    trials = trials.rename(columns={"value": "mean_cv_rmse", "state": "trial_state"})
    trials["is_best_trial"] = trials["number"].eq(study.best_trial.number)
    trials_path = paths["reports_dir"] / "tuning_results.csv"
    trials.to_csv(trials_path, index=False)
    print(f"  Saved all trial results: {trials_path.relative_to(paths['root'])}")

    print("\n[4/7] Training final tuned model on all persisted training rows...")
    best_parameters = suggest_hyperparameters(study.best_trial)
    tuned_model = xgb.XGBRegressor(**best_parameters)
    tuned_model.fit(X_train, y_train)

    print("\n[5/7] Evaluating tuned model once on the held-out test partition...")
    train_metrics = calculate_metrics(y_train, tuned_model.predict(X_train))
    test_predictions = tuned_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)
    baseline_test_rmse = load_baseline_test_rmse(paths["models_dir"] / "xgboost_v1.0.json")
    rmse_improvement = baseline_test_rmse - test_metrics["rmse"]
    print(f"  Best mean CV RMSE: ${study.best_trial.value:.2f}/week")
    print(f"  Tuned test RMSE: ${test_metrics['rmse']:.2f}/week")
    print(f"  Baseline test RMSE: ${baseline_test_rmse:.2f}/week")
    print(f"  Test RMSE improvement versus baseline: ${rmse_improvement:.2f}/week")

    print("\n[6/7] Saving feature importance and tuned model...")
    feature_importance = pd.DataFrame(
        {
            "feature": data.selected_feature_names,
            "importance": tuned_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    feature_importance_path = paths["reports_dir"] / "feature_importance_tuned.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    model_path = paths["models_dir"] / "xgboost_tuned.pkl"
    with model_path.open("wb") as file:
        pickle.dump(tuned_model, file)

    with model_path.open("rb") as file:
        reloaded_model = pickle.load(file)
    reloaded_predictions = reloaded_model.predict(X_test)
    serialization_verified = bool(
        reloaded_predictions.shape == test_predictions.shape
        and np.all(np.isfinite(reloaded_predictions))
        and np.allclose(reloaded_predictions, test_predictions)
    )
    if not serialization_verified:
        raise RuntimeError("Reloaded tuned model did not reproduce valid held-out predictions.")
    print(f"  Saved and verified: {model_path.relative_to(paths['root'])}")
    print(f"  Saved: {feature_importance_path.relative_to(paths['root'])}")

    print("\n[7/7] Saving tuned-model metadata...")
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_type": "XGBoost Regressor (Tuned)",
        "model_version": "xgboost_tuned",
        "trained_at_utc": generated_at,
        "optimization": {
            "library": "Optuna",
            "n_trials": N_TRIALS,
            "direction": "minimize",
            "objective": "mean 5-fold cross-validation RMSE on persisted training rows only",
            "sampler": "TPESampler",
            "sampler_seed": RANDOM_STATE,
            "best_trial_number": int(study.best_trial.number),
            "best_cv_rmse": float(study.best_trial.value),
            "best_parameters": best_parameters,
            "all_trial_results": str(trials_path.relative_to(paths["root"])),
        },
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
            "test_size": float(split_manifest["test_size"]),
            "random_state": int(split_manifest["random_state"]),
            "split_manifest": str(paths["split_manifest_path"].relative_to(paths["root"])),
            "training_row_positions": split_manifest["train_row_positions"],
            "test_row_positions": split_manifest["test_row_positions"],
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "baseline_test_rmse": baseline_test_rmse,
            "test_rmse_improvement_vs_baseline": rmse_improvement,
        },
        "serialization_verified": serialization_verified,
        "library_versions": {
            "optuna": optuna.__version__,
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    metadata_path = paths["models_dir"] / "xgboost_tuned.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"  Saved: {metadata_path.relative_to(paths['root'])}")

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()