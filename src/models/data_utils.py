"""Shared data loading, validation, and split utilities for model training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
RAW_CATEGORICAL_COLUMNS = (
    "affluence_bin",
    "property_type",
    "growth_category",
)


@dataclass(frozen=True)
class TrainingData:
    """Validated, model-ready data and its selected feature metadata."""

    X: pd.DataFrame
    y: pd.Series
    source_feature_names: list[str]
    selected_feature_names: list[str]
    excluded_feature_names: list[str]


def get_project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[2]


def get_project_paths() -> dict[str, Path]:
    """Return all paths used by the baseline model-training stage."""
    root = get_project_root()
    processed_dir = root / "data" / "processed"

    return {
        "root": root,
        "processed_dir": processed_dir,
        "X_path": processed_dir / "X_train.pkl",
        "y_path": processed_dir / "y_train.pkl",
        "feature_names_path": processed_dir / "feature_names.txt",
        "split_manifest_path": processed_dir / "train_test_split_v1.0.json",
        "models_dir": root / "models",
        "reports_dir": root / "reports",
    }


def _read_feature_names(feature_names_path: Path) -> list[str]:
    """Read the ordered feature manifest, rejecting blank or duplicate entries."""
    feature_names = [
        line.strip()
        for line in feature_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not feature_names:
        raise ValueError("The feature-name manifest is empty.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("The feature-name manifest contains duplicate feature names.")

    return feature_names


def load_and_validate_training_data() -> TrainingData:
    """Load processed training data and return a validated numeric feature matrix.

    The processed feature matrix deliberately retains human-readable categorical
    columns alongside one-hot encoded equivalents. The raw categorical columns
    are excluded because their encoded versions are already available for the
    baseline XGBoost model.
    """
    paths = get_project_paths()
    required_paths = (paths["X_path"], paths["y_path"], paths["feature_names_path"])
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required processed training artifacts are missing: " + ", ".join(missing_paths)
        )

    X_raw = pd.read_pickle(paths["X_path"])
    y_raw = pd.read_pickle(paths["y_path"])
    source_feature_names = _read_feature_names(paths["feature_names_path"])

    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError("X_train.pkl must contain a pandas DataFrame.")
    if not isinstance(y_raw, pd.Series):
        raise TypeError("y_train.pkl must contain a pandas Series.")
    if X_raw.empty or y_raw.empty:
        raise ValueError("Training features and target must both contain at least one row.")
    if len(X_raw) != len(y_raw):
        raise ValueError(
            f"Feature and target row counts differ: {len(X_raw)} features, {len(y_raw)} targets."
        )
    if X_raw.columns.has_duplicates:
        raise ValueError("The feature matrix contains duplicate column names.")
    if list(X_raw.columns) != source_feature_names:
        raise ValueError(
            "Feature-name manifest does not exactly match X_train.pkl column order."
        )
    if X_raw.isnull().any().any():
        null_count = int(X_raw.isnull().sum().sum())
        raise ValueError(f"Feature matrix contains {null_count} missing values.")
    if y_raw.isnull().any():
        raise ValueError(f"Target vector contains {int(y_raw.isnull().sum())} missing values.")

    missing_categoricals = [
        column for column in RAW_CATEGORICAL_COLUMNS if column not in X_raw.columns
    ]
    if missing_categoricals:
        raise ValueError(
            "Expected raw categorical columns are missing: " + ", ".join(missing_categoricals)
        )

    selected_feature_names = [
        column for column in source_feature_names if column not in RAW_CATEGORICAL_COLUMNS
    ]
    X = X_raw.loc[:, selected_feature_names].copy()

    non_numeric_columns = [
        column
        for column in X.columns
        if not (pd.api.types.is_numeric_dtype(X[column]) or pd.api.types.is_bool_dtype(X[column]))
    ]
    if non_numeric_columns:
        raise TypeError(
            "Selected model features must be numeric or boolean. Invalid columns: "
            + ", ".join(non_numeric_columns)
        )

    X = X.astype(float)
    y = pd.to_numeric(y_raw, errors="raise").astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values.")
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Target vector contains non-finite values.")

    return TrainingData(
        X=X,
        y=y,
        source_feature_names=source_feature_names,
        selected_feature_names=selected_feature_names,
        excluded_feature_names=list(RAW_CATEGORICAL_COLUMNS),
    )


def create_reproducible_split(
    data: TrainingData,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Create an 80/20 split and return its data plus a serializable manifest."""
    row_positions = np.arange(len(data.X))
    train_positions, test_positions = train_test_split(
        row_positions,
        test_size=test_size,
        random_state=random_state,
    )

    X_train = data.X.iloc[train_positions].copy()
    X_test = data.X.iloc[test_positions].copy()
    y_train = data.y.iloc[train_positions].copy()
    y_test = data.y.iloc[test_positions].copy()

    source_index = data.X.index
    manifest = {
        "split_name": "baseline_xgboost_v1.0",
        "created_by": "src/models/train.py",
        "random_state": random_state,
        "test_size": test_size,
        "total_rows": int(len(data.X)),
        "train_rows": int(len(train_positions)),
        "test_rows": int(len(test_positions)),
        "train_row_positions": [int(position) for position in train_positions],
        "test_row_positions": [int(position) for position in test_positions],
        "train_source_index_labels": [_to_json_value(source_index[position]) for position in train_positions],
        "test_source_index_labels": [_to_json_value(source_index[position]) for position in test_positions],
    }

    return X_train, X_test, y_train, y_test, manifest


def persist_split_manifest(manifest: dict[str, Any]) -> Path:
    """Write the train/test split manifest for later model-evaluation stages."""
    split_manifest_path = get_project_paths()["split_manifest_path"]
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with split_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return split_manifest_path


def _to_json_value(value: Any) -> Any:
    """Convert pandas and NumPy scalar values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value