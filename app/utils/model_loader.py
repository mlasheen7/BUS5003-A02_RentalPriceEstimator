"""Load the trained rent model for the app.

Trained .pkl files are gitignored (models/* in .gitignore), so they are not in
the repository and not on Streamlit Cloud. This loader therefore:

1. uses models/xgboost_tuned.pkl if it exists, else models/xgboost_v1.0.pkl;
2. otherwise rebuilds the tuned model from what IS committed: the processed
   features (X_train.pkl, y_train.pkl), the saved train/test split and the best
   Optuna parameters in models/xgboost_tuned.json. This takes about a second.
   Results can differ very slightly from the original run across machines, so
   the rebuilt model is re-scored on the same held-out rows and the app shows
   that score.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_CATEGORICAL_COLUMNS = ("affluence_bin", "property_type", "growth_category")


@dataclass
class LoadedModel:
    model: Any
    version: str
    source: str  # "saved file" or "rebuilt from saved parameters"
    metrics: Dict[str, Any]

    @property
    def test_mae(self) -> Optional[float]:
        return self.metrics.get("test", {}).get("mae")

    @property
    def test_r2(self) -> Optional[float]:
        return self.metrics.get("test", {}).get("r2")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _rebuild_tuned_model(meta: Dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    X_raw = pd.read_pickle(PROCESSED_DIR / "X_train.pkl")
    y = pd.read_pickle(PROCESSED_DIR / "y_train.pkl").astype(float)
    features = [c for c in X_raw.columns if c not in RAW_CATEGORICAL_COLUMNS]
    X = X_raw.loc[:, features].astype(float)

    split = _read_json(PROCESSED_DIR / "train_test_split_v1.0.json")
    train_positions = [int(p) for p in split["train_row_positions"]]

    params = meta["optimization"]["best_parameters"]
    model = XGBRegressor(**params)
    model.fit(X.iloc[train_positions], y.iloc[train_positions])

    # Score the rebuilt model on the same held-out rows, so the app reports
    # the accuracy of the model it is actually using.
    test_positions = [int(p) for p in split["test_row_positions"]]
    y_test = y.iloc[test_positions].to_numpy()
    errors = model.predict(X.iloc[test_positions]) - y_test
    ss_res = float((errors ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    metrics = {"test": {
        "mae": float(abs(errors).mean()),
        "rmse": float((errors ** 2).mean() ** 0.5),
        "r2": 1 - ss_res / ss_tot,
    }}
    return model, metrics


def load_model() -> LoadedModel:
    for version in ("xgboost_tuned", "xgboost_v1.0"):
        pkl = MODELS_DIR / f"{version}.pkl"
        if pkl.exists():
            with pkl.open("rb") as f:
                model = pickle.load(f)
            meta = _read_json(MODELS_DIR / f"{version}.json")
            return LoadedModel(model, version, "saved file", meta.get("metrics", {}))

    meta = _read_json(MODELS_DIR / "xgboost_tuned.json")
    if not meta:
        raise FileNotFoundError(
            "No trained model found. Add models/xgboost_tuned.pkl or models/xgboost_tuned.json."
        )
    model, metrics = _rebuild_tuned_model(meta)
    return LoadedModel(model, "xgboost_tuned", "rebuilt from saved parameters", metrics)
