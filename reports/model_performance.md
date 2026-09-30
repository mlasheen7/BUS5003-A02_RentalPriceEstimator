# Rental Price Estimator — Baseline Model Performance

Generated (UTC): 2026-09-30T11:27:59.441901+00:00

## Data and Split

- Training rows: 941
- Held-out test rows: 236
- Train/test split: 80/20
- Random state: 42
- Cross-validation: 5-fold KFold on the training partition only
- Final selected model features: 23
- Excluded raw categorical features: affluence_bin, property_type, growth_category
- Persisted split manifest: `data/processed/train_test_split_v1.0.json`

## Baseline XGBoost Configuration

```json
{
  "n_estimators": 100,
  "learning_rate": 0.1,
  "max_depth": 5,
  "min_child_weight": 1,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "objective": "reg:squarederror",
  "random_state": 42,
  "verbosity": 0,
  "n_jobs": 1
}
```

## Metrics

| Partition | RMSE (AUD/week) | MAE (AUD/week) | R² |
|---|---:|---:|---:|
| Train | $18.08 | $13.53 | 0.9931 |
| Held-out test | $49.18 | $31.60 | 0.9436 |
| 5-fold CV (training only) | $58.76 ± $7.29 | — |

## Validation Checks

- Initial guide target (test RMSE < $70/week): PASS.
- Final guide target (test RMSE ≤ $50/week): PASS.
- CV versus held-out test RMSE difference: 19.5% (guide threshold: within 10%).
- Train-to-test RMSE gap: $31.10/week.
- Serialized model reload and held-out prediction check: PASS.

## Final Selected Features

- `bedrooms`
- `Median_Household_Income_Weekly_AUD`
- `IRSAD_Score`
- `Labour_Force_Participation_Rate_Pct`
- `Unemployment_Rate_Pct`
- `Year12_Completion_Rate_Pct`
- `employment_rate`
- `distance_to_cbd_km`
- `income_x_seifa`
- `house_change_perc_24-25`
- `Population_2021`
- `rental_count`
- `affluence_very_low`
- `affluence_low`
- `affluence_medium`
- `affluence_high`
- `affluence_very_high`
- `property_flat`
- `property_house`
- `growth_declining`
- `growth_slow_growth`
- `growth_moderate_growth`
- `growth_strong_growth`

## Top 10 Built-In Feature Importances

1. `Year12_Completion_Rate_Pct` — 0.290931
2. `bedrooms` — 0.229338
3. `property_house` — 0.113048
4. `distance_to_cbd_km` — 0.068939
5. `property_flat` — 0.060090
6. `IRSAD_Score` — 0.049451
7. `income_x_seifa` — 0.028572
8. `affluence_high` — 0.022031
9. `rental_count` — 0.018663
10. `house_change_perc_24-25` — 0.014832

## Scope

This report covers the baseline training stage only. Hyperparameter tuning, fairness auditing, and SHAP analysis have not been run.
