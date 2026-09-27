# Rental Price Estimator — Fairness Audit

Generated (UTC): 2026-09-22T01:01:25.051063+00:00

## Audit Scope

- Model: `xgboost_tuned`
- Held-out rows evaluated: 236
- Training rows were not used for fairness evaluation.
- Grouping field: existing `affluence_bin` from `merged_rental_data.csv`.
- Affluence groups: very_low, low, medium, high, very_high.
- Persisted split manifest: `data/processed/train_test_split_v1.0.json`

## Overall Held-out Performance

- RMSE: $43.76/week

## Group Metrics

| Affluence group | Sample count | RMSE (AUD/week) | MAE (AUD/week) | Mean residual / bias (AUD/week) | Median absolute error (AUD/week) |
|---|---:|---:|---:|---:|---:|
| very_low | 57 | $29.15 | $22.01 | $-1.71 | $19.26 |
| low | 51 | $26.88 | $20.99 | $1.21 | $17.71 |
| medium | 42 | $42.07 | $33.28 | $8.35 | $24.10 |
| high | 44 | $42.79 | $33.33 | $-1.57 | $28.47 |
| very_high | 42 | $70.99 | $48.95 | $-27.77 | $26.60 |

## Fairness Variance Check

Formula: `std(group RMSE) / mean(group RMSE) * 100`
- Fairness variance: 37.08%
- Guide target: < 15%
- Result: FAIL

## Residual Definition

Residual / bias is calculated as `actual_rent - predicted_rent`. A positive mean indicates under-prediction on average; a negative mean indicates over-prediction on average.
