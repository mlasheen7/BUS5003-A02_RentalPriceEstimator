# Data

## Layout

| Path | Contents | In Git? |
|---|---|---|
| `data/raw/` | Source files downloaded from the providers below | **No** — empty except `.gitkeep` |
| `data/processed/` | Cleaned, merged, model-ready outputs | **Yes** |

### Why `data/processed/` is committed

It is under 500 KB in total, and it **cannot be regenerated from a fresh clone**
because `data/raw/` is not in the repo. Committing it means the ML engineer and the
frontend lead can start work without first sourcing the raw files.

If these outputs ever grow past a few MB, move them to shared storage and drop them
from Git.

---

## Raw files you need

`src/data/pipeline.py` expects these three files in `data/raw/`. The filenames must
match exactly:

| Filename | Source | Notes |
|---|---|---|
| `Moving_Annual_Median_Rent_by_SuburbTown_-_Main_File.csv` | Vic DFFH — Rental Report, moving annual median rent by suburb | Quarterly release |
| `OBA_Victoria_Suburb_Profiling_Clean.xlsx` | ABS Census 2021 suburb profiles, cleaned | Sheet name must be `Clean_Data` |
| `valuation_data.xlsx` | Vic Valuer-General property values by locality | Sheet name must be `Sheet1` |

Ask the Data Engineer for the current copies if you cannot source them.

---

## Regenerating the processed data

```bash
source venv/bin/activate
python src/data/pipeline.py
```

Run it from the **repository root** — the script uses paths relative to the
working directory.

Outputs written to `data/processed/`:

| File | Contents |
|---|---|
| `merged_rental_data.csv` | Full merged dataset, one row per suburb × bedrooms × property type |
| `X_train.pkl` | Feature matrix (26 features) |
| `y_train.pkl` | Target — `median_weekly_rent` (AUD/week) |
| `feature_names.txt` | Ordered feature list |
| `data_dictionary.txt` | Field definitions and record counts |

---

## Current dataset

- **1,177 records** — 200 suburbs × bedrooms {1,2,3,4} × property type {flat, house}
- **Target:** `median_weekly_rent`, AUD per week
- **Aggregation:** mean of the last 12 months of available data
- Hyphenated suburb groups (e.g. `Albert Park-Middle Park`) are split into one row
  per suburb, and `unit` is normalised to `flat`.

See `data/processed/data_dictionary.txt` for the full field list.

---

## Refresh procedure

1. Download the updated source files into `data/raw/` using the exact filenames above.
2. Re-run the pipeline.
3. Check the printed record counts and rent statistics look sane.
4. Commit **only** `data/processed/` — never `data/raw/`.
5. Open a PR (see `CONTRIBUTING.md`); the refresh must be reviewed like any change.
