# Data

## What is committed, and what is not

| Directory | In Git? | Why |
|-----------|---------|-----|
| `data/raw/` | ❌ No | Source files from government agencies. Large, and we redistribute them from the original source rather than from this repo. The directory is kept by `.gitkeep`. |
| `data/processed/` | ✅ Yes | Small (~470 KB total) and it is the **only** record of the raw inputs for anyone who does not have them. Without it, a fresh clone cannot train a model. |

## Raw source files

`src/data/pipeline.py` expects exactly these three filenames in `data/raw/`:

| Filename | Source | Contents |
|----------|--------|----------|
| `Moving_Annual_Median_Rent_by_SuburbTown_-_Main_File.csv` | Victorian Rental Bond Board / DFFH rental report | Moving annual median weekly rent by suburb, bedroom count and property type |
| `OBA_Victoria_Suburb_Profiling_Clean.xlsx` (sheet `Clean_Data`) | ABS Census 2021, compiled | Population, median household income, Year 12 completion, unemployment, labour force participation, IRSAD/SEIFA score, latitude, longitude |
| `valuation_data.xlsx` (sheet `Sheet1`) | Victorian property valuations | Per-locality house and unit price change percentages (`house_change_perc_24-25`, `house_change_perc_15-25`, `unit_change_perc_24-25`) |

If a filename or sheet name does not match, the pipeline exits with an error at
Step 1. Ask a teammate for the files rather than renaming your own copies.

## Processed outputs

Produced by `python src/data/pipeline.py`, run from the repository root:

| File | Contents |
|------|----------|
| `merged_rental_data.csv` | 1,177 rows × 33 columns. One row per suburb × bedrooms × property type |
| `X_train.pkl` | Feature matrix, 1,177 × 26 |
| `y_train.pkl` | Target: `median_weekly_rent` (AUD/week) |
| `feature_names.txt` | The 26 feature column names, in order |
| `data_dictionary.txt` | Generated summary: structure, features, target, key fields |

## Refresh procedure

1. Put the three raw files in `data/raw/` with the exact names above.
2. Activate the venv and run from the repository root:
   ```bash
   python src/data/pipeline.py
   ```
3. Review the diff on `data/processed/`. Row counts and rent statistics are printed
   at the end of the run — sanity-check them before committing.
4. Commit `data/processed/` **and** say in the PR that the data was regenerated, so
   whoever is training a model knows to retrain.

## Current dataset shape

- **1,177 records**, **200 suburbs**, last 12 months of rental data aggregated
- Target `median_weekly_rent`: mean **$591/week**, range **$229 – $1,805**
- No nulls in the merged output

Available suburb × bedroom × property type combinations:

| Bedrooms | Flat | House |
|----------|------|-------|
| 1 | 194 | — |
| 2 | 199 | 197 |
| 3 | 193 | 198 |
| 4 | — | 196 |

⚠️ **Note for modelling and for the UI:** 1-bedroom exists only as a flat and
4-bedroom only as a house, because that is how the source report publishes them.
Property type is therefore confounded with bedroom count at both ends of the range —
the model cannot learn a house/flat difference at 1 or 4 bedrooms, and the UI should
not offer those combinations.

## Known issue in the generated features

`feature_names.txt` currently lists the raw categorical columns `affluence_bin`,
`property_type` and `growth_category` **alongside** their own one-hot dummies. They
are picked up by the `startswith()` prefix filters in `src/data/pipeline.py`
(Step 10). That means `X_train.pkl` contains three string columns, which XGBoost
will reject unless they are dropped first — the dummies already encode them.

This is not fixed here because it changes pipeline output. Drop them when loading,
or fix the prefix filters in a separate PR.
