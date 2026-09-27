# Rental Price Estimator 🏠

AI-powered rental price predictor for Victoria, Australia. Brings transparency to the rental market using public government data.

## Quick Start

### Prerequisites
- **Python 3.9 – 3.11.** Not 3.12+ — `pycaret` pins `numpy<1.27` / `pandas<2.2`, and those versions have no 3.12 wheels.
- Git

### Setup (15 min)
```bash
# 1. Clone
git clone https://github.com/mlasheen7/BUS5003-A02_RentalPriceEstimator.git
cd BUS5003-A02_RentalPriceEstimator

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\Activate.ps1   # Windows PowerShell

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and add your own API keys

# 5. Test
pytest
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for a step-by-step walkthrough and troubleshooting.

## Project Structure
```
BUS5003-A02_RentalPriceEstimator/
├── data/
│   ├── raw/              # Source files (not in Git — see data/README.md)
│   └── processed/        # Cleaned, merged data (committed)
├── src/                  # Source code
│   ├── data/             # Ingestion, cleaning, merging  → pipeline.py lives here
│   ├── features/         # Feature engineering
│   ├── models/           # Training, evaluation, prediction
│   └── utils/            # Config, logging
├── app/                  # Streamlit application
│   ├── pages/            # Predict / comparables / FAQ / about
│   ├── utils/            # Model loader, Claude API client
│   └── assets/           # Images
├── notebooks/            # Jupyter notebooks (EDA, analysis)
├── tests/                # Unit tests
├── models/               # Trained models (not in Git)
├── reports/              # Analysis reports, figures
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── .env.example          # Template for secrets
├── pytest.ini            # Test configuration
└── README.md             # This file
```

## Running the Pipeline

The data pipeline reads from `data/raw/` and writes to `data/processed/`. Run it from the repository root:

```bash
python src/data/pipeline.py
```

You need the three raw source files first — `data/README.md` lists them and where they come from. The committed contents of `data/processed/` are the output of the last run, so you can start modelling without them.

## AI Explanations

Each prediction comes with a short plain-English explanation of what drove it.
`app/utils/shap_explainer.py` breaks the prediction into its SHAP drivers
(AUD/week per factor), and `app/utils/api_client.py` turns them into 2–3 sentences
through [OpenRouter](https://openrouter.ai). If the key is missing or the API
fails, a template explanation built from the same drivers is shown instead, so the
app never breaks.

Setup:

1. Add `OPENROUTER_API_KEY` to your `.env` (see `.env.example`). Set a spending
   limit on the key in the OpenRouter dashboard.
2. Optionally change `EXPLAIN_MODEL` (default `anthropic/claude-haiku-4.5`) to any
   OpenRouter model ID. No code change is needed.
3. Build the model once, since `models/*.pkl` is not in Git:
   ```bash
   python src/models/train.py
   python src/models/tune_hyperparameters.py
   ```
4. Try it: `python -m app.utils.api_client footscray 2 flat`

Using it from the app:

```python
explainer = ShapExplainer(model)          # app.utils.shap_explainer
client = ExplanationClient()              # app.utils.api_client
row = find_suburb_row(merged, suburb, bedrooms, property_type)  # LookupError if not in the data
result = explainer.explain(explainer.feature_row(row))
response = client.explain(row, result)    # response["explanation"], response["used_fallback"], ...
# Explanations take about 2 seconds; wrap the call in st.spinner("Explaining...").
```

Build `explainer` and `client` once (e.g. with `st.cache_resource`). The tests mock
the API, so `pytest` needs no key and makes no paid calls.

## Development Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full branch → PR → review → merge process.

The short version:

1. **Activate venv:** `source venv/bin/activate`
2. **Pull latest:** `git pull origin main`
3. **Branch:** `git checkout -b feature/your-feature`
4. **Write code and tests**
5. **Run tests:** `pytest`
6. **Commit and push:** `git push -u origin feature/your-feature`
7. **Open a PR** and request a review from a teammate

Nobody pushes to `main` directly, and nobody merges their own PR.

### Committing Data
- **DO** commit: `.gitignore`, `data/README.md`, `requirements.txt`, `data/processed/`
- **DON'T** commit: raw source files, trained models, `.env`, or any secret

## Testing
```bash
# Run all tests
pytest

# Run a single file
pytest tests/test_setup.py -v

# Run a single test
pytest tests/test_setup.py::test_pandas_installed
```

Coverage is reported for `src/` and `app/` automatically via `pytest.ini`.

## Data

See [data/README.md](data/README.md) for:
- Data sources (ABS Census, Valuations, Rental Bonds)
- The exact raw filenames the pipeline expects
- Data refresh procedure
- Field descriptions

## Team

Four collaborators. Roles as described in the implementation guides:

| Role | Guide |
|------|-------|
| Data Engineer | `Complete Implementation Guide_Data Pipeline Structure and Code` |
| ML Engineer | `MODEL_TRAINING_TUNING_GUIDE.md` |
| API Engineer | `CLAUDE_API_SETUP_GUIDE.md` |
| Frontend Lead | `UI_INTERFACE_SETUP_GUIDE.md` |

## Timeline
- Sprint 1 (Weeks 1–3): Data foundation
- Sprint 2 (Weeks 4–6): Model training
- Sprint 3 (Weeks 7–9): App development & deployment

## License
MIT

## Questions?
Post in the team channel, or raise an issue on this repo.
