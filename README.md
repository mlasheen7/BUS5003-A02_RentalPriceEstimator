# Rental Price Estimator 🏠

AI-powered rental price predictor for Victoria, Australia. Brings transparency to the
rental market using public government data.

Given a suburb, bedroom count and property type, the app predicts a fair weekly rent,
explains the prediction in plain English via the Claude API, and shows comparable
suburbs.

## Quick Start

### Prerequisites
- Python 3.9+
- Git

### Setup (15 min)

```bash
# 1. Clone
git clone https://github.com/mlasheen7/BUS5003-A02_RentalPriceEstimator.git
cd BUS5003-A02_RentalPriceEstimator

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your API keys

# 5. Test
pytest
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for a step-by-step walkthrough and troubleshooting.

## Project Structure

```
BUS5003-A02_RentalPriceEstimator/
├── data/
│   ├── raw/                  # Source files (NOT in Git)
│   ├── processed/            # Cleaned, merged data (IS in Git - see data/README.md)
│   └── README.md             # Data sources & refresh procedure
├── src/                      # Reusable source code
│   ├── data/pipeline.py      # Merge + feature engineering pipeline
│   ├── features/             # Feature engineering
│   ├── models/               # Training, prediction, evaluation
│   └── utils/                # Config, logging
├── app/                      # Streamlit application
│   ├── pages/                # Predict, comparables, FAQ, about
│   └── utils/                # Model loader, Claude API client
├── notebooks/                # Jupyter notebooks (EDA, analysis)
├── tests/                    # Unit tests
├── models/                   # Trained models (NOT in Git)
├── reports/                  # Analysis reports, figures
├── docs/                     # Documentation
├── .github/workflows/        # CI (runs pytest on every PR)
├── requirements.txt          # Python dependencies
├── .env.example              # Template for secrets
├── pytest.ini                # Test configuration
├── CONTRIBUTING.md           # Branch → PR → review → merge workflow
└── README.md                 # This file
```

## Development Workflow

**We never commit to `main`.** Every change goes through a pull request that a
teammate reviews and merges. The full rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

### Daily

```bash
source venv/bin/activate
git checkout main && git pull origin main
git checkout -b feature/your-thing
# ... write code and tests ...
pytest
git add <files> && git commit -m "Describe the change"
git push -u origin feature/your-thing
gh pr create --base main --fill --reviewer <teammate>
```

Then a reviewer approves and merges. The author does not merge their own PR.

## Testing

```bash
# Run everything (coverage is on by default via pytest.ini)
pytest

# One test
pytest tests/test_setup.py::test_pandas_installed

# Skip coverage
pytest -p no:cacheprovider --no-cov
```

CI runs `pytest` on Python 3.9 and 3.11 for every pull request.

## Data

See [data/README.md](data/README.md) for data sources, the exact raw filenames the
pipeline expects, field descriptions and the refresh procedure.

Regenerate the processed dataset (requires raw files in `data/raw/`):

```bash
python src/data/pipeline.py
```

## Guides

| Guide | Audience |
|---|---|
| [Complete Implementation Guide](Complete%20Implementation%20Guide_Data%20Pipeline%20Structure%20and%20Code) | Everyone — pipeline flow and structure |
| [MODEL_TRAINING_TUNING_GUIDE.md](MODEL_TRAINING_TUNING_GUIDE.md) | ML Engineer |
| [CLAUDE_API_SETUP_GUIDE.md](CLAUDE_API_SETUP_GUIDE.md) | API Engineer |
| [UI_INTERFACE_SETUP_GUIDE.md](UI_INTERFACE_SETUP_GUIDE.md) | Frontend Lead |

## Team

| Role | GitHub |
|---|---|
| Repo owner / reviewer | [@mlasheen7](https://github.com/mlasheen7) |
| Contributor | [@pratiksjbrana01](https://github.com/pratiksjbrana01) |
| Contributor | [@raihanhossaingalib1-a11y](https://github.com/raihanhossaingalib1-a11y) |
| Contributor | [@thisissiddharthbisht](https://github.com/thisissiddharthbisht) |

## Timeline

- Sprint 1 (Weeks 1–3): Data foundation
- Sprint 2 (Weeks 4–6): Model training
- Sprint 3 (Weeks 7–9): App development & deployment

## License

MIT

## Questions?

Post in Slack #rental-estimator or contact the team lead.
