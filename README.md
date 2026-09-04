# Rental Price Estimator 🏠

AI-powered rental price predictor for Victoria, Australia. Brings transparency to the rental market using public government data.

## Quick Start

### Prerequisites
- Python 3.9+
- Git

### Setup (15 min)
\`\`\`bash
# 1. Clone
git clone https://github.com/[your-username]/rental-price-estimator.git
cd rental-price-estimator

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your API keys

# 5. Test
pytest tests/test_setup.py -v
\`\`\`

## Project Structure
\`\`\`
rental-price-estimator/
├── data/
│   ├── raw/              # Downloaded data (not in Git)
│   └── processed/        # Cleaned, merged data
├── notebooks/            # Jupyter notebooks (EDA, analysis)
├── src/                  # Source code (data, models, utils)
├── tests/                # Unit tests
├── models/               # Trained models (not in Git)
├── reports/              # Analysis reports, figures
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── .env.example          # Template for secrets
├── pytest.ini            # Test configuration
└── README.md             # This file
\`\`\`

## Development Workflow

### Daily
1. **Activate venv:** \`source venv/bin/activate\`
2. **Pull latest:** \`git pull origin main\`
3. **Write code/tests** in feature branch
4. **Run tests:** \`pytest tests/\`
5. **Commit:** \`git add . && git commit -m "Your message"\`
6. **Push:** \`git push origin [branch-name]\`

### Committing Data
- **DO:** Commit \`.gitignore\`, \`data/README.md\`, \`requirements.txt\`
- **DON'T:** Commit large CSV files, models, .env, or secrets

## Testing
\`\`\`bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_setup.py::test_pandas_installed

# With coverage
pytest tests/ --cov=src
\`\`\`

## Data

See \`data/README.md\` for:
- Data sources (ABS Census, Valuations, Rental Bonds)
- Data refresh procedure
- Field descriptions

## Contributing

1. Create a feature branch: \`git checkout -b feature/your-feature\`
2. Make changes & write tests
3. Push & create Pull Request
4. Get review from team member
5. Merge to main

## Team
- Data Engineer: [Name]
- ML Engineer: [Name]
- ... etc

## Timeline
- Sprint 1 (Weeks 1–3): Data foundation
- Sprint 2 (Weeks 4–6): Model training
- Sprint 3 (Weeks 7–9): App development & deployment

## License
MIT

## Questions?
Post in Slack #rental-estimator or contact team lead.