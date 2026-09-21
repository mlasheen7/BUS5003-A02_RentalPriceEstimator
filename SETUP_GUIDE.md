# Development Environment Setup Guide

Complete these steps once. Time: ~15 minutes.

## Prerequisites
- Git installed ([git-scm.com](https://git-scm.com))
- Python 3.9+ installed ([python.org](https://python.org))
- GitHub account with access to this repo (ask @mlasheen7)

## Step 1: Clone Repository (2 min)

```bash
cd ~/Documents   # or your preferred directory
git clone https://github.com/mlasheen7/BUS5003-A02_RentalPriceEstimator.git
cd BUS5003-A02_RentalPriceEstimator
```

## Step 2: Create Virtual Environment (3 min)

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# OR venv\Scripts\activate      # Windows
```

Verify: you should see `(venv)` in your terminal prompt.

## Step 3: Install Dependencies (5 min)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes — `pycaret` and `xgboost` are large.

## Step 4: Configure Environment (2 min)

```bash
cp .env.example .env
```

Then open `.env` and paste in the real values. Ask the Data Engineer for the API keys.

`.env` is gitignored. **Never commit it.**

## Step 5: Verify Installation (3 min)

```bash
pytest
```

Expected: all 4 tests pass. ✅

## Step 6: Set Your Git Identity (1 min)

Set your name and email **for this repo only**, so commits are attributed correctly
even if your machine's global git config belongs to a different account:

```bash
git config --local user.name "Your Name"
git config --local user.email "your@email.com"
```

Check it took effect:

```bash
git config --local --get user.email
```

## Troubleshooting

**"python3: command not found"**
- Install Python from [python.org](https://python.org)
- On Mac, try: `brew install python3`

**"ModuleNotFoundError: No module named 'pandas'"**
- Make sure venv is activated (look for the `(venv)` prompt)
- Run: `pip install -r requirements.txt`

**"Permission denied" on `venv\Scripts\activate`**
- On Windows PowerShell, run:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**"Repository not found" when pushing**
- You are authenticated as the wrong GitHub account, or you do not have access yet.
- Check with `gh auth status`, and ask @mlasheen7 to add you as a collaborator.

**`pip: command not found` inside the venv**
- The venv was created without pip. Recreate it with `python3 -m venv venv`,
  or if you use `uv`, install with `uv pip install -r requirements.txt`.

**Pipeline fails with "Moving_Annual_Median_Rent... not found"**
- `data/raw/` ships empty by design. See [data/README.md](data/README.md) for the
  files you need and where to get them.

**Still stuck?**
- Post in Slack #rental-estimator
- Check [README.md](README.md) for additional help

## Next Steps

- Read [CONTRIBUTING.md](CONTRIBUTING.md) — we branch, open a PR, and a **reviewer**
  merges it. Nobody pushes to `main`.
- Data Engineer: download the raw files listed in [data/README.md](data/README.md)
- ML Engineer: start EDA in `notebooks/`, see
  [MODEL_TRAINING_TUNING_GUIDE.md](MODEL_TRAINING_TUNING_GUIDE.md)
- Frontend Lead: see [UI_INTERFACE_SETUP_GUIDE.md](UI_INTERFACE_SETUP_GUIDE.md)
- API Engineer: see [CLAUDE_API_SETUP_GUIDE.md](CLAUDE_API_SETUP_GUIDE.md)
