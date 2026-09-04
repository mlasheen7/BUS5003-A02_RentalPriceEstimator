# Development Environment Setup Guide

Complete these steps once. Time: ~15 minutes.

## Prerequisites
- Git installed ([git-scm.com](https://git-scm.com))
- Python 3.9+ installed ([python.org](https://python.org))
- GitHub account (you should have access to this repo)

## Step 1: Clone Repository (2 min)
\`\`\`bash
cd ~/Documents  # or your preferred directory
git clone https://github.com/[your-username]/rental-price-estimator.git
cd rental-price-estimator
\`\`\`

## Step 2: Create Virtual Environment (3 min)
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR venv\Scripts\activate  # Windows
\`\`\`

Verify: You should see `(venv)` in your terminal prompt.

## Step 3: Install Dependencies (5 min)
\`\`\`bash
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## Step 4: Configure Environment (2 min)
\`\`\`bash
cp .env.example .env
# Ask the Data Engineer for actual API keys to paste into .env
\`\`\`

## Step 5: Verify Installation (3 min)
\`\`\`bash
pytest tests/test_setup.py -v
\`\`\`

Expected: All 4 tests should pass. ✅

## Troubleshooting

**"python3: command not found"**
- Install Python from [python.org](https://python.org)
- On Mac, try: `brew install python3`

**"ModuleNotFoundError: No module named 'pandas'"**
- Make sure venv is activated (see `(venv)` prompt)
- Run: `pip install -r requirements.txt`

**"Permission denied" on venv/Scripts/activate**
- On Windows PowerShell, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Still stuck?**
- Post in Slack #rental-estimator
- Check README.md for additional help

## Next Steps
- Data Engineer: Download data files for Story 2
- ML Engineer: Start EDA in notebooks/