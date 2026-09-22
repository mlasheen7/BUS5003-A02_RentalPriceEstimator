# Development Environment Setup Guide

Complete these steps once. Time: ~15 minutes.

## Prerequisites
- Git installed ([git-scm.com](https://git-scm.com))
- **Python 3.9, 3.10 or 3.11** ([python.org](https://python.org))
- Access to this repository

> ⚠️ **Python 3.12+ will not work.** `pycaret 3.3.2` requires `numpy<1.27` and
> `pandas<2.2`, and neither `numpy 1.24.3` nor `pandas 2.0.3` publishes a wheel for
> 3.12. `pip install -r requirements.txt` will try to build them from source and fail.
> Check your version with `python3 --version` before you start.

## Step 1: Clone Repository (2 min)
```bash
cd ~/Documents  # or wherever you keep projects
git clone https://github.com/mlasheen7/BUS5003-A02_RentalPriceEstimator.git
cd BUS5003-A02_RentalPriceEstimator
```

## Step 2: Create Virtual Environment (3 min)
```bash
python3 -m venv venv

source venv/bin/activate        # Mac/Linux
# venv\Scripts\Activate.ps1     # Windows PowerShell
# venv\Scripts\activate.bat     # Windows cmd.exe
```

Verify: you should see `(venv)` at the start of your terminal prompt.

## Step 3: Install Dependencies (5 min)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Configure Environment (2 min)
```bash
cp .env.example .env            # Mac/Linux
# copy .env.example .env        # Windows
```

Then open `.env` and fill in your own values. **Use your own API key** — get one at
[console.anthropic.com](https://console.anthropic.com/). Never commit `.env`; it is
gitignored for a reason.

## Step 5: Verify Installation (3 min)
```bash
pytest
```

Expected: all tests pass. ✅

## Troubleshooting

**"python3: command not found"**
- Install Python from [python.org](https://python.org)
- On Mac, try: `brew install python@3.11`
- On Windows, `python3` may be `python` or `py -3.11`

**"Could not build wheels for numpy" / "pandas"**
- You are almost certainly on Python 3.12+. Check with `python3 --version`.
- Install Python 3.11 and rebuild the venv: `rm -rf venv && python3.11 -m venv venv`

**"ModuleNotFoundError: No module named 'pandas'"**
- Make sure the venv is activated (look for the `(venv)` prompt)
- Run: `pip install -r requirements.txt`

**"Permission denied" running venv\Scripts\Activate.ps1**
- In Windows PowerShell, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Tests fail on `data/raw`**
- `data/raw/` ships empty by design. The `.gitkeep` file keeps the directory; the
  source files are not committed. See `data/README.md` for how to get them.

**Still stuck?**
- Post in the team channel, or open an issue on this repo
- Check `README.md` and `CONTRIBUTING.md` for additional help

## Next Steps
- Read [CONTRIBUTING.md](CONTRIBUTING.md) — branch, PR and review process
- Read [data/README.md](data/README.md) — data sources and refresh procedure
- Pick up your role's guide from the repository root
