# Running and deploying the Streamlit app

## What was added

| File | What it does |
|---|---|
| `app/main.py` | Home page (the file you run) |
| `app/pages/1_Predict.py` | Estimate, "Why this price?" SHAP chart and AI explanation, similar suburbs |
| `app/pages/2_Comparables.py` | Similar suburbs on a bar chart and a map |
| `app/pages/3_FAQ.py`, `app/pages/4_About.py` | FAQ and About |
| `app/utils/model_loader.py` | Loads `models/xgboost_tuned.pkl`, or rebuilds the tuned model from `models/xgboost_tuned.json` when the .pkl is missing (it is gitignored, so it is never on GitHub or Streamlit Cloud) |
| `app/utils/app_state.py` | Cached data, model, SHAP explainer, OpenRouter client, comparables |
| `app/requirements.txt` | Slim package list that Streamlit Cloud installs |
| `.streamlit/config.toml`, `.streamlit/secrets.toml.example` | Theme, and a template for the API key |

The app uses the team's existing `shap_explainer.py` and `api_client.py` unchanged.
`requirements.txt` now pins `streamlit==1.40.2` (1.27 is too old for page switching).

## Run it on your laptop (Windows)

```powershell
cd BUS5003-A02_RentalPriceEstimator
venv\Scripts\Activate.ps1
pip install -r app/requirements.txt
streamlit run app/main.py
```

It opens at http://localhost:8501. For AI explanations, put `OPENROUTER_API_KEY=...` in your `.env` file.
Without a key the app still works and shows a template explanation.

## Deploy on Streamlit Community Cloud

1. Push these files to the `main` branch on GitHub.
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Repository `mlasheen7/BUS5003-A02_RentalPriceEstimator`, branch `main`, main file path `app/main.py`.
4. Open **Advanced settings**: choose **Python 3.11**, and paste into **Secrets**:
   ```
   OPENROUTER_API_KEY = "your key"
   EXPLAIN_MODEL = "openai/gpt-4o-mini"
   ```
5. Click **Deploy**. The first build takes about 5 minutes.

The repo is private, so whoever deploys needs access to it (the repo owner is easiest).
