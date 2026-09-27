"""Plain-English explanations of rent predictions, generated through OpenRouter.

OpenRouter exposes many providers' models behind one OpenAI-compatible API, so
the model is a setting (EXPLAIN_MODEL), not code. If the key is missing or the
request fails for any reason, explain() returns a template explanation built
from the same SHAP drivers - the app never breaks because the API did.

Usage from the app:

    client = ExplanationClient()          # build once, e.g. in st.cache_resource
    result = client.explain(suburb_row, shap_explainer.explain(X_row))
    st.info(result["explanation"])

Try it against the real API:  python -m app.utils.api_client footscray 2 flat
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

try:
    from .shap_explainer import Driver, PredictionExplanation, growth_is_known
except ImportError:  # Supports direct execution: python app/utils/api_client.py
    from shap_explainer import Driver, PredictionExplanation, growth_is_known  # type: ignore


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
# Tried in order by OpenRouter if the primary model is unavailable.
DEFAULT_BACKUP_MODELS = ("openai/gpt-4o-mini",)
TIMEOUT_SECONDS = (3.05, 10)  # (connect, read)
MAX_TOKENS = 200
TOP_DRIVERS = 5

SYSTEM_PROMPT = """You explain rental price estimates to renters in Victoria, Australia.

Rules:
- Use only the facts you are given. Do not invent numbers, amenities, landmarks, transport links or market trends.
- Never add, combine, subtract or otherwise calculate numbers. Quote each dollar figure exactly as given, and give each factor its own figure.
- The factors come from a statistical model. Describe their effect on the estimate ("raises the estimate by about $126", "the model links this with a lower estimate"). Never say a factor adds to, costs, reduces or causes the rent itself.
- Mention the two or three largest factors and whether each raises or lowers the estimate.
- Write 2 to 3 sentences of plain text, under 90 words. No markdown, no lists, no headings.
- Money is in Australian dollars per week."""


class ExplanationClient:
    """Generates and caches explanations; safe to share across app sessions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        backup_models: Sequence[str] = DEFAULT_BACKUP_MODELS,
        session: Optional[Any] = None,
        timeout: Tuple[float, float] = TIMEOUT_SECONDS,
    ):
        self.api_key = _setting("OPENROUTER_API_KEY") if api_key is None else api_key
        self.model = model or _setting("EXPLAIN_MODEL") or DEFAULT_MODEL
        self.backup_models = [m for m in backup_models if m != self.model]
        self.session = session or requests.Session()
        self.timeout = timeout
        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def explain(
        self, suburb_row: Mapping[str, Any], result: PredictionExplanation
    ) -> Dict[str, Any]:
        """Return an explanation for one prediction.

        Keys: explanation, used_fallback, model, latency_s, prompt_tokens,
        completion_tokens, cost_usd, error.
        """
        cache_key = (
            str(suburb_row["suburb"]),
            int(suburb_row["bedrooms"]),
            str(suburb_row["property_type"]),
            round(result.prediction),
            self.model,
        )
        if cache_key in self._cache:
            return {**self._cache[cache_key], "latency_s": 0.0}

        start = time.perf_counter()
        drivers = result.top_drivers(TOP_DRIVERS, include_growth=growth_is_known(suburb_row))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(suburb_row, result, drivers)},
        ]

        if not self.api_key:
            response = _fallback(suburb_row, result, drivers, "OPENROUTER_API_KEY is not set")
        else:
            response = self._call_api(messages)
            if response.get("error"):
                response = _fallback(suburb_row, result, drivers, response["error"])

        response["latency_s"] = round(time.perf_counter() - start, 3)
        if not response["used_fallback"]:
            self._cache[cache_key] = dict(response)
        return response

    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "models": [self.model, *self.backup_models],
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3,
            "usage": {"include": True},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Rental Price Estimator",
        }
        try:
            reply = self.session.post(
                OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as exc:
            return {"error": f"Request failed: {type(exc).__name__}"}

        if reply.status_code != 200:
            return {"error": f"OpenRouter returned HTTP {reply.status_code}"}

        try:
            body = reply.json()
            text = (body["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError):
            return {"error": "Unexpected response format from OpenRouter"}
        if not text:
            return {"error": "OpenRouter returned an empty explanation"}

        usage = body.get("usage") or {}
        return {
            "explanation": text,
            "used_fallback": False,
            "model": body.get("model", self.model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"),
            "error": None,
        }


def build_prompt(
    suburb_row: Mapping[str, Any], result: PredictionExplanation, drivers: Sequence[Driver]
) -> str:
    """The user message: only facts, so the model has nothing to make up."""
    lines = [
        f"Suburb: {_title(suburb_row['suburb'])}",
        f"Property: {_describe_property(suburb_row)}",
        f"Estimated rent: ${result.prediction:,.0f} per week",
        f"Model's starting point before suburb and property details: ${result.baseline:,.0f} per week",
        "",
        "Largest factors in this estimate (effect on weekly rent):",
    ]
    lines += [f"- {_describe_driver(d)}" for d in drivers]
    income = float(suburb_row["Median_Household_Income_Weekly_AUD"])
    seifa = float(suburb_row["IRSAD_Score"])
    lines += [
        "",
        "Suburb facts:",
        f"- Median household income: ${income:,.0f} per week",
        f"- Socio-economic advantage score (SEIFA): {seifa:,.0f} (Australian average about 1,000)",
        f"- Distance to the Melbourne CBD: {float(suburb_row['distance_to_cbd_km']):.1f} km",
    ]
    if growth_is_known(suburb_row):
        lines.append(
            f"- House price growth in 2024-25: {float(suburb_row['house_change_perc_24-25']):+.1f}%"
        )
    else:
        lines.append("- House price growth: not available for this suburb; do not mention it")
    return "\n".join(lines)


def _fallback(
    suburb_row: Mapping[str, Any],
    result: PredictionExplanation,
    drivers: Sequence[Driver],
    error: str,
) -> Dict[str, Any]:
    factors = "; ".join(_describe_driver(d) for d in drivers[:3])
    text = (
        f"The estimated rent for a {_describe_property(suburb_row)} in "
        f"{_title(suburb_row['suburb'])} is ${result.prediction:,.0f} per week. "
        f"The factors that moved this estimate most were: {factors}."
    )
    return {
        "explanation": text,
        "used_fallback": True,
        "model": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
        "error": error,
    }


def _describe_driver(driver: Driver) -> str:
    direction = "raises" if driver.effect >= 0 else "lowers"
    return (
        f"{driver.label} ({driver.value}) {direction} the estimate "
        f"by about ${abs(driver.effect):,.0f}"
    )


def _describe_property(suburb_row: Mapping[str, Any]) -> str:
    return f"{int(suburb_row['bedrooms'])}-bedroom {suburb_row['property_type']}"


def _title(name: Any) -> str:
    return str(name).title()


def _setting(name: str) -> Optional[str]:
    """Read a setting from the environment / .env, then Streamlit secrets."""
    load_dotenv()
    value = os.getenv(name)
    if value:
        return value
    try:
        from streamlit import runtime

        if not runtime.exists():  # Plain Python, not `streamlit run`
            return None
        import streamlit as st

        return st.secrets.get(name)
    except Exception:  # No secrets file configured
        return None


def _demo(argv: Sequence[str]) -> None:
    """Explain one real prediction with the trained model and the live API."""
    import pickle
    import sys
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from app.utils.shap_explainer import ShapExplainer, find_suburb_row

    suburb, bedrooms, property_type = (argv + ["footscray", "2", "flat"][len(argv):])[:3]
    model_path = root / "models" / "xgboost_tuned.pkl"
    if not model_path.exists():
        raise SystemExit(
            "models/xgboost_tuned.pkl not found. Run python src/models/train.py "
            "and python src/models/tune_hyperparameters.py first."
        )
    with model_path.open("rb") as file:
        model = pickle.load(file)
    merged = pd.read_csv(root / "data" / "processed" / "merged_rental_data.csv")

    row = find_suburb_row(merged, suburb, int(bedrooms), property_type)
    explainer = ShapExplainer(model)
    result = explainer.explain(explainer.feature_row(row))
    response = ExplanationClient().explain(row, result)

    print(f"{_describe_property(row)} in {_title(row['suburb'])}: ${result.prediction:,.0f}/week")
    print(f"(actual median in the data: ${float(row['median_weekly_rent']):,.0f}/week)\n")
    print(response["explanation"], "\n")
    print(f"model={response['model']}  fallback={response['used_fallback']}  "
          f"latency={response['latency_s']}s  tokens={response['prompt_tokens']}+"
          f"{response['completion_tokens']}  cost=${response['cost_usd']}")
    if response["error"]:
        print(f"error: {response['error']}")


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1:])
