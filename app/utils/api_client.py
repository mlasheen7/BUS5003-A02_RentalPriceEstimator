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
DEFAULT_MODEL = "openai/gpt-4o-mini"
# Tried in order by OpenRouter if the primary model is unavailable.
DEFAULT_BACKUP_MODELS = ("anthropic/claude-haiku-4.5",)
TIMEOUT_SECONDS = (3.05, 10)  # (connect, read)
MAX_TOKENS = 200
TOP_DRIVERS = 5

SYSTEM_PROMPT = """You explain rental price estimates to renters in Victoria, Australia.

Rules:
- Use only the facts you are given. Do not invent numbers, amenities, landmarks, transport links or market trends.
- Never add, combine, subtract or otherwise calculate numbers. Quote each dollar figure exactly as given, and give each factor its own figure.
- The factors come from a statistical model. Describe their effect on the estimate ("raises the estimate by about $126", "the model links this with a lower estimate"). Never say a factor adds to, costs, reduces or causes the rent itself.
- Mention the two or three largest factors and whether each raises or lowers the estimate.
- Stop after the last factor. No closing summary or filler phrase about the overall price.
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
        """Return an explanation for one prediction."""
        
        print(f"\n🔍 explain() called")
        print(f"   Suburb: {suburb_row['suburb']}")
        print(f"   API Key set: {bool(self.api_key)}")
        
        cache_key = (
            str(suburb_row["suburb"]),
            int(suburb_row["bedrooms"]),
            str(suburb_row["property_type"]),
            round(result.prediction),
            self.model,
        )
        
        if cache_key in self._cache:
            print(f"   📦 Returning cached response")
            return {**self._cache[cache_key], "latency_s": 0.0}

        start = time.perf_counter()
        drivers = result.top_drivers(TOP_DRIVERS, include_growth=growth_is_known(suburb_row))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(suburb_row, result, drivers)},
        ]

        if not self.api_key:
            print(f"   ❌ No API key - using fallback")
            response = _fallback(suburb_row, result, drivers, "OPENROUTER_API_KEY is not set")
        else:
            print(f"   🔄 Calling API...")
            response = self._call_api(messages)
            if response.get("error"):
                print(f"   ❌ API error: {response['error']} - using fallback")
                response = _fallback(suburb_row, result, drivers, response["error"])
            else:
                print(f"   ✅ API success")

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
            "provider": {"sort": "latency"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Rental Price Estimator",
        }
        
        # DEBUG: Print API call details
        print(f"\n📡 _call_api() executing")
        print(f"   Model: {self.model}")
        print(f"   API Key present: {bool(self.api_key)}")
        print(f"   API Key: {self.api_key[:20] if self.api_key else 'NONE'}...")
        print(f"   URL: {OPENROUTER_URL}")
        
        try:
            print(f"   Posting request...")
            reply = self.session.post(
                OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout
            )
            print(f"   Got response: HTTP {reply.status_code}")
            
        except requests.RequestException as exc:
            print(f"   ❌ Request failed: {type(exc).__name__}: {exc}")
            return {"error": f"Request failed: {type(exc).__name__}"}

        if reply.status_code != 200:
            print(f"   ❌ HTTP Error: {reply.status_code}")
            print(f"   Response: {reply.text[:300]}")
            return {"error": f"OpenRouter returned HTTP {reply.status_code}"}

        try:
            body = reply.json()
            text = (body["choices"][0]["message"]["content"] or "").strip()
            print(f"   ✅ Got text: {text[:50]}...")
            
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"   ❌ Parse error: {e}")
            print(f"   Response body: {reply.text[:200]}")
            return {"error": "Unexpected response format from OpenRouter"}
            
        if not text:
            print(f"   ❌ Empty text returned")
            return {"error": "OpenRouter returned an empty explanation"}

        usage = body.get("usage") or {}
        print(f"   ✅ Success! Cost: ${usage.get('cost', 'unknown')}")
        
        return {
            "explanation": text,
            "used_fallback": False,
            "model": body.get("model", self.model),
            "provider": body.get("provider"),
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
        "Largest factors in this estimate (effect on the weekly estimate):",
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
        "provider": None,
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
    
    # DEBUG: Print where key came from
    print(f"🔍 DEBUG: Looking for {name}")
    print(f"   From .env: {value[:15] if value else 'NOT FOUND'}...")
    
    if value:
        return value
    try:
        from streamlit import runtime

        if not runtime.exists():
            print(f"   Not in Streamlit session")
            return None
        import streamlit as st

        st_value = st.secrets.get(name)
        print(f"   From st.secrets: {st_value[:15] if st_value else 'NOT FOUND'}...")
        return st_value
    except Exception as e:
        print(f"   Error reading secrets: {e}")
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
    print(f"model={response['model']} via {response['provider']}  "
          f"fallback={response['used_fallback']}  "
          f"latency={response['latency_s']}s  tokens={response['prompt_tokens']}+"
          f"{response['completion_tokens']}  cost=${response['cost_usd']}")
    if response["error"]:
        print(f"error: {response['error']}")


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1:])
