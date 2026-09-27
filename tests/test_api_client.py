"""Tests for app/utils/api_client.py.

The HTTP session is always a fake, so these never touch the network, never
need a key, and cost nothing.
"""
import pytest
import requests

from app.utils.api_client import OPENROUTER_URL, ExplanationClient, build_prompt
from app.utils.shap_explainer import Driver, PredictionExplanation

FOOTSCRAY = {
    "suburb": "footscray",
    "bedrooms": 2,
    "property_type": "flat",
    "Median_Household_Income_Weekly_AUD": 1763.0,
    "IRSAD_Score": 1030.0,
    "distance_to_cbd_km": 6.0,
    "house_change_perc_24-25": -4.0,
}
RICHMOND = {**FOOTSCRAY, "suburb": "richmond", "house_change_perc_24-25": 0.0}

RESULT = PredictionExplanation(
    prediction=531.4,
    baseline=590.7,
    drivers=[
        Driver("bedrooms", "number of bedrooms", "2", -70.0),
        Driver("distance_to_cbd_km", "distance to the Melbourne CBD", "6.0 km", 29.0),
        Driver("price_growth", "house price growth band", "declining", 20.0),
        Driver("Unemployment_Rate_Pct", "unemployment rate", "6.6%", -18.0),
        Driver("house_change_perc_24-25", "house price growth in 2024-25", "-4.0%", 15.0),
        Driver("IRSAD_Score", "socio-economic advantage score (SEIFA)", "1,030", -13.0),
    ],
)


class FakeResponse:
    def __init__(self, status_code=200, body=None, bad_json=False):
        self.status_code = status_code
        self._body = body
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def ok_body(text="Footscray rent is below the typical estimate mainly because it is a 2-bedroom flat."):
    return {
        "model": "anthropic/claude-haiku-4.5",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 380, "completion_tokens": 60, "cost": 0.00068},
    }


def make_client(session, api_key="test-key", model="anthropic/claude-haiku-4.5"):
    return ExplanationClient(api_key=api_key, model=model, session=session)


def test_successful_call_returns_the_model_text():
    session = FakeSession(FakeResponse(body=ok_body()))
    out = make_client(session).explain(FOOTSCRAY, RESULT)

    assert out["explanation"].startswith("Footscray rent")
    assert out["used_fallback"] is False
    assert out["error"] is None
    assert out["model"] == "anthropic/claude-haiku-4.5"
    assert (out["prompt_tokens"], out["completion_tokens"]) == (380, 60)
    assert out["cost_usd"] == pytest.approx(0.00068)
    assert out["latency_s"] >= 0


def test_request_uses_openrouter_with_the_configured_model_and_key():
    session = FakeSession(FakeResponse(body=ok_body()))
    make_client(session, model="openai/gpt-4o-mini").explain(FOOTSCRAY, RESULT)

    url, kwargs = session.calls[0]
    assert url == OPENROUTER_URL
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "openai/gpt-4o-mini"
    assert kwargs["json"]["models"][0] == "openai/gpt-4o-mini"
    assert "openai/gpt-4o-mini" not in kwargs["json"]["models"][1:]
    assert kwargs["timeout"]


def test_model_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "some/other-model")
    client = ExplanationClient(api_key="k", session=FakeSession())
    assert client.model == "some/other-model"


def test_prompt_carries_the_facts_and_the_rules():
    session = FakeSession(FakeResponse(body=ok_body()))
    make_client(session).explain(FOOTSCRAY, RESULT)
    system, user = session.calls[0][1]["json"]["messages"]

    assert system["role"] == "system"
    for rule in ["Use only the facts", "Never add, combine, subtract",
                 "Never say a factor adds to, costs, reduces or causes the rent"]:
        assert rule in system["content"], rule
    assert user["role"] == "user"
    for fact in ["Footscray", "2-bedroom flat", "$531 per week", "$1,763 per week",
                 "1,030", "6.0 km", "lowers the estimate by about $70", "-4.0%"]:
        assert fact in user["content"], fact


def test_unknown_growth_is_kept_out_of_the_prompt():
    prompt = build_prompt(RICHMOND, RESULT, RESULT.top_drivers(5, include_growth=False))
    assert "growth" not in prompt.split("Suburb facts:")[0].lower()
    assert "not available for this suburb; do not mention it" in prompt

    session = FakeSession(FakeResponse(body=ok_body()))
    make_client(session).explain(RICHMOND, RESULT)
    sent = session.calls[0][1]["json"]["messages"][1]["content"]
    assert "house price growth band" not in sent
    assert "house price growth in 2024-25" not in sent


def test_missing_key_falls_back_without_calling_the_api():
    session = FakeSession(FakeResponse(body=ok_body()))
    out = make_client(session, api_key="").explain(FOOTSCRAY, RESULT)

    assert session.calls == []
    assert out["used_fallback"] is True
    assert "OPENROUTER_API_KEY" in out["error"]
    assert "$531 per week" in out["explanation"]
    assert "number of bedrooms (2) lowers the estimate by about $70" in out["explanation"]


@pytest.mark.parametrize(
    "session",
    [
        FakeSession(exc=requests.Timeout()),
        FakeSession(exc=requests.ConnectionError()),
        FakeSession(FakeResponse(status_code=429)),
        FakeSession(FakeResponse(status_code=500)),
        FakeSession(FakeResponse(bad_json=True)),
        FakeSession(FakeResponse(body={"choices": []})),
        FakeSession(FakeResponse(body=ok_body(text="   "))),
    ],
    ids=["timeout", "connection", "429", "500", "bad-json", "no-choices", "empty-text"],
)
def test_any_api_failure_falls_back(session):
    out = make_client(session).explain(FOOTSCRAY, RESULT)
    assert out["used_fallback"] is True
    assert out["error"]
    assert "Footscray" in out["explanation"]


def test_fallback_does_not_mention_unknown_growth():
    out = make_client(FakeSession(exc=requests.Timeout())).explain(RICHMOND, RESULT)
    assert "growth" not in out["explanation"].lower()


def test_repeat_requests_are_served_from_the_cache():
    session = FakeSession(FakeResponse(body=ok_body()))
    client = make_client(session)
    first = client.explain(FOOTSCRAY, RESULT)
    second = client.explain(FOOTSCRAY, RESULT)

    assert len(session.calls) == 1
    assert second["explanation"] == first["explanation"]
    assert second["latency_s"] == 0.0


def test_fallbacks_are_not_cached():
    session = FakeSession(FakeResponse(status_code=503))
    client = make_client(session)
    client.explain(FOOTSCRAY, RESULT)
    client.explain(FOOTSCRAY, RESULT)
    assert len(session.calls) == 2
