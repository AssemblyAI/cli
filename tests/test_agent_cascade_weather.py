"""Tests for the keyless Open-Meteo weather tool behind `assembly live`.

The tool's only network seam is the injected ``fetch`` callable, so the whole
geocode -> forecast -> format flow runs with no sockets (pytest-socket stays armed).
"""

from __future__ import annotations

from aai_cli.agent_cascade import weather_tool

# Canned Open-Meteo payloads keyed by URL prefix, replayed through the fetch seam.
_GEOCODE = {
    "results": [{"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "France"}]
}
_FORECAST = {
    "current": {"temperature_2m": 14.3, "weather_code": 2},
    "daily": {
        "time": ["2026-06-22", "2026-06-23", "2026-06-24"],
        "temperature_2m_max": [17.2, 17.0, 19.1],
        "temperature_2m_min": [9.0, 9.4, 11.2],
        "weather_code": [2, 61, 0],
    },
}


def _fake_fetch(geocode=_GEOCODE, forecast=_FORECAST):
    """A fetch seam that returns canned geocode/forecast JSON by URL."""

    def fetch(url: str) -> object:
        return geocode if "geocoding-api" in url else forecast

    return fetch


# --- describe_weather_code ---------------------------------------------------


def test_describe_weather_code_known():
    assert weather_tool.describe_weather_code(0) == "clear sky"
    assert weather_tool.describe_weather_code(61) == "light rain"


def test_describe_weather_code_unknown_falls_back():
    # An unmapped WMO code must not raise; it returns the generic fallback.
    assert weather_tool.describe_weather_code(999) == "unsettled weather"


# --- _geocode ----------------------------------------------------------------


def test_geocode_returns_top_match_and_hits_geocoding_host():
    seen = {}

    def fetch(url: str) -> object:
        seen["url"] = url
        return _GEOCODE

    result = weather_tool._geocode("Paris", fetch=fetch)
    assert result == ("Paris", 48.85, 2.35)
    assert "geocoding-api.open-meteo.com" in seen["url"]
    assert "name=Paris" in seen["url"]


def test_geocode_no_results_is_none():
    assert weather_tool._geocode("Nowhereville", fetch=lambda url: {"results": []}) is None


def test_geocode_missing_results_key_is_none():
    assert weather_tool._geocode("x", fetch=lambda url: {}) is None


# --- _forecast ---------------------------------------------------------------


def test_forecast_requests_current_and_daily_for_coordinates():
    seen = {}

    def fetch(url: str) -> object:
        seen["url"] = url
        return _FORECAST

    data = weather_tool._forecast(48.85, 2.35, fetch=fetch)
    assert data == _FORECAST
    assert "api.open-meteo.com/v1/forecast" in seen["url"]
    assert "latitude=48.85" in seen["url"]
    assert "longitude=2.35" in seen["url"]
    assert "current=temperature_2m" in seen["url"]
    assert "daily=temperature_2m_max" in seen["url"]
    assert "forecast_days=3" in seen["url"]


# --- format_report -----------------------------------------------------------


def test_format_report_renders_current_in_both_units_and_two_forecast_days():
    report = weather_tool.format_report("Paris", _FORECAST)
    # Current line: rounded °C, derived °F, and the condition text.
    assert "In Paris it's 14°C (58°F) and partly cloudy." in report
    # Two forecast days, labelled, °C lows-to-highs with their own conditions.
    assert "Tomorrow 9 to 17°C, light rain." in report
    assert "Then 11 to 19°C, clear sky." in report


# --- build_weather_tool (end to end via the seam) ----------------------------


def test_tool_name_and_happy_path():
    tool = weather_tool.build_weather_tool(fetch=_fake_fetch())
    assert tool.name == weather_tool.WEATHER_TOOL_NAME == "get_weather"
    out = tool.invoke({"location": "Paris"})
    assert "In Paris it's 14°C (58°F) and partly cloudy." in out
    assert "Tomorrow 9 to 17°C, light rain." in out


def test_tool_location_not_found_message():
    tool = weather_tool.build_weather_tool(fetch=lambda url: {"results": []})
    assert tool.invoke({"location": "Nowhereville"}) == (
        "I couldn't find a place called 'Nowhereville'."
    )


def test_tool_network_error_is_graceful():
    def boom(url: str) -> object:
        raise RuntimeError("open-meteo down")

    tool = weather_tool.build_weather_tool(fetch=boom)
    assert tool.invoke({"location": "Paris"}) == "I couldn't get the weather right now."
