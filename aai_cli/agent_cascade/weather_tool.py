"""A keyless live-weather tool for the `assembly live` voice agent.

Backed by Open-Meteo, which needs no API key — so unlike the optional Firecrawl
search, this tool is *always* present, giving every live session at least one real
capability. The flow is geocode (place name -> coordinates) -> forecast (current +
a short daily outlook) -> a single short string the agent reads aloud.

The only network seam is :data:`Fetcher` (a ``url -> parsed JSON`` callable),
injected in tests so the whole flow runs with no sockets — the same shape
other URL-fetch tools in the live agent use. Everything else (the WMO-code text, the spoken
formatting) is pure and tested directly. Failures never raise out to the graph:
``get_weather`` catches them and returns a short spoken apology so a weather
outage can't sink a live turn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from aai_cli.core import jsonshape

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# The registered tool name. ``brain.py`` detects weather availability and labels the
# live-UI affordance by this name, so a test pins it.
WEATHER_TOOL_NAME = "get_weather"

# A fetcher GETs a URL and returns parsed JSON. Injected in tests (the only net seam).
# This is the same pattern used by the URL-fetch tools in the live agent.
Fetcher = Callable[[str], object]

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 15.0  # pragma: no mutate — a tuning knob; ±a few seconds is equivalent
_FORECAST_DAYS = 3  # today + the next two days (the two spoken outlook lines)

# WMO weather-interpretation codes -> short spoken phrases. A code not listed here
# (Open-Meteo can add more) falls back in :func:`describe_weather_code` rather than
# raising, so an unfamiliar code never sinks a turn.
_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}

# Spoken labels for the next two forecast days (index 1 and 2 of the daily arrays).
_DAY_LABELS = ("Tomorrow", "Then")


def describe_weather_code(code: int) -> str:
    """Return a short spoken phrase for a WMO weather code, or a generic fallback."""
    return _WMO_DESCRIPTIONS.get(code, "unsettled weather")


def _c_to_f(celsius: float) -> int:
    """Convert Celsius to a rounded Fahrenheit integer for the spoken report."""
    return round(celsius * 9 / 5 + 32)


def _get_json(url: str) -> object:
    """GET ``url`` and return its parsed JSON body (the default network seam)."""
    import httpx

    response = httpx.get(url, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _geocode(name: str, *, fetch: Fetcher) -> tuple[str, float, float] | None:
    """Resolve a place name to ``(display name, latitude, longitude)``, or None.

    Asks Open-Meteo's geocoding endpoint for the single best match. No match (an
    empty or absent ``results`` list) returns None so the tool can speak a clear
    "couldn't find that place" instead of guessing.
    """
    query = urlencode({"name": name, "count": 1, "language": "en", "format": "json"})
    payload = jsonshape.as_mapping(fetch(f"{_GEOCODE_URL}?{query}"))
    results = jsonshape.mapping_list(payload.get("results")) if payload is not None else []
    if not results:
        return None
    top = results[0]
    return (
        str(top.get("name", name)),
        jsonshape.as_float(top.get("latitude")),
        jsonshape.as_float(top.get("longitude")),
    )


def _forecast(lat: float, lon: float, *, fetch: Fetcher) -> dict[str, object]:
    """Fetch the current conditions plus a short daily outlook for coordinates."""
    query = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "forecast_days": _FORECAST_DAYS,
            "timezone": "auto",
        }
    )
    return jsonshape.as_mapping(fetch(f"{_FORECAST_URL}?{query}")) or {}


def _forecast_lines(daily: dict[str, object]) -> list[str]:
    """The spoken outlook lines for the next days, e.g. ``Tomorrow 9 to 17°C, rain.``"""
    highs = jsonshape.object_list(daily.get("temperature_2m_max"))
    lows = jsonshape.object_list(daily.get("temperature_2m_min"))
    codes = jsonshape.object_list(daily.get("weather_code"))
    lines: list[str] = []
    for offset, label in enumerate(_DAY_LABELS, start=1):
        if offset < len(highs) and offset < len(lows) and offset < len(codes):
            low = round(jsonshape.as_float(lows[offset]))
            high = round(jsonshape.as_float(highs[offset]))
            cond = describe_weather_code(jsonshape.as_int(codes[offset]))
            lines.append(f"{label} {low} to {high}°C, {cond}.")
    return lines


def format_report(name: str, data: dict[str, object]) -> str:
    """Render the Open-Meteo forecast as one short, speakable string.

    The current temperature is given in both units (the agent speaks whichever fits
    the conversation); the outlook days stay in °C to keep the spoken reply short.
    """
    current = jsonshape.as_mapping(data.get("current")) or {}
    daily = jsonshape.as_mapping(data.get("daily")) or {}
    temp = jsonshape.as_float(current.get("temperature_2m"))
    desc = describe_weather_code(jsonshape.as_int(current.get("weather_code")))
    lines = [f"In {name} it's {round(temp)}°C ({_c_to_f(temp)}°F) and {desc}."]
    lines.extend(_forecast_lines(daily))
    return " ".join(lines)


def build_weather_tool(fetch: Fetcher = _get_json) -> BaseTool:
    """Wrap the Open-Meteo lookup as the ``get_weather`` tool (``fetch`` injectable)."""
    from langchain_core.tools import tool

    @tool(WEATHER_TOOL_NAME)
    def get_weather(location: str) -> str:
        """Get the current weather and a short forecast for a place by name (e.g. a
        city). Use when asked about the weather, temperature, or forecast somewhere."""
        try:
            located = _geocode(location, fetch=fetch)
            if located is None:
                return f"I couldn't find a place called '{location}'."
            name, lat, lon = located
            return format_report(name, _forecast(lat, lon, fetch=fetch))
        except Exception:
            # Best-effort: a transient Open-Meteo outage (the fetch seam raises) must
            # not bubble into brain's "couldn't complete the turn" path and kill the
            # spoken reply — speak a short apology instead. Mirrors mcp_tools._safe_load.
            return "I couldn't get the weather right now."

    return get_weather
