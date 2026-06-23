"""A keyless live-weather tool for the `assembly live` voice agent.

Backed by Open-Meteo, which needs no API key — so unlike the optional Firecrawl
search, this tool is *always* present, giving every live session at least one real
capability. The flow is geocode (place name -> coordinates) -> forecast (current
conditions + a multi-day daily outlook) -> a labelled, multi-line report.

The report is deliberately **comprehensive**, not a single spoken sentence: it
surfaces every field an LLM might need to answer a follow-up — feels-like
temperature, humidity, precipitation (amount *and* probability), wind speed /
direction / gusts, cloud cover, pressure, day/night, the UV index, and sunrise /
sunset — across today plus the next several days. The live agent reads aloud only
the slice the conversation calls for; the rest is there when a turn drills in.

The only network seam is :data:`Fetcher` (a ``url -> parsed JSON`` callable),
injected in tests so the whole flow runs with no sockets — the same shape
other URL-fetch tools in the live agent use. Everything else (the WMO-code text, the
spoken formatting) is pure and tested directly. Failures never raise out to the graph:
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
_FORECAST_DAYS = 7  # today + the next six days of daily outlook

# The current-conditions variables we ask Open-Meteo for (defaults: °C, km/h, %, mm, hPa).
_CURRENT_SERIES = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "is_day",
)
# The daily-outlook variables. ``time`` is always returned by Open-Meteo and is *not*
# a requestable variable, so it is read from the response but kept out of the request.
_DAILY_VARS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "uv_index_max",
    "sunrise",
    "sunset",
)
# Every parallel daily array we read out of the response (includes the auto-returned time).
_DAILY_SERIES = ("time", *_DAILY_VARS)
# The arrays a day row must have to be worth rendering; the shortest of them bounds how
# many complete days we emit, so a ragged response skips trailing days instead of raising.
_DAILY_REQUIRED = ("temperature_2m_max", "temperature_2m_min", "weather_code")

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

# 16-point compass, indexed by (bearing / 22.5) rounded mod 16, for spoken wind direction.
_COMPASS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


def describe_weather_code(code: int) -> str:
    """Return a short spoken phrase for a WMO weather code, or a generic fallback."""
    return _WMO_DESCRIPTIONS.get(code, "unsettled weather")


def describe_wind_direction(degrees: int) -> str:
    """Return the 16-point compass name (e.g. ``NW``) for a wind bearing in degrees."""
    return _COMPASS[round(degrees / 22.5) % 16]


def _c_to_f(celsius: float) -> int:
    """Convert Celsius to a rounded Fahrenheit integer for the spoken report."""
    return round(celsius * 9 / 5 + 32)


def _num(value: float) -> str:
    """Render a measurement to at most one decimal, dropping a trailing ``.0`` (``2`` not ``2.0``)."""
    return f"{round(value, 1):g}"


def _local_time(iso: str) -> str:
    """Pull the ``HH:MM`` out of an Open-Meteo local timestamp (``2026-06-22T05:48`` -> ``05:48``)."""
    return iso[11:16] if "T" in iso else iso


def _get_json(url: str) -> object:
    """GET ``url`` and return its parsed JSON body (the default network seam)."""
    import httpx2 as httpx

    response = httpx.get(url, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _geocode(name: str, *, fetch: Fetcher) -> tuple[str, str | None, float, float] | None:
    """Resolve a place name to ``(display name, country, latitude, longitude)``, or None.

    Asks Open-Meteo's geocoding endpoint for the single best match. No match (an
    empty or absent ``results`` list) returns None so the tool can speak a clear
    "couldn't find that place" instead of guessing. ``country`` is None when the
    match carries no country (so the report just names the place).
    """
    query = urlencode({"name": name, "count": 1, "language": "en", "format": "json"})
    payload = jsonshape.as_mapping(fetch(f"{_GEOCODE_URL}?{query}"))
    results = jsonshape.mapping_list(payload.get("results")) if payload is not None else []
    if not results:
        return None
    top = results[0]
    country = top.get("country")
    return (
        str(top.get("name", name)),
        str(country) if country is not None else None,
        jsonshape.as_float(top.get("latitude")),
        jsonshape.as_float(top.get("longitude")),
    )


def _forecast(lat: float, lon: float, *, fetch: Fetcher) -> dict[str, object]:
    """Fetch the full current conditions plus a multi-day daily outlook for coordinates."""
    query = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join(_CURRENT_SERIES),
            "daily": ",".join(_DAILY_VARS),
            "forecast_days": _FORECAST_DAYS,
            "timezone": "auto",
        }
    )
    return jsonshape.as_mapping(fetch(f"{_FORECAST_URL}?{query}")) or {}


def _format_current(name: str, country: str | None, current: dict[str, object]) -> list[str]:
    """Render the current-conditions block as a few labelled sentences."""
    place = f"{name}, {country}" if country else name
    temp = jsonshape.as_float(current.get("temperature_2m"))
    feels = jsonshape.as_float(current.get("apparent_temperature"))
    desc = describe_weather_code(jsonshape.as_int(current.get("weather_code")))
    daypart = "daytime" if jsonshape.as_int(current.get("is_day")) else "nighttime"
    humidity = round(jsonshape.as_float(current.get("relative_humidity_2m")))
    cloud = round(jsonshape.as_float(current.get("cloud_cover")))
    precip = jsonshape.as_float(current.get("precipitation"))
    wind = round(jsonshape.as_float(current.get("wind_speed_10m")))
    gust = round(jsonshape.as_float(current.get("wind_gusts_10m")))
    bearing = round(jsonshape.as_float(current.get("wind_direction_10m")))
    pressure = round(jsonshape.as_float(current.get("pressure_msl")))
    return [
        f"Current conditions in {place}: {round(temp)}°C ({_c_to_f(temp)}°F), "
        f"feels like {round(feels)}°C ({_c_to_f(feels)}°F), {desc}, {daypart}.",
        f"Humidity {humidity}%, cloud cover {cloud}%, precipitation {_num(precip)} mm.",
        f"Wind {wind} km/h from the {describe_wind_direction(bearing)} ({bearing}°), "
        f"gusting to {gust} km/h.",
        f"Pressure {pressure} hPa.",
    ]


def _day_label(offset: int, date: str) -> str:
    """Label a daily row: today/tomorrow carry their date, later days are the date alone."""
    if offset == 0:
        return f"Today ({date})"
    if offset == 1:
        return f"Tomorrow ({date})"
    return date


def _daily_rows(daily: dict[str, object]) -> list[dict[str, object]]:
    """Transpose Open-Meteo's parallel daily arrays into per-day row mappings.

    The number of complete rows is bounded by the shortest *required* array, so a
    response that drops a trailing value simply yields fewer days instead of raising.
    """
    columns = {field: jsonshape.object_list(daily.get(field)) for field in _DAILY_SERIES}
    count = min(len(columns[field]) for field in _DAILY_REQUIRED)
    return [
        {field: values[index] for field, values in columns.items() if index < len(values)}
        for index in range(count)
    ]


def _format_day(offset: int, row: dict[str, object]) -> str:
    """Render one daily row as a single comprehensive outlook line."""
    date = str(row.get("time", ""))
    high = jsonshape.as_float(row.get("temperature_2m_max"))
    low = jsonshape.as_float(row.get("temperature_2m_min"))
    cond = describe_weather_code(jsonshape.as_int(row.get("weather_code")))
    precip = _num(jsonshape.as_float(row.get("precipitation_sum")))
    prob = round(jsonshape.as_float(row.get("precipitation_probability_max")))
    wind = round(jsonshape.as_float(row.get("wind_speed_10m_max")))
    gust = round(jsonshape.as_float(row.get("wind_gusts_10m_max")))
    bearing = round(jsonshape.as_float(row.get("wind_direction_10m_dominant")))
    uv = _num(jsonshape.as_float(row.get("uv_index_max")))
    sunrise = _local_time(str(row.get("sunrise", "")))
    sunset = _local_time(str(row.get("sunset", "")))
    return (
        f"{_day_label(offset, date)}: high {round(high)}°C ({_c_to_f(high)}°F), "
        f"low {round(low)}°C ({_c_to_f(low)}°F), {cond}. "
        f"Precipitation {precip} mm, {prob}% chance. "
        f"Wind up to {wind} km/h, gusts {gust} km/h from the {describe_wind_direction(bearing)}. "
        f"UV index {uv}. Sunrise {sunrise}, sunset {sunset}."
    )


def format_report(name: str, country: str | None, data: dict[str, object]) -> str:
    """Render the Open-Meteo forecast as one comprehensive, labelled multi-line report.

    Temperatures are given in both units (the agent speaks whichever fits the
    conversation); everything else uses Open-Meteo's metric defaults (km/h, %, mm,
    hPa). The current block is followed by one line per forecast day.
    """
    current = jsonshape.as_mapping(data.get("current")) or {}
    daily = jsonshape.as_mapping(data.get("daily")) or {}
    lines = _format_current(name, country, current)
    lines.extend(_format_day(offset, row) for offset, row in enumerate(_daily_rows(daily)))
    return "\n".join(lines)


def build_weather_tool(fetch: Fetcher = _get_json) -> BaseTool:
    """Wrap the Open-Meteo lookup as the ``get_weather`` tool (``fetch`` injectable)."""
    from langchain_core.tools import tool

    @tool(WEATHER_TOOL_NAME)
    def get_weather(location: str) -> str:
        """Get the current weather and a multi-day forecast for a place by name (e.g. a
        city). Use when asked about the weather, temperature, or forecast somewhere."""
        try:
            located = _geocode(location, fetch=fetch)
            if located is None:
                return f"I couldn't find a place called '{location}'."
            name, country, lat, lon = located
            return format_report(name, country, _forecast(lat, lon, fetch=fetch))
        except Exception:
            # Best-effort: a transient Open-Meteo outage (the fetch seam raises) must
            # not bubble into brain's "couldn't complete the turn" path and kill the
            # spoken reply — speak a short apology instead. Mirrors mcp_tools._safe_load.
            return "I couldn't get the weather right now."

    return get_weather
