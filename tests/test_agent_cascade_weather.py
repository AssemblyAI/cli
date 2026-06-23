"""Tests for the keyless Open-Meteo weather tool behind `assembly live`.

The tool's only network seam is the injected ``fetch`` callable, so the whole
geocode -> forecast -> format flow runs with no sockets (pytest-socket stays armed).
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from aai_cli.agent_cascade import weather_tool

# Canned Open-Meteo payloads keyed by URL prefix, replayed through the fetch seam.
_GEOCODE: dict[str, object] = {
    "results": [{"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "France"}]
}
_FORECAST: dict[str, object] = {
    "current": {
        "temperature_2m": 14.3,
        "apparent_temperature": 12.8,
        "relative_humidity_2m": 72,
        "precipitation": 0.0,
        "weather_code": 2,
        "cloud_cover": 60,
        "wind_speed_10m": 12.0,
        "wind_direction_10m": 315,
        "wind_gusts_10m": 25.0,
        "pressure_msl": 1013.0,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-06-22", "2026-06-23", "2026-06-24"],
        "weather_code": [2, 61, 0],
        "temperature_2m_max": [17.2, 17.0, 19.1],
        "temperature_2m_min": [9.0, 9.4, 11.2],
        "precipitation_sum": [0.0, 4.2, 0.0],
        "precipitation_probability_max": [0, 60, 10],
        "wind_speed_10m_max": [20.0, 24.0, 18.0],
        "wind_gusts_10m_max": [35.0, 40.0, 30.0],
        "wind_direction_10m_dominant": [315, 225, 0],
        "uv_index_max": [5.0, 4.5, 6.0],
        "sunrise": ["2026-06-22T05:48", "2026-06-23T05:48", "2026-06-24T05:49"],
        "sunset": ["2026-06-22T21:56", "2026-06-23T21:57", "2026-06-24T21:57"],
    },
}

# The exact report the canned forecast must render to — pins every field/round/unit so a
# mutated format string, dropped field, or wrong conversion fails this assertion.
_EXPECTED_LINES = (
    "Current conditions in Paris, France: 14°C (58°F), feels like 13°C (55°F), "
    "partly cloudy, daytime.",
    "Humidity 72%, cloud cover 60%, precipitation 0 mm.",
    "Wind 12 km/h from the NW (315°), gusting to 25 km/h.",
    "Pressure 1013 hPa.",
    "Today (2026-06-22): high 17°C (63°F), low 9°C (48°F), partly cloudy. "
    "Precipitation 0 mm, 0% chance. Wind up to 20 km/h, gusts 35 km/h from the NW. "
    "UV index 5. Sunrise 05:48, sunset 21:56.",
    "Tomorrow (2026-06-23): high 17°C (63°F), low 9°C (49°F), light rain. "
    "Precipitation 4.2 mm, 60% chance. Wind up to 24 km/h, gusts 40 km/h from the SW. "
    "UV index 4.5. Sunrise 05:48, sunset 21:57.",
    "2026-06-24: high 19°C (66°F), low 11°C (52°F), clear sky. "
    "Precipitation 0 mm, 10% chance. Wind up to 18 km/h, gusts 30 km/h from the N. "
    "UV index 6. Sunrise 05:49, sunset 21:57.",
)
_EXPECTED_REPORT = "\n".join(_EXPECTED_LINES)


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


# --- describe_wind_direction -------------------------------------------------


def test_describe_wind_direction_cardinals_and_wraparound():
    # Pins the /22.5 step, the %16 wrap, and a few table indices: a mutated divisor,
    # modulus, or compass entry shifts at least one of these.
    assert weather_tool.describe_wind_direction(0) == "N"
    assert weather_tool.describe_wind_direction(45) == "NE"
    assert weather_tool.describe_wind_direction(90) == "E"
    assert weather_tool.describe_wind_direction(180) == "S"
    assert weather_tool.describe_wind_direction(225) == "SW"
    assert weather_tool.describe_wind_direction(315) == "NW"
    assert weather_tool.describe_wind_direction(360) == "N"  # wraps via %16


# --- _num / _local_time ------------------------------------------------------


def test_num_drops_trailing_zero_and_keeps_one_decimal():
    assert weather_tool._num(0.0) == "0"
    assert weather_tool._num(5.0) == "5"
    assert weather_tool._num(4.2) == "4.2"
    assert weather_tool._num(2.34) == "2.3"  # rounds to one decimal


def test_local_time_extracts_hh_mm_and_passes_through_plain():
    assert weather_tool._local_time("2026-06-22T05:48") == "05:48"
    # Truncated to HH:MM even when seconds trail (pins the slice end, not just its start).
    assert weather_tool._local_time("2026-06-22T05:48:30") == "05:48"
    assert weather_tool._local_time("") == ""  # no "T": returned unchanged


# --- _geocode ----------------------------------------------------------------


def test_geocode_returns_top_match_and_hits_geocoding_host():
    seen = {}

    def fetch(url: str) -> object:
        seen["url"] = url
        return _GEOCODE

    result = weather_tool._geocode("Paris", fetch=fetch)
    assert result == ("Paris", "France", 48.85, 2.35)
    assert "geocoding-api.open-meteo.com" in seen["url"]
    assert "name=Paris" in seen["url"]
    assert "count=1" in seen["url"]


def test_geocode_without_country_yields_none_country():
    def fetch(url: str) -> object:
        return {"results": [{"name": "Atlantis", "latitude": 0.0, "longitude": 0.0}]}

    assert weather_tool._geocode("Atlantis", fetch=fetch) == ("Atlantis", None, 0.0, 0.0)


def test_geocode_no_results_is_none():
    assert weather_tool._geocode("Nowhereville", fetch=lambda url: {"results": []}) is None


def test_geocode_missing_results_key_is_none():
    assert weather_tool._geocode("x", fetch=lambda url: {}) is None


# --- _forecast ---------------------------------------------------------------


def test_forecast_requests_full_current_and_daily_series_for_coordinates():
    seen = {}

    def fetch(url: str) -> object:
        seen["url"] = url
        return _FORECAST

    data = weather_tool._forecast(48.85, 2.35, fetch=fetch)
    assert data == _FORECAST

    parts = urlsplit(seen["url"])
    assert parts.path.endswith("/v1/forecast")
    qs = {key: value[0] for key, value in parse_qs(parts.query).items()}
    assert qs["latitude"] == "48.85"
    assert qs["longitude"] == "2.35"
    assert qs["forecast_days"] == "7"
    assert qs["timezone"] == "auto"
    # The full field sets are pinned literally (not against the module constants) so a
    # mutated/dropped/reordered variable fails here rather than sailing through.
    assert qs["current"] == (
        "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
        "weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
        "pressure_msl,is_day"
    )
    assert qs["daily"] == (
        "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,"
        "wind_direction_10m_dominant,uv_index_max,sunrise,sunset"
    )


# --- format_report -----------------------------------------------------------


def test_format_report_renders_full_current_and_every_forecast_day():
    assert weather_tool.format_report("Paris", "France", _FORECAST) == _EXPECTED_REPORT


def test_format_report_without_country_and_at_night():
    # country=None drops the ", <country>" suffix; an absent is_day reads as nighttime.
    data: dict[str, object] = {
        "current": {"temperature_2m": 10.0, "weather_code": 0},
        "daily": _FORECAST["daily"],
    }
    report = weather_tool.format_report("Testville", None, data)
    assert report.startswith("Current conditions in Testville: 10°C (50°F),")
    assert "nighttime." in report
    assert ", None" not in report  # the country suffix is omitted, not rendered as None


# --- build_weather_tool (end to end via the seam) ----------------------------


def test_tool_name_and_happy_path():
    tool = weather_tool.build_weather_tool(fetch=_fake_fetch())
    assert tool.name == weather_tool.WEATHER_TOOL_NAME == "get_weather"
    assert tool.invoke({"location": "Paris"}) == _EXPECTED_REPORT


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


# --- _daily_rows ragged-array guard -----------------------------------------


def test_format_report_skips_trailing_days_when_a_required_array_is_short():
    # weather_code shorter than the temp arrays: the shortest *required* array bounds the
    # row count, so only today renders. Kills the min()->max() and the count mutations.
    data: dict[str, object] = {
        "current": {"temperature_2m": 10.0, "weather_code": 0, "is_day": 1},
        "daily": {
            "time": ["2026-06-22", "2026-06-23", "2026-06-24"],
            "temperature_2m_max": [12.0, 13.0, 14.0],
            "temperature_2m_min": [5.0, 6.0, 7.0],
            "weather_code": [0],  # only today's code present
        },
    }
    report = weather_tool.format_report("Testville", None, data)
    assert "Today (2026-06-22):" in report
    assert "Tomorrow" not in report
    assert "2026-06-24:" not in report


# --- _WMO_DESCRIPTIONS table pin --------------------------------------------


def test_wmo_descriptions_table_is_exact():
    # Pin the whole code->phrase table: a mutated integer key makes the dict differ
    # from this literal, failing the test. (The table is only import-time evaluated,
    # so the mutation gate reruns the full suite and relies on this test to kill it.)
    assert weather_tool._WMO_DESCRIPTIONS == {
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


def test_compass_table_is_exact():
    # The 16-point table backs describe_wind_direction; pin it so a reordered/renamed
    # point is caught even where the bearing tests don't sample that index.
    assert weather_tool._COMPASS == (
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


def test_get_json_fetches_and_parses_via_httpx(monkeypatch):
    # Exercises the default network seam (httpx GET -> raise_for_status -> json), mocking
    # httpx2 so no socket opens. Asserts the URL/timeout passthrough and that the response is
    # status-checked, so the mutation gate can't drop any of those lines silently.
    import httpx2 as httpx

    calls: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            calls["raised"] = True

        def json(self) -> object:
            return {"ok": True}

    def fake_get(url: str, timeout: float) -> _Resp:
        calls["url"] = url
        calls["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)

    assert weather_tool._get_json("https://example.test/x") == {"ok": True}
    assert calls["url"] == "https://example.test/x"
    assert calls["timeout"] == weather_tool._TIMEOUT
    assert calls["raised"] is True
