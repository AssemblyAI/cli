# Live weather tool for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Give the `assembly live` voice agent (the `agent-cascade` command) a keyless,
always-available weather tool backed by [Open-Meteo](https://open-meteo.com/).
It returns the current conditions plus a short forecast for a named place, in a
form short enough to read aloud — bringing the live agent closer to the
"talk to a multimodal assistant" experience without any API-key setup.

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Its only built-in tool today is Firecrawl
web search, which is bound only when `FIRECRAWL_API_KEY` is set — so an unkeyed
session runs tool-free. Open-Meteo needs no key, so the weather tool is the
first built-in tool that is **always** present, giving every live session at
least one real capability.

Tools are LangChain `BaseTool`s. The established pattern for a custom (non-vendor)
tool is `aai_cli/code_agent/fetch_tool.py`: pure, directly-testable helpers plus a
single thin network seam (a `Callable`) that is injected in tests so the suite
needs no sockets.

## Scope

- **Live-only.** The tool lives in `aai_cli/agent_cascade/` and is bound only in
  the live voice agent. The coding agent's toolset is unchanged.
- **Data source: Open-Meteo (keyless).** Free, no signup, with a companion
  geocoding endpoint to turn a place name into coordinates.
- **Coverage: full current conditions + a multi-day forecast** (today + the next
  six days). The report surfaces every field an LLM might draw on to answer a
  follow-up — temperature and feels-like, humidity, precipitation (amount *and*
  probability), wind speed / direction / gusts, cloud cover, pressure, day/night,
  the UV index, and sunrise / sunset — rather than a single spoken sentence. The
  agent reads aloud only the slice the conversation calls for.

### Out of scope (YAGNI)

- No units-configuration flag (temperatures are returned in **both** °C and °F;
  the agent speaks whichever fits the conversation).
- No `--no-weather` opt-out flag (the tool is read-only and cheap; web search's
  only "gate" is its key requirement, which does not apply here).
- No geocoding disambiguation UI — always take the top match.

## Architecture

A new module `aai_cli/agent_cascade/weather_tool.py`, beside `mcp_tools.py`.

```
get_weather(location)  ──▶  _geocode(location)     ──▶ Open-Meteo geocoding API
                       └──▶  _forecast(lat, lon)    ──▶ Open-Meteo forecast API
                       └──▶  format_report(...)     ──▶ speakable string
```

### Components

- `WEATHER_TOOL_NAME = "get_weather"` — the registered tool name. `brain.py`
  detects weather availability and labels the tool affordance by this name, so a
  test pins it.
- `Fetcher = Callable[[str], object]` — GETs a URL and returns parsed JSON. The
  default `_get_json` uses `httpx`; tests inject a fake mapping URLs → canned
  JSON. **This is the only network seam.**
- `_geocode(name, *, fetch)` → resolved display name + country + latitude/longitude,
  or `None` when there is no match. Endpoint:
  `https://geocoding-api.open-meteo.com/v1/search?name=<name>&count=1`.
- `_forecast(lat, lon, *, fetch)` → the full `current` and `daily` blocks. Endpoint:
  `https://api.open-meteo.com/v1/forecast` with the full current series
  (`temperature_2m`, `apparent_temperature`, `relative_humidity_2m`,
  `precipitation`, `weather_code`, `cloud_cover`, `wind_speed_10m`,
  `wind_direction_10m`, `wind_gusts_10m`, `pressure_msl`, `is_day`) and daily series
  (`weather_code`, `temperature_2m_max/min`, `precipitation_sum`,
  `precipitation_probability_max`, `wind_speed_10m_max`, `wind_gusts_10m_max`,
  `wind_direction_10m_dominant`, `uv_index_max`, `sunrise`, `sunset`),
  `forecast_days=7`, `timezone=auto`. Temperatures in Celsius (°F derived in formatting).
- `describe_weather_code(code)` — pure WMO weather-code → human text
  ("partly cloudy", "light rain", …) with a fallback for an unknown code.
- `describe_wind_direction(degrees)` — pure bearing → 16-point compass name.
- `format_report(name, country, data)` — pure → a comprehensive, labelled
  multi-line report: a current-conditions block (both units for temperature/
  feels-like, plus humidity, cloud cover, precipitation, wind, pressure, day/night)
  followed by one line per forecast day (high/low in both units, condition,
  precipitation amount + probability, wind + gusts + direction, UV index, sunrise/
  sunset). °F is computed as `round(c * 9 / 5 + 32)`.
- `build_weather_tool(fetch=_get_json)` — the `@tool(WEATHER_TOOL_NAME)` wrapper
  exposing `get_weather(location: str) -> str`. The `fetch` seam is injectable
  for hermetic tests. Plus `WEATHER_TOOL_NAME`, these are the module's only
  public names.

### Data flow per call

1. The model calls `get_weather` with a location string.
2. `_geocode` resolves it to coordinates + a clean display name (or `None`).
3. `_forecast` fetches current + 3-day daily data for those coordinates.
4. `format_report` renders a short, speakable summary in both units.

## Wiring into `brain.py`

- `build_live_tools()` appends the weather tool. Because Open-Meteo is keyless,
  the weather tool is always present even when web search is absent — so the
  live session always has at least one tool.
- `_tool_capabilities()` detects `WEATHER_TOOL_NAME` and adds the phrase
  *"tell you the current weather and short forecast for a place"* to the spoken
  capability clause.
- `_TOOL_LABELS[WEATHER_TOOL_NAME] = "Checking the weather"` so the live UI shows
  a meaningful affordance while the tool runs (matching `"Searching the web"`).

The existing `_NO_TOOLS_GUIDANCE` path still works: it is reached only when
`build_system_prompt` is handed an explicitly empty toolset (which tests do),
not in a normal live session.

## Error handling

The tool is best-effort and **never raises** out to the graph — a weather
outage must not trip `brain`'s "the agent couldn't complete the turn" path or
sink a live demo. `get_weather` catches its own failures and returns a short
speakable string instead:

- No geocoding match → *"I couldn't find a place called '<location>'."*
- Network/HTTP error (the `fetch` seam raises) → *"I couldn't get the weather
  right now."*

## Testing

Targets the gate's 100% patch-coverage + diff-scoped mutation requirements:
assertions must *fail* if a changed line breaks, not merely execute it.

- Pure helpers tested directly:
  - URL building for `_geocode` and `_forecast` (params present and correct).
  - `describe_weather_code` for several known codes **and** the unknown-code
    fallback.
  - `format_report` — both-unit rendering, the current line, and the
    forecast-day lines.
- The tool driven end-to-end with a fake `fetch`:
  - Happy path: canned geocode + forecast JSON → expected speakable string.
  - Not-found path: empty geocoding results → the "couldn't find" message.
  - Network-error path: `fetch` raises → the "couldn't get the weather" message.
- `brain` wiring:
  - `build_live_tools()` includes a tool named `WEATHER_TOOL_NAME`.
  - `_tool_capabilities()` (or `build_system_prompt`) advertises weather.
  - `_TOOL_LABELS` / `_tool_label` returns "Checking the weather".

All tests are hermetic — no real network — via the injected `fetch` seam, in
keeping with the rest of the cascade's STT/LLM/TTS fakes.
