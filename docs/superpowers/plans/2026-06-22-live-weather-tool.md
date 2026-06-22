# Live Weather Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `assembly live` voice agent a keyless, always-available weather tool (current conditions + short forecast) backed by Open-Meteo.

**Architecture:** A new live-only module `aai_cli/agent_cascade/weather_tool.py` following the `fetch_tool.py` shape — pure, directly-tested helpers (WMO-code text, formatting) plus a single injectable network seam (`Fetcher = Callable[[str], object]`). The tool geocodes a place name, fetches a 3-day forecast, and renders a short speakable string. `brain.py` binds it into the deepagents graph, advertises it in the system prompt, and gives it a live-UI affordance label.

**Tech Stack:** Python 3.12+, `httpx` (already a dependency, used by `fetch_tool.py`), LangChain `@tool`, `aai_cli.core.jsonshape` for safe JSON parsing, pytest.

## Global Constraints

Copied verbatim from the repo invariants — every task's requirements include these:

- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- mypy is **strict** on `aai_cli` (`disallow_untyped_defs`); no new `# type: ignore` / `Any` / `cast(` — the gate count-checks these against the merge-base. Parse untyped JSON through `aai_cli.core.jsonshape` (`as_mapping`, `mapping_list`, `object_list`, `as_int`, `as_float`) rather than indexing raw `object`.
- The tool is **keyless** — it must **not** read any env var (no `aai_cli.core.env` use, no API key); Open-Meteo needs none.
- Tests are **hermetic**: pytest-socket is armed (`--disable-socket`). Never make a real HTTP call in a test — always drive the tool through the injected `fetch` seam.
- The gate requires **100% patch coverage** vs `origin/main` **and** survives **diff-scoped mutation testing**: every changed line needs an assertion that would *fail* if the line broke (assert behavior, not just execution). A pure tuning constant that can't be asserted gets `# pragma: no mutate`.
- Docstrings on public functions are imperative, sentence-case; internal helper docstrings keep normal punctuation. Run-logic shared beyond one command would live in `app/`, but this is live-only, so it stays in `agent_cascade/`.
- Files stay under 500 lines; cyclomatic complexity max B per function.
- Do not edit `uv.lock` — no new dependencies (`httpx` is already present).

---

### Task 1: The `weather_tool.py` module

**Files:**
- Create: `aai_cli/agent_cascade/weather_tool.py`
- Test: `tests/test_agent_cascade_weather.py`

**Interfaces:**
- Consumes: `aai_cli.core.jsonshape` (`as_mapping`, `mapping_list`, `object_list`, `as_int`, `as_float`); `httpx` (in the default seam only); `langchain_core.tools.tool` / `BaseTool`.
- Produces (later tasks rely on these exact names/types):
  - `WEATHER_TOOL_NAME: str = "get_weather"`
  - `Fetcher = Callable[[str], object]`
  - `describe_weather_code(code: int) -> str`
  - `format_report(name: str, data: dict[str, object]) -> str`
  - `_geocode(name: str, *, fetch: Fetcher) -> tuple[str, float, float] | None`
  - `_forecast(lat: float, lon: float, *, fetch: Fetcher) -> dict[str, object]`
  - `build_weather_tool(fetch: Fetcher = _get_json) -> BaseTool` — exposes `get_weather(location: str) -> str`

- [ ] **Step 1: Write the failing test file (all behaviors at once)**

Create `tests/test_agent_cascade_weather.py`:

```python
"""Tests for the keyless Open-Meteo weather tool behind `assembly live`.

The tool's only network seam is the injected ``fetch`` callable, so the whole
geocode -> forecast -> format flow runs with no sockets (pytest-socket stays armed).
"""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade import weather_tool

# Canned Open-Meteo payloads keyed by URL prefix, replayed through the fetch seam.
_GEOCODE = {
    "results": [
        {"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "France"}
    ]
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_agent_cascade_weather.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.agent_cascade.weather_tool'`.

- [ ] **Step 3: Write the module**

Create `aai_cli/agent_cascade/weather_tool.py`:

```python
"""A keyless live-weather tool for the `assembly live` voice agent.

Backed by Open-Meteo, which needs no API key — so unlike the optional Firecrawl
search, this tool is *always* present, giving every live session at least one real
capability. The flow is geocode (place name -> coordinates) -> forecast (current +
a short daily outlook) -> a single short string the agent reads aloud.

The only network seam is :data:`Fetcher` (a ``url -> parsed JSON`` callable),
injected in tests so the whole flow runs with no sockets — the same shape
``code_agent.fetch_tool`` uses. Everything else (the WMO-code text, the spoken
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_weather.py -q`
Expected: PASS (all 11 tests).

Note on the °F assertion: 14.3 °C → `round(14.3*9/5+32)` = `round(57.74)` = **58**, matching the test.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/weather_tool.py tests/test_agent_cascade_weather.py
git commit -m "feat: keyless Open-Meteo weather tool for assembly live"
```

---

### Task 2: Wire the weather tool into the live agent's brain

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (imports near line 26; `_TOOL_LABELS` ~line 47; `_tool_capabilities` ~line 83; `build_live_tools` ~line 137)
- Test: `tests/test_agent_cascade_brain.py` (add new tests; update the two existing `build_live_tools` tests at lines 378-388)

**Interfaces:**
- Consumes (from Task 1): `weather_tool.WEATHER_TOOL_NAME`, `weather_tool.build_weather_tool`.
- Produces: `build_live_tools()` now always includes the weather tool; `_tool_capabilities()` advertises weather; `_tool_label(WEATHER_TOOL_NAME)` → `"Checking the weather"`.

- [ ] **Step 1: Update the two existing `build_live_tools` tests to expect the always-present weather tool**

In `tests/test_agent_cascade_brain.py`, the current tests (lines 378-388) assert the toolset is *only* web search / empty. Weather is now always present, so replace both:

```python
def test_build_live_tools_has_weather_and_web_search_when_keyed(monkeypatch):
    search = _NamedTool(brain.WEB_SEARCH_TOOL_NAME)
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: search)
    names = [tool.name for tool in brain.build_live_tools()]
    # Web search is the optional keyed leg; the keyless weather tool is always present.
    assert brain.WEB_SEARCH_TOOL_NAME in names
    assert weather_tool.WEATHER_TOOL_NAME in names


def test_build_live_tools_is_just_weather_without_firecrawl_key(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: None)
    # No FIRECRAWL_API_KEY -> no web search, but the keyless weather tool still loads.
    names = [tool.name for tool in brain.build_live_tools()]
    assert names == [weather_tool.WEATHER_TOOL_NAME]
```

Add the import at the top of the test module (near the other `from aai_cli.agent_cascade import …` imports, ~line 18):

```python
from aai_cli.agent_cascade import weather_tool
```

Note: `_NamedTool(name)` already exists in this file (it exposes a `.name`); using it for the search double keeps `[tool.name for tool …]` working without a real Firecrawl object.

- [ ] **Step 2: Add new tests for the weather wiring (capability phrase + affordance label)**

Append to `tests/test_agent_cascade_brain.py`:

```python
def test_weather_tool_advertised_in_system_prompt():
    prompt = brain.build_system_prompt(
        "persona", tools=[_NamedTool(weather_tool.WEATHER_TOOL_NAME)]
    )
    assert "current weather and short forecast" in prompt
    # And it isn't the no-tools fallback.
    assert "no external tools" not in prompt


def test_tool_label_maps_weather():
    assert brain._tool_label(weather_tool.WEATHER_TOOL_NAME) == "Checking the weather"
```

- [ ] **Step 3: Run the new/updated tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q -k "weather or build_live_tools"`
Expected: FAIL — `build_live_tools` still returns only web search (no weather), the capability phrase isn't in the prompt, and `_tool_label` returns `"Using get_weather"`.

- [ ] **Step 4: Implement the wiring in `brain.py`**

Add the import beside the existing `firecrawl_search` import (line 26):

```python
from aai_cli.agent_cascade import weather_tool
```

Add the affordance label to `_TOOL_LABELS` (line 47) — keep web search, add weather:

```python
_TOOL_LABELS = {
    WEB_SEARCH_TOOL_NAME: "Searching the web",
    weather_tool.WEATHER_TOOL_NAME: "Checking the weather",
}
```

Extend `_tool_capabilities` (lines 83-94) to advertise weather when the tool is bound:

```python
def _tool_capabilities(tools: Sequence[BaseTool]) -> list[str]:
    """The spoken-capability phrases backed by present built-in tools.

    The live agent's built-in legs are the keyless Open-Meteo weather tool (always
    present) and Firecrawl web search (only when ``FIRECRAWL_API_KEY`` is set) — so the
    prompt advertises each only when the agent can really do it. Advertising a missing
    tool made it announce an action ("I'll search…") it then couldn't take.
    """
    names = {tool.name for tool in tools}
    capabilities: list[str] = []
    if WEB_SEARCH_TOOL_NAME in names:
        capabilities.append("search the web for current or unfamiliar facts")
    if weather_tool.WEATHER_TOOL_NAME in names:
        capabilities.append("tell someone the current weather and short forecast for a place")
    return capabilities
```

Update `build_live_tools` (lines 137-151) to always include the weather tool:

```python
def build_live_tools() -> list[BaseTool]:
    """The live agent's built-in tools: the keyless weather tool, plus Firecrawl web
    search when ``FIRECRAWL_API_KEY`` is set.

    Deliberately minimal. A low-latency spoken turn does best with a few obvious tools
    rather than a large menu it must choose among. Open-Meteo needs no key, so the
    weather tool is always present (every session has at least one real capability);
    web search is reused (un-approval-gated) from the coding agent and added only when
    keyed. Extra tools remain strictly opt-in via ``--mcp-config``.
    """
    from aai_cli.agent_cascade.weather_tool import build_weather_tool
    from aai_cli.code_agent.firecrawl_search import build_web_search_tool

    tools: list[BaseTool] = [build_weather_tool()]
    search = build_web_search_tool()
    if search is not None:
        tools.append(search)
    return tools
```

Note: keep the module-level `from aai_cli.agent_cascade import weather_tool` (used by `_TOOL_LABELS` and `_tool_capabilities` at import time); the function-local `build_weather_tool` import mirrors the existing lazy `build_web_search_tool` import so module import stays light.

- [ ] **Step 5: Run the brain tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: PASS (the two updated `build_live_tools` tests, the two new weather tests, and all pre-existing tests — e.g. `test_build_system_prompt_*`, `test_tool_label_maps_web_search_and_falls_back_for_others` — still green).

- [ ] **Step 6: Check for snapshot/help drift**

`build_live_tools` and the system prompt are not part of any `--help` golden or TUI SVG (the prompt is internal, the tool list isn't rendered), so no snapshot regeneration is expected. Confirm:

Run: `uv run pytest tests/test_snapshots_help_root.py tests/test_agent_cascade_command.py -q`
Expected: PASS with no snapshot changes. (If anything fails on a golden, regenerate with `uv run pytest --snapshot-update` and eyeball the diff before committing — but none is anticipated.)

- [ ] **Step 7: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
git commit -m "feat: bind the live weather tool into the assembly live agent"
```

---

### Task 3: Full gate + documentation consistency

**Files:**
- Possibly modify: `REFERENCE.md` / `README.md` only if they enumerate the live agent's built-in tools (the docs-consistency gate checks env vars / exit codes / command refs, not tool lists — verify before editing).

- [ ] **Step 1: Check whether docs enumerate the live tools**

Run: `git grep -n -i "web search\|firecrawl\|built-in tool" REFERENCE.md README.md`
If a passage lists the live agent's built-in tools (e.g. "the live agent can search the web"), add weather alongside it in the same terse style. If nothing enumerates them, skip — do not invent a section (YAGNI).

- [ ] **Step 2: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` This is the authoritative gate — it runs ruff/mypy/pyright/vulture/import-linter, the 100%-patch-coverage gate, and the diff-scoped mutation gate against `origin/main`. Do not claim done until it prints that line.

Likely failure points and fixes:
- **Mutation survivor on `_TIMEOUT` / `_FORECAST_DAYS`** — `_TIMEOUT` is already `# pragma: no mutate`. If `_FORECAST_DAYS` survives (a `±1` mutant the tests don't kill), assert it via the `forecast_days=3` URL substring (already in `test_forecast_requests_current_and_daily_for_coordinates`) — that pins it.
- **Mutation survivor on a `_WMO_DESCRIPTIONS` entry** — the dict literal lines aren't individually asserted. The two codes the tests check (0, 61) and the fallback are killed; other entries are pure data. If the gate flags an unasserted entry as a changed line, either assert that code in `test_describe_weather_code_known` or accept it's data (the gate only mutates *changed* lines, so add the assertion rather than a pragma on data).
- **Patch-coverage gap** — if `diff-cover` reports an uncovered line, it is almost certainly a branch in `_geocode`/`format_report` (e.g. the `payload is not None` guard or the `or {}` fallback). Add a test feeding a wrong-shaped payload (e.g. `fetch=lambda url: []`) to cover it.

- [ ] **Step 3: Re-run targeted gates if either tail gate failed**

```bash
uv run pytest -q -n auto --cov=aai_cli --cov-branch --cov-context=test --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=100
uv run python scripts/mutation_gate.py origin/main
```

- [ ] **Step 4: Final commit (only if Step 1 changed docs)**

```bash
git add REFERENCE.md README.md
git commit -m "docs: note the live weather tool"
```

If no docs changed, Tasks 1 and 2 are already committed and the feature is complete.

---

## Self-Review

**Spec coverage:**
- Keyless Open-Meteo source → Task 1 (`_get_json`, no env read). ✓
- Geocode → forecast → format flow → Task 1 (`_geocode`/`_forecast`/`format_report`). ✓
- Current conditions + short forecast (today + 2 days) → `_FORECAST_DAYS = 3`, `_DAY_LABELS`. ✓
- Both °C and °F for current → `format_report` + `_c_to_f`; °C-only outlook → `_forecast_lines`. ✓
- WMO-code text with unknown fallback → `describe_weather_code`. ✓
- Live-only placement in `agent_cascade/` → Task 1 file path. ✓
- Always present, web search stays keyed → Task 2 `build_live_tools`. ✓
- System-prompt capability phrase → Task 2 `_tool_capabilities`. ✓
- Live-UI affordance label → Task 2 `_TOOL_LABELS` / `_tool_label`. ✓
- Best-effort error handling (not-found + network) → Task 1 `get_weather` try/except + tests. ✓
- Hermetic tests via injected `fetch` → Task 1 `_fake_fetch`, no sockets. ✓
- Out of scope (no units flag, no opt-out flag, top-match only) → respected; nothing added. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `Fetcher = Callable[[str], object]`, `_geocode -> tuple[str, float, float] | None`, `_forecast -> dict[str, object]`, `format_report(name, data: dict[str, object])`, `WEATHER_TOOL_NAME = "get_weather"`, `build_weather_tool(fetch=_get_json) -> BaseTool` — used identically in Tasks 1, 2, and the tests. ✓
