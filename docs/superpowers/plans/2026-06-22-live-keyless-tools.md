# Eight Keyless Tools for `assembly live` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eight always-on, keyless tools to the `assembly live` voice agent (`agent-cascade`): `look_up_topic`, `calculate`, `convert_units`, `define_word`, `get_time_in`, `date_math`, `check_holiday`, and `sun_times`.

**Architecture:** Each tool is a self-contained module in `aai_cli/agent_cascade/`, mirroring the existing `weather_tool.py`/`datetime_tool.py` pattern: a `*_TOOL_NAME` constant, a `build_*_tool(seam=default)` factory returning a LangChain `BaseTool`, at most one injected seam (`Fetcher`/`Clock`) for hermetic tests, and a body that **never raises** — failures return a short spoken apology. A new shared `geocode.py` factors out the Open-Meteo geocoding used by `get_weather`, `get_time_in`, and `sun_times`. `brain.py` binds all eight (no key gate) and advertises each.

**Tech Stack:** Python 3.12–3.13, LangChain `BaseTool`, deepagents graph, `httpx`/`httpx2`, and five new libraries — `pint`, `simpleeval`, `python-dateutil`, `holidays`, `astral`.

**Spec:** `docs/superpowers/specs/2026-06-22-live-keyless-tools-design.md`

## Global Constraints

- **`from __future__ import annotations`** at the top of every module; modern typing (`X | None`).
- **Never raise out of a tool body.** Catch failures and return a short, speakable apology string (the `weather_tool` rule). Use a broad `except Exception:` for network/library calls, exactly as `weather_tool.get_weather` does.
- **One seam per tool, injected for tests.** `Fetcher = Callable[[str], object]` (parsed-JSON GET) and/or `Clock = Callable[[], datetime]`. The default fetcher uses `httpx.get(url, timeout=…).raise_for_status()` then `.json()`. No test touches the real network/clock.
- **Lazy imports for the new libraries** (`pint`, `simpleeval`, `dateutil`, `holidays`, `astral`) and for `langchain_core.tools` — import inside the factory/body, never at module top, to keep CLI startup fast (`webpage_tool` is the precedent).
- **Tuning knobs get `# pragma: no mutate`** (e.g. text-truncation caps) — the mutation gate can't kill an order-equivalent ±1 change.
- **Tests must assert behavior, not just execute it** (diff-scoped mutation gate + 100% patch coverage vs `origin/main`). Assert the exact returned string/branch.
- **Help/docstring copy:** tool docstrings are the model-facing usage contract — write them as full sentences (they are NOT the period-less CLI help copy, and are not snapshot-pinned).
- **Dependency floors:** pin each new dependency's floor to the **second-newest** release (safe-chain's minimum-package-age check rejects the newest at resolution).
- **Commit discipline:** a PreToolUse hook blocks `git commit` unless `./scripts/check.sh` last passed for the current tree. Per-task commits in this plan are WIP commits — prefix them `AAI_ALLOW_COMMIT=1`. Each PR ends with a **gate task** that runs `./scripts/check.sh` to completion (`All checks passed.`) before the real commit / PR.
- **Two PRs:** Task 1 is **PR-A** (dependencies only). Tasks 2–12 are **PR-B** (the feature), which lands after PR-A.

---

## PR-A — Dependencies

### Task 1: Add the five libraries

**Files:**
- Modify: `pyproject.toml` (the `dependencies = [` array, around line 27)
- Modify: `uv.lock` (regenerated, not hand-edited)
- Test: `tests/test_live_tool_deps.py` (Create)

**Interfaces:**
- Produces: importable `pint`, `simpleeval`, `dateutil`, `holidays`, `astral` for Tasks 4, 5, 8, 9, 10.

- [ ] **Step 1: Find the second-newest released version of each library**

Run (network permitting; otherwise consult PyPI):
```bash
for p in pint simpleeval python-dateutil holidays astral; do
  echo "== $p =="; uv pip index versions "$p" 2>/dev/null | head -3
done
```
For each, choose the **second-newest** version as the floor (safe-chain rejects the newest). Record the chosen floors.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_live_tool_deps.py
"""The five libraries the new live tools need must be importable."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["pint", "simpleeval", "dateutil", "holidays", "astral"],
)
def test_live_tool_dependency_importable(module: str) -> None:
    assert importlib.import_module(module) is not None
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `uv run pytest tests/test_live_tool_deps.py -q`
Expected: FAIL — `ModuleNotFoundError` for at least `pint`/`simpleeval`/`holidays`/`astral`.

- [ ] **Step 4: Add the dependencies**

In `pyproject.toml`, inside the `dependencies = [` array, add (substituting the floors chosen in Step 1):
```toml
    "pint>=X.Y",          # physical-unit conversion for the live convert_units tool
    "simpleeval>=X.Y",    # safe arithmetic-expression eval for the live calculate tool
    "python-dateutil>=X.Y",  # date arithmetic for the live date_math tool (declared directly; deptry forbids using it only transitively)
    "holidays>=X.Y",      # offline public-holiday data for the live check_holiday tool
    "astral>=X.Y",        # sunrise/sunset + moon phase for the live sun_times tool
```

- [ ] **Step 5: Regenerate the lock**

Run: `uv lock`
Expected: `uv.lock` updates; `uv lock --check` then passes.

- [ ] **Step 6: Run the test to confirm it passes**

Run: `uv run pytest tests/test_live_tool_deps.py -q`
Expected: PASS (5 parametrized cases).

- [ ] **Step 7: Run the full gate, then commit (this is its own PR)**

Run: `./scripts/check.sh`
Expected: `All checks passed.`
```bash
git add pyproject.toml uv.lock tests/test_live_tool_deps.py
git commit -m "build: add pint, simpleeval, python-dateutil, holidays, astral for live tools"
```
Open PR-A. Tasks 2–12 land in PR-B on top of it.

---

## PR-B — The eight tools

### Task 2: Shared `geocode.py` + refactor `weather_tool`

**Files:**
- Create: `aai_cli/agent_cascade/geocode.py`
- Modify: `aai_cli/agent_cascade/weather_tool.py` (replace its private `_geocode` + `_GEOCODE_URL` with delegation)
- Test: `tests/test_agent_cascade_geocode.py` (Create)

**Interfaces:**
- Produces: `geocode.Fetcher`, `geocode.GeoResult(name, latitude, longitude, timezone)`, `geocode.geocode(name, *, fetch) -> GeoResult | None`. Consumed by Tasks 7 (`get_time_in`) and 10 (`sun_times`), and now by `weather_tool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_geocode.py
from __future__ import annotations

from aai_cli.agent_cascade import geocode


def _fake_fetch(payload: object):
    def fetch(url: str) -> object:
        assert "geocoding-api.open-meteo.com" in url
        assert "name=Tokyo" in url and "count=1" in url
        return payload
    return fetch


def test_geocode_returns_top_match_with_timezone() -> None:
    result = geocode.geocode(
        "Tokyo",
        fetch=_fake_fetch(
            {"results": [{"name": "Tokyo", "latitude": 35.6895, "longitude": 139.69, "timezone": "Asia/Tokyo"}]}
        ),
    )
    assert result == geocode.GeoResult("Tokyo", 35.6895, 139.69, "Asia/Tokyo")


def test_geocode_no_match_returns_none() -> None:
    assert geocode.geocode("Nowheresville", fetch=_fake_fetch({"results": []})) is None
    assert geocode.geocode("Nowheresville", fetch=_fake_fetch({})) is None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_geocode.py -q`
Expected: FAIL — `ModuleNotFoundError: aai_cli.agent_cascade.geocode`.

- [ ] **Step 3: Create `geocode.py`**

```python
# aai_cli/agent_cascade/geocode.py
"""Shared Open-Meteo geocoding for the `assembly live` voice agent's place-aware tools.

The weather, world-clock, and sun-times tools all turn a spoken place name into
coordinates via Open-Meteo's keyless geocoding endpoint; the time/sun tools additionally
need the IANA timezone the same response already carries. This module is the single
geocoding implementation so that logic isn't duplicated across tools.

The only network seam is :data:`Fetcher` (a ``url -> parsed JSON`` callable), injected in
tests so the flow runs with no sockets — the same shape ``weather_tool`` uses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

from aai_cli.core import jsonshape

# A fetcher GETs a URL and returns parsed JSON. Injected in tests (the only net seam).
Fetcher = Callable[[str], object]

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass(frozen=True)
class GeoResult:
    """A resolved place: display name, coordinates, and IANA timezone."""

    name: str
    latitude: float
    longitude: float
    timezone: str


def geocode(name: str, *, fetch: Fetcher) -> GeoResult | None:
    """Resolve a place name to a :class:`GeoResult`, or None when there is no match.

    Asks Open-Meteo's geocoding endpoint for the single best match. No match (an empty or
    absent ``results`` list) returns None so a caller can speak a clear "couldn't find that
    place" instead of guessing.
    """
    query = urlencode({"name": name, "count": 1, "language": "en", "format": "json"})
    payload = jsonshape.as_mapping(fetch(f"{_GEOCODE_URL}?{query}"))
    results = jsonshape.mapping_list(payload.get("results")) if payload is not None else []
    if not results:
        return None
    top = results[0]
    return GeoResult(
        name=str(top.get("name", name)),
        latitude=jsonshape.as_float(top.get("latitude")),
        longitude=jsonshape.as_float(top.get("longitude")),
        timezone=str(top.get("timezone", "")),
    )
```

- [ ] **Step 4: Refactor `weather_tool._geocode` to delegate**

In `aai_cli/agent_cascade/weather_tool.py`, delete the `_GEOCODE_URL` constant and replace the body of `_geocode` so it calls the shared module (keep its 3-tuple return shape so `_forecast`/`get_weather` are unchanged):
```python
from aai_cli.agent_cascade import geocode as _geocode_mod


def _geocode(name: str, *, fetch: Fetcher) -> tuple[str, float, float] | None:
    """Resolve a place name to ``(display name, latitude, longitude)`` via the shared geocoder."""
    result = _geocode_mod.geocode(name, fetch=fetch)
    if result is None:
        return None
    return (result.name, result.latitude, result.longitude)
```
Leave `_FORECAST_URL` and the rest of `weather_tool` intact.

- [ ] **Step 5: Run geocode + weather tests**

Run: `uv run pytest tests/test_agent_cascade_geocode.py tests/test_agent_cascade_weather.py -q`
Expected: PASS. If a pre-existing weather test referenced `weather_tool._GEOCODE_URL`, repoint that assertion at `geocode._GEOCODE_URL` (the URL/params are unchanged, so the fake-fetch happy paths still match).

- [ ] **Step 6: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/geocode.py aai_cli/agent_cascade/weather_tool.py tests/test_agent_cascade_geocode.py
AAI_ALLOW_COMMIT=1 git commit -m "refactor(live): extract shared geocode.py from weather_tool"
```

---

### Task 3: `look_up_topic` (Wikipedia)

**Files:**
- Create: `aai_cli/agent_cascade/topic_tool.py`
- Test: `tests/test_agent_cascade_topic.py` (Create)

**Interfaces:**
- Produces: `topic_tool.LOOKUP_TOOL_NAME = "look_up_topic"`, `topic_tool.build_lookup_tool(fetch=…) -> BaseTool`. Consumed by Task 12 (`brain.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_topic.py
from __future__ import annotations

from aai_cli.agent_cascade.topic_tool import LOOKUP_TOOL_NAME, build_lookup_tool


def _tool(payload: object):
    def fetch(url: str) -> object:
        assert "en.wikipedia.org/api/rest_v1/page/summary/" in url
        return payload
    return build_lookup_tool(fetch=fetch)


def test_lookup_returns_extract() -> None:
    tool = _tool({"type": "standard", "title": "Ada Lovelace", "extract": "Ada Lovelace was a mathematician."})
    assert tool.invoke({"topic": "Ada Lovelace"}) == "Ada Lovelace was a mathematician."


def test_lookup_disambiguation_apology() -> None:
    tool = _tool({"type": "disambiguation", "title": "Mercury", "extract": "Mercury may refer to..."})
    assert "couldn't find a clear summary" in tool.invoke({"topic": "Mercury"})


def test_lookup_empty_extract_apology() -> None:
    tool = _tool({"type": "standard", "extract": ""})
    assert "couldn't find a clear summary" in tool.invoke({"topic": "Zzz"})


def test_lookup_fetch_error_apology() -> None:
    def boom(url: str) -> object:
        raise RuntimeError("502")
    tool = build_lookup_tool(fetch=boom)
    assert tool.invoke({"topic": "Anything"}) == "I couldn't look that up right now."


def test_tool_name() -> None:
    assert build_lookup_tool().name == LOOKUP_TOOL_NAME == "look_up_topic"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_topic.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `topic_tool.py`**

```python
# aai_cli/agent_cascade/topic_tool.py
"""A keyless encyclopedia-lookup tool for the `assembly live` voice agent.

Backed by Wikipedia's REST summary endpoint, which needs no API key and returns a clean,
~1-paragraph ``extract`` purpose-built to be read aloud — so "who is…", "what is…", and
"tell me about…" work without the keyed Firecrawl search. The only network seam is
:data:`Fetcher`, injected in tests. Failures never raise out to the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import quote

from aai_cli.core import jsonshape

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

LOOKUP_TOOL_NAME = "look_up_topic"

# A fetcher GETs a URL and returns parsed JSON. Injected in tests (the only net seam).
Fetcher = Callable[[str], object]

_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Cap the spoken extract; the endpoint is already ~1 paragraph, this just guards an outlier.
_MAX_CHARS = 1200  # pragma: no mutate — a tuning knob; ±a few chars is equivalent


def _get_json(url: str) -> object:
    """GET ``url`` and return its parsed JSON body (the default network seam)."""
    import httpx

    response = httpx.get(url, timeout=15.0)
    response.raise_for_status()
    return response.json()


def build_lookup_tool(fetch: Fetcher = _get_json) -> BaseTool:
    """Wrap the Wikipedia summary lookup as the ``look_up_topic`` tool (``fetch`` injectable)."""
    from langchain_core.tools import tool

    @tool(LOOKUP_TOOL_NAME)
    def look_up_topic(topic: str) -> str:
        """Look up an encyclopedic summary of a person, place, organization, or topic. Use
        for "who is…", "what is…", or "tell me about…" questions about well-known subjects."""
        try:
            payload = jsonshape.as_mapping(fetch(f"{_SUMMARY_URL}{quote(topic)}")) or {}
            extract = str(payload.get("extract", "")).strip()
            if not extract or payload.get("type") == "disambiguation":
                return f"I couldn't find a clear summary for '{topic}'."
            return extract[:_MAX_CHARS]
        except Exception:
            return "I couldn't look that up right now."

    return look_up_topic
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_topic.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/topic_tool.py tests/test_agent_cascade_topic.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add look_up_topic Wikipedia tool"
```

---

### Task 4: `calculate` (simpleeval)

**Files:**
- Create: `aai_cli/agent_cascade/calc_tool.py`
- Test: `tests/test_agent_cascade_calc.py` (Create)

**Interfaces:**
- Produces: `calc_tool.CALC_TOOL_NAME = "calculate"`, `calc_tool.build_calc_tool() -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_calc.py
from __future__ import annotations

import pytest

from aai_cli.agent_cascade.calc_tool import CALC_TOOL_NAME, build_calc_tool


@pytest.fixture
def calc():
    return build_calc_tool()


def test_percentage(calc) -> None:
    assert calc.invoke({"expression": "0.15 * 240"}) == "36"


def test_integer_division_rounds_cleanly(calc) -> None:
    # 87 / 3 is 28.999999999999996 in float; must read as 29, not the artifact.
    assert calc.invoke({"expression": "87 / 3"}) == "29"


def test_precedence_and_unary(calc) -> None:
    assert calc.invoke({"expression": "3 + 4 * 5"}) == "23"
    assert calc.invoke({"expression": "-2 ** 2"}) in {"-4", "4"}  # operator precedence, no crash


def test_non_integer_keeps_decimals(calc) -> None:
    assert calc.invoke({"expression": "10 / 3"}) == "3.3333"


def test_rejects_names(calc) -> None:
    assert calc.invoke({"expression": "foo + 1"}) == "I couldn't compute that."


def test_rejects_function_calls(calc) -> None:
    assert calc.invoke({"expression": "sqrt(4)"}) == "I couldn't compute that."


def test_rejects_syntax_error(calc) -> None:
    assert calc.invoke({"expression": "3 +"}) == "I couldn't compute that."


def test_division_by_zero(calc) -> None:
    assert calc.invoke({"expression": "1 / 0"}) == "I couldn't compute that."


def test_power_bomb_is_capped(calc) -> None:
    # simpleeval's MAX_POWER guard turns this into an error, not a hang.
    assert calc.invoke({"expression": "9 ** 9 ** 9"}) == "I couldn't compute that."


def test_tool_name() -> None:
    assert build_calc_tool().name == CALC_TOOL_NAME == "calculate"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_calc.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `calc_tool.py`**

```python
# aai_cli/agent_cascade/calc_tool.py
"""A pure, offline arithmetic tool for the `assembly live` voice agent.

The LLM does mental math unreliably; this tool evaluates an arithmetic expression exactly.
It uses ``simpleeval`` with no names or functions registered, so only arithmetic over
numeric literals is allowed — ``simpleeval`` itself guards the resource-exhaustion cases
(an exponent bomb via ``MAX_POWER``, oversized strings). There is no network seam: the
only non-determinism would be I/O, and there is none. Failures never raise out to the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

CALC_TOOL_NAME = "calculate"

# Round non-integer results so float artifacts (87/3 -> 28.999999999999996) never leak into
# speech. Asserted by a test (10/3 -> "3.3333"), so this is NOT a free no-mutate knob.
_PRECISION = 4


def _format(value: float) -> str:
    """Render a numeric result for speech: integers bare (``36``), else rounded (``3.3333``).

    ``value`` is whatever ``simpleeval.eval`` returned (typed ``Any`` by the untyped library,
    so no cast is needed); a non-numeric result makes ``round`` raise, caught by the caller.
    """
    number = round(value, _PRECISION)
    if number == int(number):
        return str(int(number))
    return str(number)


def build_calc_tool() -> BaseTool:
    """Wrap a no-names/no-functions ``simpleeval`` evaluator as the ``calculate`` tool."""
    from langchain_core.tools import tool

    @tool(CALC_TOOL_NAME)
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression and return the result. ``expression`` must be a
        plain arithmetic expression using only numbers and the operators + - * / // % ** and
        parentheses — no words, units, or variable names. Translate the spoken question into
        such an expression yourself first. Examples: "15% of 240" -> "0.15 * 240"; "split 87
        three ways" -> "87 / 3"; "3 plus 4 times 5" -> "3 + 4 * 5"."""
        from simpleeval import SimpleEval

        try:
            evaluator = SimpleEval()
            evaluator.names = {}
            evaluator.functions = {}
            return _format(evaluator.eval(expression))
        except Exception:
            return "I couldn't compute that."

    return calculate
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_calc.py -q`
Expected: PASS (10 tests). If `-2 ** 2` surprises you, the assertion accepts both — it only checks no crash.

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/calc_tool.py tests/test_agent_cascade_calc.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add calculate tool via simpleeval"
```

---

### Task 5: `convert_units` (pint + frankfurter.app)

**Files:**
- Create: `aai_cli/agent_cascade/units_tool.py`
- Test: `tests/test_agent_cascade_units.py` (Create)

**Interfaces:**
- Produces: `units_tool.CONVERT_TOOL_NAME = "convert_units"`, `units_tool.build_convert_tool(fetch=…) -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_units.py
from __future__ import annotations

from aai_cli.agent_cascade.units_tool import CONVERT_TOOL_NAME, build_convert_tool


def test_length_conversion_via_pint() -> None:
    out = build_convert_tool().invoke({"value": 5, "from_unit": "mile", "to_unit": "kilometer"})
    assert out.startswith("5") and "8.05" in out and "kilometer" in out


def test_temperature_conversion_via_pint() -> None:
    out = build_convert_tool().invoke({"value": 350, "from_unit": "degF", "to_unit": "degC"})
    assert "176.67" in out


def test_incompatible_units_apology() -> None:
    out = build_convert_tool().invoke({"value": 5, "from_unit": "mile", "to_unit": "kilogram"})
    assert out == "I couldn't convert those units."


def test_currency_conversion_via_fetch() -> None:
    def fetch(url: str) -> object:
        assert "api.frankfurter.app/latest" in url
        assert "from=USD" in url and "to=EUR" in url and "amount=100" in url
        return {"rates": {"EUR": 92.4}}
    out = build_convert_tool(fetch=fetch).invoke({"value": 100, "from_unit": "usd", "to_unit": "eur"})
    assert "100 USD is 92.4 EUR." == out


def test_currency_fetch_error_apology() -> None:
    def boom(url: str) -> object:
        raise RuntimeError("down")
    out = build_convert_tool(fetch=boom).invoke({"value": 100, "from_unit": "USD", "to_unit": "EUR"})
    assert out == "I couldn't get that exchange rate right now."


def test_tool_name() -> None:
    assert build_convert_tool().name == CONVERT_TOOL_NAME == "convert_units"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_units.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `units_tool.py`**

```python
# aai_cli/agent_cascade/units_tool.py
"""A units-and-currency conversion tool for the `assembly live` voice agent.

Physical units convert offline via ``pint``; currencies convert via keyless
frankfurter.app (the only network path, behind the injected :data:`Fetcher`). Path
selection is by ISO-4217 currency code. Failures never raise out to the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from aai_cli.core import jsonshape

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

CONVERT_TOOL_NAME = "convert_units"

Fetcher = Callable[[str], object]

_FRANKFURTER_URL = "https://api.frankfurter.app/latest"
# ISO-4217 codes we treat as currency (vs. a physical unit). A small, common set is enough;
# anything else falls to the pint path, which apologizes if it isn't a real unit.
_CURRENCIES = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "MXN", "BRL", "SEK", "NOK", "NZD", "ZAR"}
)


def _get_json(url: str) -> object:
    """GET ``url`` and return parsed JSON (the default currency-fetch seam)."""
    import httpx

    response = httpx.get(url, timeout=15.0)
    response.raise_for_status()
    return response.json()


def _round(value: float) -> str:
    """Render a converted amount: integers bare, else two decimals (speakable money/units)."""
    rounded = round(value, 2)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def _convert_currency(value: float, src: str, dst: str, *, fetch: Fetcher) -> str:
    """Convert via frankfurter.app; raises on a network error or an absent rate."""
    query = urlencode({"amount": value, "from": src, "to": dst})
    payload = jsonshape.as_mapping(fetch(f"{_FRANKFURTER_URL}?{query}")) or {}
    rates = jsonshape.as_mapping(payload.get("rates")) or {}
    amount = jsonshape.as_float(rates[dst])  # KeyError if missing -> caught by caller
    return f"{_round(value)} {src} is {_round(amount)} {dst}."


def _convert_units(value: float, src: str, dst: str) -> str:
    """Convert physical units via pint; raises pint errors on bad/incompatible units."""
    import pint

    quantity = pint.UnitRegistry().Quantity(value, src).to(dst)
    return f"{_round(value)} {src} is {_round(float(quantity.magnitude))} {dst}."


def build_convert_tool(fetch: Fetcher = _get_json) -> BaseTool:
    """Wrap unit + currency conversion as the ``convert_units`` tool (``fetch`` injectable)."""
    from langchain_core.tools import tool

    @tool(CONVERT_TOOL_NAME)
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert a value between units or currencies. For physical units pass names pint
        understands (e.g. "mile", "kilometer", "kg", "lb", "degF", "degC"); for money pass
        ISO currency codes (e.g. "USD", "EUR"). Examples: 5 mile -> kilometer; 350 degF ->
        degC; 100 USD -> EUR."""
        src, dst = from_unit.upper(), to_unit.upper()
        if src in _CURRENCIES and dst in _CURRENCIES:
            try:
                return _convert_currency(value, src, dst, fetch=fetch)
            except Exception:
                return "I couldn't get that exchange rate right now."
        try:
            return _convert_units(value, from_unit, to_unit)
        except Exception:
            return "I couldn't convert those units."

    return convert_units
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_units.py -q`
Expected: PASS (6 tests). If pint formats `8.046...`, the rounding yields `8.05`.

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/units_tool.py tests/test_agent_cascade_units.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add convert_units tool (pint + frankfurter)"
```

---

### Task 6: `define_word` (dictionaryapi.dev)

**Files:**
- Create: `aai_cli/agent_cascade/define_tool.py`
- Test: `tests/test_agent_cascade_define.py` (Create)

**Interfaces:**
- Produces: `define_tool.DEFINE_TOOL_NAME = "define_word"`, `define_tool.build_define_tool(fetch=…) -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_define.py
from __future__ import annotations

from aai_cli.agent_cascade.define_tool import DEFINE_TOOL_NAME, build_define_tool


def _tool(payload: object):
    def fetch(url: str) -> object:
        assert "api.dictionaryapi.dev/api/v2/entries/en/" in url
        return payload
    return build_define_tool(fetch=fetch)


def test_define_with_synonyms() -> None:
    tool = _tool(
        [{"meanings": [{"partOfSpeech": "adjective",
                        "definitions": [{"definition": "lasting a very short time."}],
                        "synonyms": ["transient", "fleeting", "momentary"]}]}]
    )
    out = tool.invoke({"word": "ephemeral"})
    assert "ephemeral (adjective): lasting a very short time." in out
    assert "Synonyms: transient, fleeting." in out  # capped at two


def test_define_without_synonyms() -> None:
    tool = _tool([{"meanings": [{"partOfSpeech": "noun", "definitions": [{"definition": "a thing."}]}]}])
    out = tool.invoke({"word": "widget"})
    assert out == "widget (noun): a thing."


def test_define_not_found_apology() -> None:
    # The API returns an OBJECT (not a list) when the word is unknown.
    tool = _tool({"title": "No Definitions Found"})
    assert tool.invoke({"word": "asdfghjkl"}) == "I couldn't find a definition for 'asdfghjkl'."


def test_define_fetch_error_apology() -> None:
    def boom(url: str) -> object:
        raise RuntimeError("500")
    assert build_define_tool(fetch=boom).invoke({"word": "x"}) == "I couldn't look up that word right now."


def test_tool_name() -> None:
    assert build_define_tool().name == DEFINE_TOOL_NAME == "define_word"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_define.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `define_tool.py`**

```python
# aai_cli/agent_cascade/define_tool.py
"""A keyless dictionary tool for the `assembly live` voice agent.

Backed by dictionaryapi.dev (no key). On success the endpoint returns a JSON array of
entries; on a miss it returns a JSON object, which we treat as "not found". The only
network seam is :data:`Fetcher`. Failures never raise out to the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import quote

from aai_cli.core import jsonshape

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

DEFINE_TOOL_NAME = "define_word"

Fetcher = Callable[[str], object]

_ENTRIES_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/"
_MAX_SYNONYMS = 2  # keep the spoken reply short


def _get_json(url: str) -> object:
    """GET ``url`` and return parsed JSON (the default network seam)."""
    import httpx

    response = httpx.get(url, timeout=15.0)
    response.raise_for_status()
    return response.json()


def _format(word: str, entries: list[object]) -> str:
    """Render the first meaning as ``word (pos): definition.`` plus up to two synonyms."""
    first = jsonshape.as_mapping(entries[0]) or {}
    meanings = jsonshape.mapping_list(first.get("meanings"))
    meaning = meanings[0]
    pos = str(meaning.get("partOfSpeech", "")).strip()
    definitions = jsonshape.mapping_list(meaning.get("definitions"))
    definition = str(definitions[0].get("definition", "")).strip()
    line = f"{word} ({pos}): {definition}"
    synonyms = [str(s) for s in jsonshape.object_list(meaning.get("synonyms"))][:_MAX_SYNONYMS]
    if synonyms:
        return f"{line} Synonyms: {', '.join(synonyms)}."
    return line


def build_define_tool(fetch: Fetcher = _get_json) -> BaseTool:
    """Wrap the dictionary lookup as the ``define_word`` tool (``fetch`` injectable)."""
    from langchain_core.tools import tool

    @tool(DEFINE_TOOL_NAME)
    def define_word(word: str) -> str:
        """Define an English word and give a couple of synonyms. Use for "define X", "what
        does X mean", or "another word for X"."""
        try:
            payload = fetch(f"{_ENTRIES_URL}{quote(word)}")
            if not isinstance(payload, list) or not payload:
                return f"I couldn't find a definition for '{word}'."
            return _format(word, payload)
        except Exception:
            return "I couldn't look up that word right now."

    return define_word
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_define.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/define_tool.py tests/test_agent_cascade_define.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add define_word dictionary tool"
```

---

### Task 7: `get_time_in` (geocode + zoneinfo)

**Files:**
- Create: `aai_cli/agent_cascade/worldclock_tool.py`
- Test: `tests/test_agent_cascade_worldclock.py` (Create)

**Interfaces:**
- Consumes: `geocode.geocode`, `geocode.GeoResult` (Task 2).
- Produces: `worldclock_tool.WORLDCLOCK_TOOL_NAME = "get_time_in"`, `worldclock_tool.build_worldclock_tool(fetch=…, now=…) -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_worldclock.py
from __future__ import annotations

from datetime import datetime, timezone

from aai_cli.agent_cascade import geocode
from aai_cli.agent_cascade.worldclock_tool import WORLDCLOCK_TOOL_NAME, build_worldclock_tool


def test_time_in_place(monkeypatch) -> None:
    monkeypatch.setattr(
        geocode, "geocode", lambda name, *, fetch: geocode.GeoResult("Tokyo", 35.6, 139.6, "Asia/Tokyo")
    )
    # 2026-06-22 12:00 UTC -> 21:00 JST.
    now = lambda: datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)
    out = build_worldclock_tool(fetch=lambda url: {}, now=now).invoke({"place": "Tokyo"})
    assert out.startswith("In Tokyo it's Monday, June 22, 2026 at 09:00 PM")
    assert "JST" in out


def test_time_in_no_match_apology(monkeypatch) -> None:
    monkeypatch.setattr(geocode, "geocode", lambda name, *, fetch: None)
    out = build_worldclock_tool(fetch=lambda url: {}).invoke({"place": "Nowhere"})
    assert out == "I couldn't find a place called 'Nowhere'."


def test_time_in_bad_timezone_apology(monkeypatch) -> None:
    monkeypatch.setattr(
        geocode, "geocode", lambda name, *, fetch: geocode.GeoResult("X", 0.0, 0.0, "Not/AZone")
    )
    out = build_worldclock_tool(fetch=lambda url: {}).invoke({"place": "X"})
    assert out == "I couldn't get the time there right now."


def test_tool_name() -> None:
    assert build_worldclock_tool().name == WORLDCLOCK_TOOL_NAME == "get_time_in"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_worldclock.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `worldclock_tool.py`**

```python
# aai_cli/agent_cascade/worldclock_tool.py
"""A world-clock tool for the `assembly live` voice agent.

Resolves a spoken place to its IANA timezone via the shared :mod:`geocode` helper, then
renders the current local time there with ``zoneinfo``. Two seams — the geocoder's
:data:`Fetcher` and a :data:`Clock` — are injected in tests. Failures never raise out to
the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from aai_cli.agent_cascade import geocode

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

WORLDCLOCK_TOOL_NAME = "get_time_in"

Clock = Callable[[], datetime]


def _get_json(url: str) -> object:
    """GET ``url`` and return parsed JSON (the default geocode-fetch seam)."""
    import httpx

    response = httpx.get(url, timeout=15.0)
    response.raise_for_status()
    return response.json()


def _now() -> datetime:
    """Return the current instant as a timezone-aware datetime (the default clock)."""
    return datetime.now().astimezone()


def build_worldclock_tool(fetch: geocode.Fetcher = _get_json, now: Clock = _now) -> BaseTool:
    """Wrap the geocode→timezone lookup as the ``get_time_in`` tool (seams injectable)."""
    from langchain_core.tools import tool

    @tool(WORLDCLOCK_TOOL_NAME)
    def get_time_in(place: str) -> str:
        """Get the current local time in a named place (a city or country). Use for "what
        time is it in X" or "is it morning in X"."""
        result = geocode.geocode(place, fetch=fetch)
        if result is None:
            return f"I couldn't find a place called '{place}'."
        try:
            local = now().astimezone(ZoneInfo(result.timezone))
            return f"In {result.name} it's {local.strftime('%A, %B %d, %Y at %I:%M %p %Z')}."
        except Exception:
            return "I couldn't get the time there right now."

    return get_time_in
```

Note: each place-aware tool owns its own `_get_json` default (matching `weather_tool`/`topic_tool`); `geocode.geocode` itself takes `fetch` as a required keyword and has no default fetcher. This keeps `geocode.py` free of a network default and avoids any cross-module private access.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_worldclock.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/worldclock_tool.py aai_cli/agent_cascade/geocode.py tests/test_agent_cascade_worldclock.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add get_time_in world-clock tool"
```

---

### Task 8: `date_math` (python-dateutil)

**Files:**
- Create: `aai_cli/agent_cascade/datemath_tool.py`
- Test: `tests/test_agent_cascade_datemath.py` (Create)

**Interfaces:**
- Produces: `datemath_tool.DATEMATH_TOOL_NAME = "date_math"`, `datemath_tool.build_datemath_tool(now=…) -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_datemath.py
from __future__ import annotations

from datetime import datetime, timezone

from aai_cli.agent_cascade.datemath_tool import DATEMATH_TOOL_NAME, build_datemath_tool

# Pin "today" to 2026-06-22 (a Monday).
_NOW = lambda: datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)


def test_single_future_date() -> None:
    out = build_datemath_tool(now=_NOW).invoke({"date": "2026-07-04"})
    assert out == "July 04, 2026 is a Saturday — 12 days from now."


def test_single_past_date() -> None:
    out = build_datemath_tool(now=_NOW).invoke({"date": "2026-06-20"})
    assert out == "June 20, 2026 is a Saturday — 2 days ago."


def test_single_today() -> None:
    out = build_datemath_tool(now=_NOW).invoke({"date": "2026-06-22"})
    assert out == "June 22, 2026 is a Monday — that's today."


def test_one_day_singular() -> None:
    out = build_datemath_tool(now=_NOW).invoke({"date": "2026-06-23"})
    assert "1 day from now." in out


def test_two_dates_span() -> None:
    out = build_datemath_tool(now=_NOW).invoke({"date": "2026-03-01", "other_date": "2026-08-25"})
    assert out.startswith("There are 177 days between March 01 and August 25, 2026")
    assert "5 months" in out and "3 weeks" in out


def test_bad_date_apology() -> None:
    assert build_datemath_tool(now=_NOW).invoke({"date": "not-a-date"}) == "I couldn't work out those dates."


def test_tool_name() -> None:
    assert build_datemath_tool().name == DATEMATH_TOOL_NAME == "date_math"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_datemath.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `datemath_tool.py`**

```python
# aai_cli/agent_cascade/datemath_tool.py
"""A date-arithmetic tool for the `assembly live` voice agent.

The LLM knows calendar facts but miscounts across them; this tool does the exact day
counting and weekday. The model passes ISO dates it worked out; the tool reports the
weekday + signed distance from today (one date) or the span between two dates (two). The
only seam is a :data:`Clock`. ``python-dateutil`` is imported lazily. Failures never raise.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

DATEMATH_TOOL_NAME = "date_math"

Clock = Callable[[], datetime]


def _now() -> datetime:
    """Return the current instant (the default clock)."""
    return datetime.now().astimezone()


def _days(n: int) -> str:
    """Pluralize a day count: ``1 day`` / ``2 days``."""
    return f"{n} day" if n == 1 else f"{n} days"


def _signed_distance(target: date, today: date) -> str:
    """A spoken distance from today: ``that's today`` / ``N days from now`` / ``N days ago``."""
    delta = (target - today).days
    if delta == 0:
        return "that's today"
    if delta > 0:
        return f"{_days(delta)} from now"
    return f"{_days(-delta)} ago"


def _breakdown(start: date, end: date) -> str:
    """A human span via ``relativedelta``, e.g. ``about 5 months and 3 weeks``."""
    from dateutil.relativedelta import relativedelta

    rel = relativedelta(end, start)
    parts: list[str] = []
    if rel.years:
        parts.append(f"{rel.years} year" + ("" if rel.years == 1 else "s"))
    if rel.months:
        parts.append(f"{rel.months} month" + ("" if rel.months == 1 else "s"))
    weeks, days = divmod(rel.days, 7)
    if weeks:
        parts.append(f"{weeks} week" + ("" if weeks == 1 else "s"))
    if days:
        parts.append(_days(days))
    return "about " + " and ".join(parts) if parts else "the same day"


def _single(target: date, today: date) -> str:
    """One-date report: weekday + signed distance from today."""
    return f"{target.strftime('%B %d, %Y')} is a {target.strftime('%A')} — {_signed_distance(target, today)}."


def _span(d1: date, d2: date) -> str:
    """Two-date report: total days between + a human breakdown."""
    start, end = sorted((d1, d2))
    total = (end - start).days
    return (
        f"There are {total} days between {start.strftime('%B %d')} and "
        f"{end.strftime('%B %d, %Y')} — {_breakdown(start, end)}."
    )


def build_datemath_tool(now: Clock = _now) -> BaseTool:
    """Wrap date arithmetic as the ``date_math`` tool (``now`` injectable)."""
    from langchain_core.tools import tool

    @tool(DATEMATH_TOOL_NAME)
    def date_math(date: str, other_date: str | None = None) -> str:
        """Count days and weekdays between dates. Work out the relevant date(s) yourself and
        pass them as YYYY-MM-DD strings; this tool does the exact counting. Pass one date for
        "what weekday is X" / "how many days until X" (e.g. "2026-12-25"); pass two for "days
        between X and Y"."""
        from dateutil import parser as date_parser

        try:
            today = now().date()
            first = date_parser.isoparse(date).date()
            if other_date is None:
                return _single(first, today)
            return _span(first, date_parser.isoparse(other_date).date())
        except Exception:
            return "I couldn't work out those dates."

    return date_math
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_datemath.py -q`
Expected: PASS (7 tests). If the span breakdown wording differs, adjust the assertion to the real `relativedelta` output (March 1 → August 25 is 5 months, 24 days → "5 months and 3 weeks and 3 days"); update the test to match exactly.

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/datemath_tool.py tests/test_agent_cascade_datemath.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add date_math tool via python-dateutil"
```

---

### Task 9: `check_holiday` (holidays)

**Files:**
- Create: `aai_cli/agent_cascade/holiday_tool.py`
- Test: `tests/test_agent_cascade_holiday.py` (Create)

**Interfaces:**
- Produces: `holiday_tool.HOLIDAY_TOOL_NAME = "check_holiday"`, `holiday_tool.build_holiday_tool(now=…) -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_holiday.py
from __future__ import annotations

from datetime import datetime, timezone

from aai_cli.agent_cascade.holiday_tool import HOLIDAY_TOOL_NAME, build_holiday_tool

_NOW = lambda: datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)


def test_date_is_a_holiday() -> None:
    out = build_holiday_tool(now=_NOW).invoke({"country": "US", "date": "2026-12-25"})
    assert out == "December 25, 2026 is Christmas Day in US."


def test_date_is_not_a_holiday() -> None:
    out = build_holiday_tool(now=_NOW).invoke({"country": "US", "date": "2026-03-03"})
    assert out == "March 03, 2026 is not a public holiday in US."


def test_next_holiday() -> None:
    out = build_holiday_tool(now=_NOW).invoke({"country": "US"})
    assert out == "The next US public holiday is Independence Day on July 04, 2026 — 12 days from now."


def test_unknown_country_apology() -> None:
    out = build_holiday_tool(now=_NOW).invoke({"country": "ZZ"})
    assert out == "I don't have holiday data for that country."


def test_bad_date_apology() -> None:
    out = build_holiday_tool(now=_NOW).invoke({"country": "US", "date": "nope"})
    assert out == "I couldn't work out that date."


def test_tool_name() -> None:
    assert build_holiday_tool().name == HOLIDAY_TOOL_NAME == "check_holiday"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_holiday.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `holiday_tool.py`**

```python
# aai_cli/agent_cascade/holiday_tool.py
"""A public-holiday tool for the `assembly live` voice agent.

Backed by the offline ``holidays`` library (no network). With a date it names the holiday
(or says there isn't one); without a date it reports the next upcoming holiday from today.
The only seam is a :data:`Clock`. ``holidays``/``dateutil`` are imported lazily. Failures
never raise out to the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

HOLIDAY_TOOL_NAME = "check_holiday"

Clock = Callable[[], datetime]


def _now() -> datetime:
    """Return the current instant (the default clock)."""
    return datetime.now().astimezone()


def _days(n: int) -> str:
    """A spoken distance: ``today`` / ``1 day from now`` / ``N days from now``."""
    if n == 0:
        return "today"
    return f"{n} day from now" if n == 1 else f"{n} days from now"


def build_holiday_tool(now: Clock = _now) -> BaseTool:
    """Wrap the holiday lookup as the ``check_holiday`` tool (``now`` injectable)."""
    from langchain_core.tools import tool

    @tool(HOLIDAY_TOOL_NAME)
    def check_holiday(country: str = "US", date: str | None = None) -> str:
        """Look up public holidays. ``country`` is a two-letter code (US, GB, DE, …; defaults
        to US when the user names no country). With a YYYY-MM-DD ``date`` it says whether that
        day is a holiday; without one it reports the next upcoming public holiday."""
        import holidays as holidays_lib
        from dateutil import parser as date_parser

        try:
            calendar_class = holidays_lib.country_holidays  # raises NotImplementedError on unknown country
            today = now().date()
            if date is not None:
                day = date_parser.isoparse(date).date()
                name = calendar_class(country, years=day.year).get(day)
                nice = day.strftime("%B %d, %Y")
                if name:
                    return f"{nice} is {name} in {country}."
                return f"{nice} is not a public holiday in {country}."
            calendar = calendar_class(country, years=[today.year, today.year + 1])
            upcoming = sorted((d, n) for d, n in calendar.items() if d >= today)
            day, name = upcoming[0]
            return (
                f"The next {country} public holiday is {name} on "
                f"{day.strftime('%B %d, %Y')} — {_days((day - today).days)}."
            )
        except NotImplementedError:
            return "I don't have holiday data for that country."
        except (ValueError, OverflowError):
            return "I couldn't work out that date."

    return check_holiday
```

Note: confirm `holidays.country_holidays("ZZ")` raises `NotImplementedError` at call time (not lazily). If the installed version raises a different type for an unknown country, widen the `except` accordingly and update the test to match.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_holiday.py -q`
Expected: PASS (6 tests). If the library names Independence Day differently (e.g. "Independence Day (observed)" when July 4 falls on a weekend — 2026-07-04 is a Saturday), adjust the assertion to the real name the library returns for 2026.

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/holiday_tool.py tests/test_agent_cascade_holiday.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add check_holiday tool via holidays"
```

---

### Task 10: `sun_times` (astral + geocode)

**Files:**
- Create: `aai_cli/agent_cascade/suntimes_tool.py`
- Test: `tests/test_agent_cascade_suntimes.py` (Create)

**Interfaces:**
- Consumes: `geocode.geocode`, `geocode.GeoResult` (Task 2).
- Produces: `suntimes_tool.SUNTIMES_TOOL_NAME = "sun_times"`, `suntimes_tool.build_suntimes_tool(fetch=…, now=…) -> BaseTool`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_cascade_suntimes.py
from __future__ import annotations

from datetime import datetime, timezone

from aai_cli.agent_cascade import geocode
from aai_cli.agent_cascade.suntimes_tool import SUNTIMES_TOOL_NAME, build_suntimes_tool, _moon_name

_NOW = lambda: datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)


def test_sun_times_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(
        geocode, "geocode", lambda name, *, fetch: geocode.GeoResult("Paris", 48.85, 2.35, "Europe/Paris")
    )
    out = build_suntimes_tool(fetch=lambda url: {}, now=_NOW).invoke({"place": "Paris"})
    assert out.startswith("In Paris today the sun rises at ")
    assert "and sets at " in out and "the moon is " in out


def test_sun_times_no_match_apology(monkeypatch) -> None:
    monkeypatch.setattr(geocode, "geocode", lambda name, *, fetch: None)
    out = build_suntimes_tool(fetch=lambda url: {}).invoke({"place": "Nowhere"})
    assert out == "I couldn't find a place called 'Nowhere'."


def test_sun_times_bad_timezone_apology(monkeypatch) -> None:
    monkeypatch.setattr(
        geocode, "geocode", lambda name, *, fetch: geocode.GeoResult("X", 0.0, 0.0, "Not/AZone")
    )
    out = build_suntimes_tool(fetch=lambda url: {}, now=_NOW).invoke({"place": "X"})
    assert out == "I couldn't get the sun times there right now."


def test_moon_name_bins() -> None:
    assert _moon_name(0.0) == "a new moon"
    assert _moon_name(7.0) == "a first-quarter moon"
    assert _moon_name(14.0) == "a full moon"
    assert _moon_name(21.0) == "a last-quarter moon"


def test_tool_name() -> None:
    assert build_suntimes_tool().name == SUNTIMES_TOOL_NAME == "sun_times"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_suntimes.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `suntimes_tool.py`**

```python
# aai_cli/agent_cascade/suntimes_tool.py
"""A sunrise/sunset + moon-phase tool for the `assembly live` voice agent.

Resolves a place to coordinates + timezone via the shared :mod:`geocode` helper, then
computes today's sun times and the moon phase offline with ``astral``. Two seams — the
geocoder's :data:`Fetcher` and a :data:`Clock` — are injected in tests. Failures never
raise out to the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from aai_cli.agent_cascade import geocode

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

SUNTIMES_TOOL_NAME = "sun_times"

Clock = Callable[[], datetime]

# astral.moon.phase() returns 0..27.99; the lunar cycle's four quarters sit at 0/7/14/21.
# Each name owns a ~3.5-wide bin centered on its landmark (new wraps around 28→0).
_MOON_NAMES = (
    (1.75, "a new moon"),
    (5.25, "a waxing crescent"),
    (8.75, "a first-quarter moon"),
    (12.25, "a waxing gibbous"),
    (15.75, "a full moon"),
    (19.25, "a waning gibbous"),
    (22.75, "a last-quarter moon"),
    (26.25, "a waning crescent"),
)


def _get_json(url: str) -> object:
    """GET ``url`` and return parsed JSON (the default geocode-fetch seam)."""
    import httpx

    response = httpx.get(url, timeout=15.0)
    response.raise_for_status()
    return response.json()


def _now() -> datetime:
    """Return the current instant (the default clock)."""
    return datetime.now().astimezone()


def _moon_name(phase: float) -> str:
    """Map an astral moon phase (0..28) to a spoken phase name."""
    for upper, name in _MOON_NAMES:
        if phase < upper:
            return name
    return "a new moon"  # 26.25..28 wraps back to new


def build_suntimes_tool(fetch: geocode.Fetcher = _get_json, now: Clock = _now) -> BaseTool:
    """Wrap the sun/moon lookup as the ``sun_times`` tool (seams injectable)."""
    from langchain_core.tools import tool

    @tool(SUNTIMES_TOOL_NAME)
    def sun_times(place: str) -> str:
        """Get today's sunrise and sunset and the current moon phase for a place. Use for
        "when does the sun set in X", "what time is sunrise in X", or "what's the moon
        phase"."""
        result = geocode.geocode(place, fetch=fetch)
        if result is None:
            return f"I couldn't find a place called '{place}'."
        try:
            from astral import LocationInfo
            from astral.moon import phase as moon_phase
            from astral.sun import sun

            tz = ZoneInfo(result.timezone)
            today = now().astimezone(tz).date()
            location = LocationInfo(latitude=result.latitude, longitude=result.longitude)
            times = sun(location.observer, date=today, tzinfo=tz)
            sunrise = times["sunrise"].strftime("%I:%M %p")
            sunset = times["sunset"].strftime("%I:%M %p")
            return (
                f"In {result.name} today the sun rises at {sunrise} and sets at {sunset}, "
                f"and the moon is {_moon_name(moon_phase(today))}."
            )
        except Exception:
            return "I couldn't get the sun times there right now."

    return sun_times
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest tests/test_agent_cascade_suntimes.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/suntimes_tool.py tests/test_agent_cascade_suntimes.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add sun_times tool via astral"
```

---

### Task 11: Wire all eight into `brain.py`

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (the imports, `_TOOL_LABELS`, `_tool_capabilities`, `build_live_tools`)
- Test: `tests/test_agent_cascade_brain.py` (Modify — add wiring assertions)

**Interfaces:**
- Consumes: every `build_*_tool` and `*_TOOL_NAME` from Tasks 3–10.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_cascade_brain.py`:
```python
def test_build_live_tools_includes_all_keyless_tools(monkeypatch) -> None:
    # Force the keyed web-search tool absent so we assert only the always-on set.
    from aai_cli.agent_cascade import brain, firecrawl_search

    monkeypatch.setattr(firecrawl_search, "build_web_search_tool", lambda: None)
    names = {t.name for t in brain.build_live_tools()}
    assert {
        "get_weather", "read_url", "get_current_datetime",
        "look_up_topic", "calculate", "convert_units", "define_word",
        "get_time_in", "date_math", "check_holiday", "sun_times",
    } <= names


def test_tool_labels_present() -> None:
    from aai_cli.agent_cascade import brain

    for name, label in [
        ("look_up_topic", "Looking that up"),
        ("calculate", "Calculating"),
        ("convert_units", "Converting units"),
        ("define_word", "Looking up a definition"),
        ("get_time_in", "Checking the time there"),
        ("date_math", "Working out dates"),
        ("check_holiday", "Checking holidays"),
        ("sun_times", "Checking sun times"),
    ]:
        assert brain._tool_label(name) == label


def test_capabilities_advertise_new_tools() -> None:
    from aai_cli.agent_cascade import brain

    tools = brain.build_live_tools()
    caps = " ".join(brain._tool_capabilities(tools))
    for phrase in [
        "look up facts about people", "do arithmetic", "convert units and currencies",
        "define words", "tell the current time in a place",
        "do date math", "public holidays", "sunrise, sunset",
    ]:
        assert phrase in caps
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k "live_tools or labels or capabilities" -q`
Expected: FAIL — missing tool names/labels/phrases.

- [ ] **Step 3: Add the imports**

At the top of `brain.py`, extend the existing tool-module import (currently `from aai_cli.agent_cascade import datetime_tool, weather_tool, webpage_tool`) to:
```python
from aai_cli.agent_cascade import (
    calc_tool,
    datemath_tool,
    datetime_tool,
    define_tool,
    holiday_tool,
    suntimes_tool,
    topic_tool,
    units_tool,
    weather_tool,
    webpage_tool,
    worldclock_tool,
)
```

- [ ] **Step 4: Extend `_TOOL_LABELS`**

In the `_TOOL_LABELS` dict, after the `datetime_tool.DATETIME_TOOL_NAME` entry, add:
```python
    topic_tool.LOOKUP_TOOL_NAME: "Looking that up",
    calc_tool.CALC_TOOL_NAME: "Calculating",
    units_tool.CONVERT_TOOL_NAME: "Converting units",
    define_tool.DEFINE_TOOL_NAME: "Looking up a definition",
    worldclock_tool.WORLDCLOCK_TOOL_NAME: "Checking the time there",
    datemath_tool.DATEMATH_TOOL_NAME: "Working out dates",
    holiday_tool.HOLIDAY_TOOL_NAME: "Checking holidays",
    suntimes_tool.SUNTIMES_TOOL_NAME: "Checking sun times",
```

- [ ] **Step 5: Extend `_tool_capabilities`**

In `_tool_capabilities`, after the existing `datetime_tool` check, add one gated phrase per tool (mirroring the existing `if … in names:` blocks):
```python
    if topic_tool.LOOKUP_TOOL_NAME in names:
        capabilities.append("look up facts about people, places, and topics")
    if calc_tool.CALC_TOOL_NAME in names:
        capabilities.append("do arithmetic and percentages")
    if units_tool.CONVERT_TOOL_NAME in names:
        capabilities.append("convert units and currencies")
    if define_tool.DEFINE_TOOL_NAME in names:
        capabilities.append("define words and give synonyms")
    if worldclock_tool.WORLDCLOCK_TOOL_NAME in names:
        capabilities.append("tell the current time in a place")
    if datemath_tool.DATEMATH_TOOL_NAME in names:
        capabilities.append("do date math, like days until a date or the weekday of a date")
    if holiday_tool.HOLIDAY_TOOL_NAME in names:
        capabilities.append("tell you about public holidays")
    if suntimes_tool.SUNTIMES_TOOL_NAME in names:
        capabilities.append("tell you sunrise, sunset, and the moon phase for a place")
```

- [ ] **Step 6: Extend `build_live_tools`**

In `build_live_tools`, add the eight factory imports and append their tools to the `tools` list before the web-search block:
```python
    from aai_cli.agent_cascade.calc_tool import build_calc_tool
    from aai_cli.agent_cascade.datemath_tool import build_datemath_tool
    from aai_cli.agent_cascade.define_tool import build_define_tool
    from aai_cli.agent_cascade.holiday_tool import build_holiday_tool
    from aai_cli.agent_cascade.suntimes_tool import build_suntimes_tool
    from aai_cli.agent_cascade.topic_tool import build_lookup_tool
    from aai_cli.agent_cascade.units_tool import build_convert_tool
    from aai_cli.agent_cascade.worldclock_tool import build_worldclock_tool
```
and extend the list literal:
```python
    tools: list[BaseTool] = [
        build_weather_tool(),
        build_read_url_tool(),
        build_datetime_tool(),
        build_lookup_tool(),
        build_calc_tool(),
        build_convert_tool(),
        build_define_tool(),
        build_worldclock_tool(),
        build_datemath_tool(),
        build_holiday_tool(),
        build_suntimes_tool(),
    ]
```
Also update the `build_live_tools` docstring to reflect the broader keyless toolset (it currently names only weather/read-url/datetime).

- [ ] **Step 7: Run the brain tests**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: PASS. Also run the prompt snapshot tests if capability copy is snapshot-pinned: `uv run pytest tests/test_agent_cascade_prompt.py -q` and regenerate with `--snapshot-update` if the system-prompt golden changed.

- [ ] **Step 8: Commit (WIP)**

```bash
AAI_ALLOW_COMMIT=1 git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py tests/test_agent_cascade_prompt.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): bind and advertise the eight new tools"
```

---

### Task 12: Gate green + open PR-B

**Files:** none (verification + the real commit)

- [ ] **Step 1: Run the full gate**

Run: `./scripts/check.sh`
Expected: `All checks passed.` Fix anything it flags — most likely: a surviving mutant on a changed line (add an assertion that fails when that line breaks), a patch-coverage gap (cover the missed branch), `vulture` flagging an unused helper, or the docstring-coverage ratchet. Re-run until green.

- [ ] **Step 2: Push the branch and open PR-B**

The per-task WIP commits stay as-is — the merge queue squash-merges them into one commit, so no local rebase/reset is needed (interactive rebase isn't available in this environment anyway). Push the branch and open PR-B against `main`, noting in the body that it depends on PR-A (the dependency PR) and should land after it. Use a squash-merge title like:

```
feat(live): add eight keyless tools to assembly live

look_up_topic, calculate, convert_units, define_word, get_time_in,
date_math, check_holiday, sun_times — all always-bound (no API key),
plus a shared geocode.py extracted from weather_tool.
```

Let it land through the merge queue (which re-runs the diff-scoped gates against the combined state).

---

## Self-Review Notes

- **Spec coverage:** all eight tools (Tasks 3–10), `geocode.py` refactor (Task 2), the five deps (Task 1), and the three `brain.py` edits (Task 11) map to tasks. The spec's per-tool failure-apology strings are asserted verbatim in each task's tests.
- **Live-API caveat:** several tests exercise the *real* libraries (`pint`, `simpleeval`, `dateutil`, `holidays`, `astral`) — these are offline, so the tests stay hermetic. Only the HTTP tools (`look_up_topic`, `define_word`, currency path of `convert_units`, and the geocoders) use the injected `Fetcher`; no test hits the network.
- **Brittle-assertion watch:** the `date_math` span breakdown, the `holidays` Independence-Day naming, and the exact `pint` rounding are computed by third-party libraries — each task's Step 4 says to reconcile the assertion with the library's actual output rather than assume the wording above.
