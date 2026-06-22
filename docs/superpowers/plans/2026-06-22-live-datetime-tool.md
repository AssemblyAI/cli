# Date/time tool for `assembly live` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `assembly live` voice agent a keyless, always-present `get_current_datetime` tool that reports the current local date and time.

**Architecture:** A new `aai_cli/agent_cascade/datetime_tool.py` wraps the system clock (via an injected `Clock` seam) as a zero-argument LangChain `BaseTool`, formatting a short speakable string, and is wired into the live deepagents graph through the three tool hooks in `brain.py`. Mirrors `weather_tool.py` minus the network/error-handling (the clock can't fail).

**Tech Stack:** Python 3.12+, stdlib `datetime`, LangChain `@tool`, deepagents, pytest. Tests are hermetic via the injected `Clock` (no real clock).

## Global Constraints

- `from __future__ import annotations` at module top; modern typing (`X | None`).
- Cross-platform `strftime` only: NO `%-d` / `%-I` (they break on Windows, where the suite also runs). Use `%d` / `%I` (zero-padded).
- The single non-determinism is an injected `Clock` callable (default `_now`); tests pass a fixed `datetime` — no real clock.
- NO try/except / no blind `except Exception` (the clock has no failure mode) → therefore NO `pyproject.toml` `BLE001` change.
- Internal helper docstrings keep trailing periods; the tool's own docstring is its model-facing description.
- Gate rules: iterate with targeted `uv run pytest`. The full `./scripts/check.sh` is deferred to the human (the branch carries unrelated WIP and can't be cleanly gated mid-flight). Commit with `AAI_ALLOW_COMMIT=1` and stage ONLY this feature's files (never `git add -A`). `uv` is safe-chain-wrapped and may emit a `EPERM listen` error in some sandboxes — retry once.

---

### Task 1: The `datetime_tool.py` module (get_current_datetime, standalone)

**Files:**
- Create: `aai_cli/agent_cascade/datetime_tool.py`
- Test: `tests/test_agent_cascade_datetime.py`

**Interfaces:**
- Produces (Task 2 relies on these exact names): `DATETIME_TOOL_NAME = "get_current_datetime"` (str); `Clock = Callable[[], datetime]`; `build_datetime_tool(now: Clock = _now) -> BaseTool` (tool named `get_current_datetime`, zero args, returns `str`); `_format(now: datetime) -> str`, `_now() -> datetime` (module-private).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_cascade_datetime.py`:

```python
"""Tests for the keyless local date/time tool behind `assembly live`.

The tool's only non-determinism is the injected ``Clock`` callable, so the whole
flow is deterministic with no real clock (and pytest-socket stays armed — no I/O).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aai_cli.agent_cascade import datetime_tool

# A fixed, timezone-aware instant: Monday, 2026-06-22 14:30 at a fixed -07:00 offset.
# A fixed offset (not a named zone) keeps %Z deterministic cross-platform without tzdata.
_FIXED = datetime(2026, 6, 22, 14, 30, tzinfo=timezone(timedelta(hours=-7)))
_EXPECTED = "It's Monday, June 22, 2026 at 02:30 PM UTC-07:00."


# --- _format -----------------------------------------------------------------


def test_format_renders_exact_speakable_string():
    assert datetime_tool._format(_FIXED) == _EXPECTED


# --- _now (default seam) -----------------------------------------------------


def test_now_returns_timezone_aware_datetime():
    n = datetime_tool._now()
    assert isinstance(n, datetime)
    # astimezone() makes it aware; a naive datetime (mutation dropping it) fails here.
    assert n.tzinfo is not None


# --- build_datetime_tool -----------------------------------------------------


def test_tool_is_named_get_current_datetime():
    tool = datetime_tool.build_datetime_tool(now=lambda: _FIXED)
    assert tool.name == datetime_tool.DATETIME_TOOL_NAME


def test_tool_returns_formatted_current_datetime():
    tool = datetime_tool.build_datetime_tool(now=lambda: _FIXED)
    assert tool.invoke({}) == _EXPECTED
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /tmp/claude-501/aai-datetime-wt && uv run pytest tests/test_agent_cascade_datetime.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.agent_cascade.datetime_tool'`.

- [ ] **Step 3: Write the module**

Create `aai_cli/agent_cascade/datetime_tool.py`:

```python
"""A keyless local date/time tool for the `assembly live` voice agent.

Reports the current local date and time so the live agent can answer "what time is
it?", "what's today's date?", or "what day is it?". It needs no network and no API
key — just the system clock — making it, like the weather tool, always present.

The only non-determinism is the :data:`Clock` seam (a ``() -> datetime`` callable),
injected in tests so the flow is deterministic with no real clock. Everything else
(the spoken formatting) is pure and tested directly. There is no failure mode to
handle: reading the local clock cannot fail, so the tool returns unconditionally.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# The registered tool name. ``brain.py`` keys its UI label and capability phrase off
# this, so a test pins it.
DATETIME_TOOL_NAME = "get_current_datetime"

# A clock returns the current instant. Injected in tests (the only non-determinism).
Clock = Callable[[], datetime]


def _now() -> datetime:
    """Return the current local time as a timezone-aware datetime (the default clock)."""
    return datetime.now().astimezone()


def _format(now: datetime) -> str:
    """Render ``now`` as one short, speakable date+time string.

    Uses only cross-platform ``strftime`` codes (no ``%-d``/``%-I``, which break on
    Windows). Zero-padded day/hour is fine — the model reads the string aloud.
    """
    return now.strftime("It's %A, %B %d, %Y at %I:%M %p %Z.")


def build_datetime_tool(now: Clock = _now) -> BaseTool:
    """Wrap the local clock as the ``get_current_datetime`` tool (``now`` injectable)."""
    from langchain_core.tools import tool

    @tool(DATETIME_TOOL_NAME)
    def get_current_datetime() -> str:
        """Get the current local date and time. Use when asked the date, the day of the
        week, or the time."""
        return _format(now())

    return get_current_datetime
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /tmp/claude-501/aai-datetime-wt && uv run pytest tests/test_agent_cascade_datetime.py -q`
Expected: PASS (4 passed). If `test_format_renders_exact_speakable_string` fails on the weekday or `%p`/`%Z` text, confirm the actual `strftime` output and reconcile the EXPECTED literal (2026-06-22 is a Monday; a fixed `-07:00` offset renders `%Z` as `UTC-07:00`; the C locale renders `%p` as `PM`).

- [ ] **Step 5: Verify lint/types and commit**

```bash
cd /tmp/claude-501/aai-datetime-wt
uv run ruff check aai_cli/agent_cascade/datetime_tool.py tests/test_agent_cascade_datetime.py
uv run pyright aai_cli/agent_cascade/datetime_tool.py
git add aai_cli/agent_cascade/datetime_tool.py tests/test_agent_cascade_datetime.py
AAI_ALLOW_COMMIT=1 git commit -m "feat: local date/time tool module for assembly live"
```
(Ignore editor "unknown import symbol" diagnostics on the brand-new module — those are stale-index false positives; `uv run pyright` is authoritative.)

---

### Task 2: Wire `get_current_datetime` into the live agent (`brain.py`)

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (import; `_TOOL_LABELS`; `_tool_capabilities`; `build_live_tools`)
- Test: `tests/test_agent_cascade_brain.py` (update the two EXACT-assertion `build_live_tools` tests; add label + capability tests)

**Interfaces:**
- Consumes from Task 1: `datetime_tool.DATETIME_TOOL_NAME`, `datetime_tool.build_datetime_tool`.
- Produces: `build_live_tools()` includes the datetime tool always; `_tool_label("get_current_datetime") == "Checking the time"`; `build_system_prompt` advertises the date/time capability.

- [ ] **Step 1: Update the existing exact-assertion tests + add new tests (failing)**

In `tests/test_agent_cascade_brain.py`:

Change the existing import `from aai_cli.agent_cascade import brain, weather_tool` to also import `datetime_tool`:
```python
from aai_cli.agent_cascade import brain, datetime_tool, weather_tool
```

Update `test_build_live_tools_has_weather_and_web_search_when_keyed` — its exact-set assertion must include the datetime tool:
```python
    assert sorted(names) == sorted(
        [brain.WEB_SEARCH_TOOL_NAME, weather_tool.WEATHER_TOOL_NAME, datetime_tool.DATETIME_TOOL_NAME]
    )
```

Replace `test_build_live_tools_is_just_weather_without_firecrawl_key` (rename — it is no longer "just weather"):
```python
def test_build_live_tools_has_weather_and_datetime_without_firecrawl_key(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: None)
    # No FIRECRAWL_API_KEY -> no web search, but the keyless weather + datetime tools load.
    names = [tool.name for tool in brain.build_live_tools()]
    assert names == [weather_tool.WEATHER_TOOL_NAME, datetime_tool.DATETIME_TOOL_NAME]
```

Add two new tests at the end of the file:
```python
def test_datetime_tool_advertised_in_system_prompt():
    prompt = brain.build_system_prompt(
        "persona", tools=[_NamedTool(datetime_tool.DATETIME_TOOL_NAME)]
    )
    assert "current date and time" in prompt


def test_tool_label_maps_datetime():
    assert brain._tool_label(datetime_tool.DATETIME_TOOL_NAME) == "Checking the time"
```

> NOTE: This plan quotes `brain.py`/test code as of base `5a6a88c`. If the branch has advanced, do NOT blindly apply literal replacements — read the CURRENT files and make the minimal edits achieving the four wiring changes; in particular re-check the exact `build_live_tools` assertions and any `_tool_capabilities` ordering test, and update every assertion that pins the exact toolset.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /tmp/claude-501/aai-datetime-wt && uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: FAIL on the two updated `build_live_tools` tests (datetime tool not built yet), `test_datetime_tool_advertised_in_system_prompt` (phrase absent), and `test_tool_label_maps_datetime` (falls back to `"Using get_current_datetime"`).

- [ ] **Step 3: Wire `brain.py` — import**

Add `datetime_tool` to the existing `from aai_cli.agent_cascade import weather_tool` line:
```python
from aai_cli.agent_cascade import datetime_tool, weather_tool
```

- [ ] **Step 4: Wire `brain.py` — `_TOOL_LABELS`**

Add the datetime label to `_TOOL_LABELS`:
```python
_TOOL_LABELS = {
    WEB_SEARCH_TOOL_NAME: "Searching the web",
    weather_tool.WEATHER_TOOL_NAME: "Checking the weather",
    datetime_tool.DATETIME_TOOL_NAME: "Checking the time",
}
```

- [ ] **Step 5: Wire `brain.py` — `_tool_capabilities`**

After the weather `if` block, add the datetime capability:
```python
    if weather_tool.WEATHER_TOOL_NAME in names:
        capabilities.append("tell someone the current weather and short forecast for a place")
    if datetime_tool.DATETIME_TOOL_NAME in names:
        capabilities.append("tell you the current date and time")
    return capabilities
```

- [ ] **Step 6: Wire `brain.py` — `build_live_tools`**

Add the lazy import and include the datetime tool in the always-present list:
```python
    from aai_cli.agent_cascade.datetime_tool import build_datetime_tool
    from aai_cli.agent_cascade.weather_tool import build_weather_tool
    from aai_cli.code_agent.firecrawl_search import build_web_search_tool

    tools: list[BaseTool] = [build_weather_tool(), build_datetime_tool()]
    search = build_web_search_tool()
    if search is not None:
        tools.append(search)
    return tools
```
Also update the `build_live_tools` docstring's first sentence to name the datetime tool (e.g. "the keyless weather and date/time tools, plus Firecrawl web search when ``FIRECRAWL_API_KEY`` is set.").

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd /tmp/claude-501/aai-datetime-wt && uv run pytest tests/test_agent_cascade_brain.py tests/test_agent_cascade_datetime.py -q`
Expected: PASS (all green).

- [ ] **Step 8: Verify lint/types and commit**

```bash
cd /tmp/claude-501/aai-datetime-wt
uv run ruff check aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
uv run pyright aai_cli/agent_cascade/brain.py
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "feat: wire get_current_datetime tool into assembly live"
```
(The committed `test_agent_cascade_brain.py` has pre-existing pyright errors from a `_NamedTool` test double and `build_completer`/`build_streamer` protocol mismatches; confirm your change adds only the same `_NamedTool`-convention kind, nothing new.)

---

## Self-Review

**Spec coverage:** new module + `DATETIME_TOOL_NAME`/`Clock`/`_now`/`_format`/`build_datetime_tool` → Task 1; always-present keyless wiring → Task 2 Step 6; `_tool_capabilities` phrase → Task 2 Step 5; `_TOOL_LABELS` "Checking the time" → Task 2 Step 4; no error handling / no BLE001 → honored (no try/except in the module); cross-platform strftime → Task 1 Step 3 + Global Constraints; hermetic clock-seam tests incl. exact-string + tz-aware assertions → Tasks 1 & 2.

**Placeholder scan:** none — every step carries full code/commands.

**Type consistency:** `DATETIME_TOOL_NAME`, `Clock`, `build_datetime_tool(now=…)`, `_format`, `_now` names match between Task 1 (definition) and Task 2 (consumption); `build_live_tools` order `[weather, datetime, search?]` matches the updated keyless test assertion.
