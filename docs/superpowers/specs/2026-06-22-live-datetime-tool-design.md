# Date/time tool for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Give the `assembly live` voice agent (the `agent-cascade` command) a keyless,
always-present tool that reports the **current local date and time**, so it can
answer "what time is it?", "what's today's date?", or "what day is it?" — the
kind of thing a live multimodal assistant is expected to know.

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Built-in tools are added in
`build_live_tools()`; today that is the keyless Open-Meteo weather tool (always
present) plus Firecrawl web search (only when `FIRECRAWL_API_KEY` is set). The
established pattern for a custom live tool is `aai_cli/agent_cascade/weather_tool.py`
/ `webpage_tool.py`: pure, directly-testable helpers plus a single injected seam
(a `Callable`) so the suite needs no real I/O.

The current date/time is the simplest such tool: **no network, no key**. Its only
non-determinism is the system clock, which is injected as a `Clock` seam so tests
are hermetic (the suite pins `TZ` and forbids unmocked time — see `tests/CLAUDE.md`).

## Scope

- **Live-only.** The tool lives in `aai_cli/agent_cascade/` and is bound only in
  the live voice agent.
- **Local time only.** Returns the current date and time in the host's local
  timezone. No timezone/place argument (YAGNI — chosen explicitly).
- **Always present** (keyless, no I/O), like the weather tool.

### Out of scope (YAGNI)

- No timezone or place-name argument ("what time is it in Tokyo?").
- No date arithmetic ("how many days until…").
- No configurable format.

## Architecture

A new module `aai_cli/agent_cascade/datetime_tool.py`, beside `weather_tool.py`.

```
get_current_datetime()  ──▶  now() (Clock seam, default _now)  ──▶ aware datetime
                        └──▶  _format(now)                      ──▶ short spoken string
```

`get_current_datetime` takes **no arguments**.

### Components

- `DATETIME_TOOL_NAME = "get_current_datetime"` — the registered tool name.
  `brain.py` keys its UI label and capability phrase off this, so a test pins it.
- `Clock = Callable[[], datetime]`, default `_now` → `datetime.now().astimezone()`
  (a timezone-aware local datetime). **The only seam**; tests inject a fixed
  `datetime` so the whole flow is deterministic with no real clock.
- `_format(now: datetime) -> str` — pure → a short, speakable string. Uses only
  **cross-platform** `strftime` codes (no `%-d`/`%-I`, which break on Windows where
  the suite also runs). Example: `"It's Monday, June 22, 2026 at 02:30 PM PDT."`
  (zero-padded day/hour is fine — the model re-speaks it).
- `build_datetime_tool(now: Clock = _now) -> BaseTool` — the
  `@tool(DATETIME_TOOL_NAME)` wrapper exposing `get_current_datetime() -> str`.
  No try/except: reading the local clock has no failure mode to swallow, so —
  unlike weather/read_url — there is **no blind `except Exception`** and therefore
  **no `pyproject.toml` `BLE001` per-file-ignore**.

### Data flow per call

1. The model calls `get_current_datetime` (no args).
2. `now()` returns the current timezone-aware local datetime.
3. `_format` renders a short, speakable date+time string the model reads aloud.

## Wiring into `brain.py`

The three spots a built-in tool touches (mirroring weather):

- `build_live_tools()` — **always** includes `build_datetime_tool()` (keyless,
  no I/O), alongside the weather tool; web search stays key-gated.
- `_tool_capabilities()` — adds *"tell you the current date and time"* when the
  tool is present.
- `_TOOL_LABELS[DATETIME_TOOL_NAME] = "Checking the time"` for the live-UI
  affordance.

## Error handling

None needed. The tool performs no I/O and the clock call cannot fail in normal
operation, so it returns a value unconditionally — no apology path, no blind
except, no lint exemption.

## Testing

Hermetic via the injected `Clock`; targets 100% patch coverage + the diff-scoped
mutation gate (assertions must *fail* if a changed line breaks).

- `_format` tested directly against a fixed, timezone-aware `datetime` → the
  EXACT expected string (kills format mutations).
- The tool driven end-to-end with an injected fixed clock → the EXACT string.
- `_now()` returns a `datetime` that is timezone-aware (`tzinfo is not None`),
  covering the default seam.
- `brain` wiring:
  - `build_live_tools()` includes a tool named `DATETIME_TOOL_NAME` (update the
    existing EXACT-set `build_live_tools` assertions — both the keyed and the
    no-firecrawl-key tests — to include it).
  - `build_system_prompt`/`_tool_capabilities` advertises the date/time phrase.
  - `_tool_label(DATETIME_TOOL_NAME) == "Checking the time"` (exact string).
