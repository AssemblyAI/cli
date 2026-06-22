# Five keyless tools for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Broaden what the `assembly live` voice agent (the `agent-cascade` command) can
do for everyday spoken requests by adding five new tools. All five are **always
bound** (none needs an API key): three use keyless public APIs, `calculate` does
offline computation via the bundled `simpleeval` library, and `convert_units`
combines the bundled `pint` library (physical units) with keyless
frankfurter.app (currency). Each returns
output short enough to read aloud, extending the existing weather / read-url /
datetime trio toward "talk to a multimodal assistant" parity — with no API-key
setup for the user.

The five tools:

1. `look_up_topic` — Wikipedia REST summary ("who is…", "what is…", "tell me about…").
2. `calculate` — pure, safe arithmetic ("what's 15% of 240", "split 87 three ways").
3. `convert_units` — physical units (via `pint`) + currency (via keyless frankfurter.app).
4. `define_word` — dictionary definition + synonyms (dictionaryapi.dev, keyless).
5. `get_time_in` — current local time in a named place (Open-Meteo geocode → `zoneinfo`).

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Its built-in tools today are `get_weather`,
`read_url`, and `get_current_datetime` (all keyless, always present) plus
`firecrawl_search` (bound only when `FIRECRAWL_API_KEY` is set). The only path
to "tell me about X" today is the *keyed* Firecrawl search; `look_up_topic`
closes that gap keylessly.

Tools are LangChain `BaseTool`s. The established pattern for a custom tool is
`aai_cli/agent_cascade/weather_tool.py` (and `datetime_tool.py`,
`webpage_tool.py`): pure, directly-testable helpers plus at most one thin seam
(a `Callable`) injected in tests so the suite needs no sockets/clock, and a tool
body that **never raises** — it catches its own failures and returns a short
spoken apology so a single tool outage can't sink a live turn.

## Scope

- **Live-only.** All five modules live in `aai_cli/agent_cascade/` and are bound
  only in the live voice agent. The coding agent's toolset is unchanged.
- **Keyless-first.** `look_up_topic`, `define_word`, and `get_time_in` use
  keyless public APIs; `calculate` needs no network; `convert_units` uses
  keyless frankfurter.app for currency and the bundled `pint` library for
  physical units. No new environment variables.
- **Speakable output.** Each tool returns one short string suitable for TTS.

### Out of scope (YAGNI)

- No per-tool opt-out flags — the tools are read-only and cheap; they are always
  bound (no key gate, since none needs a key).
- No disambiguation UI anywhere — `look_up_topic` and `get_time_in` take the top
  match; ambiguity is handled in the spoken reply, not a prompt.
- No locale/units configuration — `convert_units` converts exactly the units the
  model names; `calculate` returns a plainly-formatted number.

## Dependencies: `pint` + `simpleeval` (separate PR)

Two new dependencies back this feature: `convert_units`'s physical-unit path
uses [`pint`](https://pint.readthedocs.io/), and `calculate` uses
[`simpleeval`](https://github.com/danthedeckie/simpleeval), a small pure-Python
safe-expression evaluator. Per the repo rule that dependency/`uv.lock` changes
ship in their own single-purpose PR, both are added in **PR-A** (dependencies
only — one logical "add the libraries the new tools need" change), and the
feature **PR-B** lands on top of it.

- Add `pint` and `simpleeval` to `[project.dependencies]` in `pyproject.toml` +
  regenerate `uv.lock`.
- Heed the safe-chain version-floor caveat: pin each floor to the second-newest
  release, or resolution fails under the age gate.
- Both are imported **lazily** inside their tool factories (`pint` is a
  non-trivial import; `simpleeval` is small but the same discipline keeps the
  import off CLI startup) — matching `webpage_tool`'s lazy `core.webpage`
  import.

## Shared component: `geocode.py` (refactor)

Both `weather_tool` and `get_time_in` resolve a spoken place name to
coordinates via Open-Meteo's keyless geocoding endpoint, and `get_time_in`
additionally needs the IANA `timezone` field that endpoint already returns.

A new module `aai_cli/agent_cascade/geocode.py` factors this out:

- `Fetcher = Callable[[str], object]` — GETs a URL → parsed JSON (the existing
  weather-tool seam shape).
- `GeoResult` — `(name: str, latitude: float, longitude: float, timezone: str)`.
- `geocode(name, *, fetch) -> GeoResult | None` — query
  `https://geocoding-api.open-meteo.com/v1/search?name=<name>&count=1&language=en&format=json`,
  return the top match (now including `timezone`), or `None` when there is no
  match.

`weather_tool._geocode` is refactored to delegate to `geocode.geocode` (it
ignores the extra `timezone` field). This keeps one geocoding implementation and
gives `get_time_in` the timezone it needs. The weather tool's public behavior
and its `format_report`/`describe_weather_code` helpers are unchanged; only the
internal geocode call moves.

## The five tools

Each is a new module beside `weather_tool.py`. All expose exactly two public
names: the `*_TOOL_NAME` constant and the `build_*_tool(...)` factory.

### 1. `topic_tool.py` — `look_up_topic`

- `LOOKUP_TOOL_NAME = "look_up_topic"`.
- Seam: `Fetcher` (the weather-tool shape). Default `_get_json` uses `httpx`.
- Endpoint: `https://en.wikipedia.org/api/rest_v1/page/summary/<title>` with the
  title URL-encoded. The response carries `extract` (a clean ~1-paragraph plain
  text purpose-built to read aloud), `title`, and `type`.
- `build_lookup_tool(fetch=_get_json)` exposes
  `look_up_topic(topic: str) -> str`: fetch, return the `extract`, capped to a
  speakable length (`_MAX_CHARS`, `# pragma: no mutate` — a tuning knob).
- Failure → apology:
  - `type == "disambiguation"` or empty `extract` → *"I couldn't find a clear
    summary for '<topic>'."*
  - 404 / HTTP / network error (the `fetch` seam raises) → *"I couldn't look
    that up right now."*

### 2. `calc_tool.py` — `calculate`

- `CALC_TOOL_NAME = "calculate"`.
- **No seam — fully deterministic and offline** (the only tool with no
  non-determinism, so no injected callable).
- Evaluates with **`simpleeval`** (lazily imported): a `SimpleEval` instance with
  `names`/`functions` left empty so only arithmetic over numeric literals is
  allowed (no variables, no function calls). `simpleeval` already guards the
  resource-exhaustion cases — `MAX_POWER` against exponent bombs (`9 ** 9 ** 9`)
  and string-length limits — so the tool keeps no hand-rolled AST walker.
- `build_calc_tool()` exposes `calculate(expression: str) -> str`. The model is
  responsible for turning a spoken word-problem into a plain arithmetic
  expression; **the tool's docstring tells it how** (see below). The tool only
  evaluates and formats.
- **Tool docstring (the model-facing usage guidance):** the `@tool` docstring
  states that `expression` must be a plain arithmetic expression using only
  numbers and the operators `+ - * / // % ** ( )`, with no words, units, or
  variable names, and gives worked examples so the model rewrites speech into a
  valid expression — e.g. *"15% of 240" → `0.15 * 240`*, *"split 87 three ways"
  → `87 / 3`*, *"3 plus 4 times 5" → `3 + 4 * 5`*. This is the deliverable the
  user called out: the formatting contract lives in the tool definition, not in
  `brain.py`'s prompt.
- **Output formatting (the real fiddly part):** render the result so it reads
  aloud cleanly — integers print without a decimal (`36`, not `36.0`), and
  non-integers are rounded to a sensible precision so float artifacts never leak
  (`87 / 3` reads as `29` after rounding, not `28.999999999999996`). This
  rounding requirement is explicit because it, not parsing safety, is the
  tool's genuine risk.
- Failure → apology: any `simpleeval` error (invalid syntax, a disallowed
  name/call, an over-`MAX_POWER` exponent) or `ZeroDivisionError` → *"I couldn't
  compute that."*

### 3. `units_tool.py` — `convert_units`

- `CONVERT_TOOL_NAME = "convert_units"`.
- Signature: `convert_units(value: float, from_unit: str, to_unit: str) -> str`.
- Seam: `Fetcher` — **used only on the currency path**; the `pint` path is
  deterministic.
- Path selection: a module-level set of ISO-4217 currency codes. If both
  `from_unit` and `to_unit` are currency codes → currency path; otherwise →
  physical-unit path.
  - **Currency:** `https://api.frankfurter.app/latest?amount=<value>&from=<F>&to=<T>`
    (keyless). Read `rates[<T>]`, format e.g. *"100 USD is 92.4 EUR."*
  - **Physical units:** lazily `import pint`, then
    `ureg.Quantity(value, from_unit).to(to_unit)`, format the magnitude + unit,
    e.g. *"5 miles is 8.05 kilometers."* / *"350 °F is 176.67 °C."*
- Failure → apology:
  - `pint` `UndefinedUnitError` / `DimensionalityError` (incompatible or unknown
    units) → *"I couldn't convert those units."*
  - currency fetch error / unknown code in `rates` → *"I couldn't get that
    exchange rate right now."*

### 4. `define_tool.py` — `define_word`

- `DEFINE_TOOL_NAME = "define_word"`.
- Seam: `Fetcher`. Endpoint:
  `https://api.dictionaryapi.dev/api/v2/entries/en/<word>` — a JSON **array** of
  entries on success; a JSON **object** (with a "No Definitions Found" title) on
  a miss.
- `build_define_tool(fetch=_get_json)` exposes `define_word(word: str) -> str`:
  take the first entry's first meaning — part of speech + first definition — and
  append up to a couple of synonyms when present, e.g. *"ephemeral (adjective):
  lasting a very short time. Synonyms: transient, fleeting."*
- Failure → apology:
  - Response is not a non-empty array (word not found) → *"I couldn't find a
    definition for '<word>'."*
  - HTTP / network error → *"I couldn't look up that word right now."*

### 5. `worldclock_tool.py` — `get_time_in`

- `WORLDCLOCK_TOOL_NAME = "get_time_in"`.
- Seams: `Fetcher` (geocoding, via `geocode.geocode`) **and** `Clock`
  (`() -> datetime`, the `datetime_tool` shape) — the current instant, injected
  in tests.
- `build_worldclock_tool(fetch=…, now=…)` exposes `get_time_in(place: str) -> str`:
  geocode the place → its IANA `timezone` → convert `now()` into that zone with
  `zoneinfo.ZoneInfo(tz)` → format with the same cross-platform `strftime` codes
  `datetime_tool` uses, e.g. *"In Tokyo it's Monday, June 22, 2026 at 11:30 PM
  JST."*
- Failure → apology:
  - No geocoding match → *"I couldn't find a place called '<place>'."*
  - Bad/unknown timezone or fetch error → *"I couldn't get the time there right
    now."*

## Wiring into `brain.py`

All five tools converge on three additive edits (the one shared file, edited
once):

1. `build_live_tools()` appends all five. All are keyless ⇒ always present (no
   `FIRECRAWL_API_KEY`-style gate); `pint` adds no key.
2. `_tool_capabilities()` adds a spoken-capability phrase per tool, each gated on
   the tool's name being present in the bound set:
   - `look_up_topic` → *"look up facts about people, places, and topics"*
   - `calculate` → *"do arithmetic and percentages"*
   - `convert_units` → *"convert units and currencies"*
   - `define_word` → *"define words and give synonyms"*
   - `get_time_in` → *"tell the current time in a place"*
3. `_TOOL_LABELS` gains a present-tense affordance label per tool:
   - `look_up_topic` → *"Looking that up"*
   - `calculate` → *"Calculating"*
   - `convert_units` → *"Converting units"*
   - `define_word` → *"Looking up a definition"*
   - `get_time_in` → *"Checking the time there"*

The existing `_NO_TOOLS_GUIDANCE` path is unaffected (reached only when
`build_system_prompt` is handed an explicitly empty toolset, which tests do).

## Error handling (cross-cutting)

Every tool body is best-effort and **never raises** out to the graph — a tool
outage must not trip `brain`'s "couldn't complete the turn" path. Each catches
its own failures and returns one of the short spoken strings listed per tool
above. `calculate` is the one tool with no network failure mode; its only
apology path is an invalid/disallowed expression.

## Testing

Targets the gate's 100% patch-coverage + diff-scoped mutation requirements:
assertions must *fail* if a changed line breaks, not merely execute it. One
`tests/test_agent_cascade_<x>_tool.py` per tool (plus a `geocode` test), all
hermetic via injected seams — no real network/clock.

- **`geocode.py`:** URL building (params present/correct), top-match extraction
  incl. the new `timezone` field, and the no-match → `None` path. `weather_tool`
  tests stay green through the refactor (its geocode now delegates).
- **`topic_tool`:** happy path (canned `extract` → expected string), the
  disambiguation/empty-extract apology, and the fetch-error apology; truncation
  at `_MAX_CHARS`.
- **`calc_tool`:** correct evaluation for several expressions incl. precedence
  and unary minus; the integer-vs-float **output formatting** (`36` not `36.0`;
  `87 / 3` → `29`, asserting no float artifact leaks); and the apology for each
  failure mode — invalid syntax, a disallowed name (e.g. `foo + 1`), division by
  zero, and an over-`MAX_POWER` exponent. The `simpleeval` instance is asserted
  to expose no names/functions (the safe-configuration contract).
- **`units_tool`:** a physical conversion via `pint` (e.g. miles→km, °F→°C), a
  currency conversion via a fake `fetch`, the unit-error apology, and the
  currency-fetch-error apology; the currency-vs-unit path selection.
- **`define_tool`:** happy path with synonyms and without, the not-found
  (object response) apology, and the fetch-error apology.
- **`worldclock_tool`:** happy path with an injected clock + fake geocode
  (assert the place, weekday/date/time, and tz abbreviation), the no-match
  apology, and the bad-timezone/fetch-error apology.
- **`brain` wiring:** `build_live_tools()` includes each new `*_TOOL_NAME`;
  `_tool_capabilities()` / `build_system_prompt` advertises each; `_tool_label`
  returns each new label. These assert behavior (the exact phrase/label), not
  mere execution, to kill mutants.

No new env vars or commands ⇒ the docs-consistency gate stays green (verify
during implementation).

## PR sequence

- **PR-A (dependencies only):** add `pint` and `simpleeval` to `pyproject.toml`
  + `uv.lock`. No feature code.
- **PR-B (feature):** `geocode.py` + the five tool modules + the `weather_tool`
  geocode refactor + the `brain.py` wiring + all tests. Lands after PR-A so
  `pint` and `simpleeval` are available.
