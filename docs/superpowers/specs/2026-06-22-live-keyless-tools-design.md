# Eight keyless tools for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Broaden what the `assembly live` voice agent (the `agent-cascade` command) can
do for everyday spoken requests by adding eight new tools. All eight are
**always bound** (none needs an API key): they use keyless public APIs, offline
local computation, or bundled offline-data libraries. Each returns output short
enough to read aloud, extending the existing weather / read-url / datetime trio
toward "talk to a multimodal assistant" parity — with no API-key setup for the
user.

The eight tools:

1. `look_up_topic` — Wikipedia REST summary ("who is…", "what is…", "tell me about…").
2. `calculate` — safe arithmetic + curated math/statistics functions via `simpleeval` ("15% of 240", "square root of 150", "standard deviation of these numbers").
3. `convert_units` — physical units (via `pint`) + currency (via keyless frankfurter.app).
4. `define_word` — dictionary definition + synonyms (dictionaryapi.dev, keyless).
5. `get_time_in` — current local time in a named place (Open-Meteo geocode → `zoneinfo`).
6. `date_math` — date arithmetic via `python-dateutil` ("days until Christmas", "what weekday is July 4").
7. `check_holiday` — public-holiday lookup via the `holidays` library ("is Monday a holiday", "next US holiday").
8. `sun_times` — sunrise/sunset + moon phase via `astral`, reusing the shared geocoder.

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

- **Live-only.** All eight modules live in `aai_cli/agent_cascade/` and are bound
  only in the live voice agent. The coding agent's toolset is unchanged.
- **Keyless-first.** `look_up_topic`, `define_word`, `get_time_in`, and
  `sun_times` use keyless public APIs (the last two geocode via the shared
  `geocode.py`); `calculate`, `date_math`, and `check_holiday` need no network
  (offline libraries); `convert_units` uses keyless frankfurter.app for currency
  and the bundled `pint` library for physical units. No new environment
  variables.
- **Speakable output.** Each tool returns one short string suitable for TTS.

### Out of scope (YAGNI)

- No per-tool opt-out flags — the tools are read-only and cheap; they are always
  bound (no key gate, since none needs a key).
- No disambiguation UI anywhere — `look_up_topic` and the geocoding tools
  (`get_time_in`, `sun_times`, `convert_units`'s sibling `get_weather`) take the
  top match; ambiguity is handled in the spoken reply, not a prompt.
- No locale/units configuration — `convert_units` converts exactly the units the
  model names; `calculate` returns a plainly-formatted number.

## Dependencies (separate PR)

Five new dependencies back this feature, all pure-Python and offline (no key, no
service):

- [`pint`](https://pint.readthedocs.io/) — physical-unit conversion (`convert_units`).
- [`simpleeval`](https://github.com/danthedeckie/simpleeval) — safe arithmetic-expression evaluation (`calculate`).
- [`python-dateutil`](https://dateutil.readthedocs.io/) — date arithmetic (`date_math`).
- [`holidays`](https://github.com/vacanza/holidays) — offline public-holiday data (`check_holiday`).
- [`astral`](https://astral.readthedocs.io/) — sunrise/sunset + moon phase computation (`sun_times`).

Per the repo rule that dependency/`uv.lock` changes ship in their own
single-purpose PR, all five are added in **PR-A** (dependencies only — one
logical "add the libraries the new tools need" change), and the feature **PR-B**
lands on top of it.

- Add all five to `[project.dependencies]` in `pyproject.toml` + regenerate
  `uv.lock`. **Declare each directly even if already present transitively**
  (`python-dateutil` very likely is) — `deptry` flags using a transitive
  dependency directly.
- Heed the safe-chain version-floor caveat: pin each floor to the second-newest
  release, or resolution fails under the age gate.
- Each is imported **lazily** inside its tool factory (keeping the import off
  CLI startup) — matching `webpage_tool`'s lazy `core.webpage` import.

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
- Evaluates with **`simpleeval`** (lazily imported), seeded with a **curated
  whitelist** of pure functions and constants so the agent can "run code to get
  an answer" without any shell. The `SimpleEval` instance is built with an
  explicit `functions` table and `names` table (which *replace* simpleeval's
  defaults, dropping `rand`/`randint` and leaving no variables):
  - **functions:** `abs`, `round`, `min`, `max`, `sum` (builtins); `sqrt`,
    `floor`, `ceil`, `log`, `log10`, `hypot`, `gcd` (`math`); `mean`, `median`,
    `stdev` (`statistics`).
  - **constants (`names`):** `pi`, `e`.
- **Resource-exhaustion guarantee preserved.** `simpleeval` bounds the `**`
  operator via `MAX_POWER`, but it does **not** bound function arguments — so
  the whitelist deliberately **excludes** anything that can explode from small
  inputs (`factorial`, `pow`, `perm`, `comb`). Everything exposed is O(n) on its
  input and can't grow a giant number from a small one, so there is no
  unguarded-exponentiation backdoor. (Trig and a few others are easy to add
  later; the set is intentionally tight to bound the test surface.)
- `build_calc_tool()` exposes `calculate(expression: str) -> str`. The model
  turns the spoken question into an expression over that vocabulary; **the
  tool's docstring tells it how** (see below). The tool only evaluates + formats.
- **Tool docstring (the model-facing usage guidance):** the `@tool` docstring
  lists the allowed operators (`+ - * / // % **` and parentheses), the constants
  (`pi`, `e`), and the functions by name, notes that aggregate functions take a
  list (`mean([2,4,9])`), and gives worked examples — *"15% of 240" → `0.15 *
  240`*, *"square root of 150" → `sqrt(150)`*, *"standard deviation of 2, 4, 4,
  4, 5" → `stdev([2,4,4,4,5])`*, *"split 87 three ways" → `87 / 3`*. The usage
  contract lives in the tool definition, not in `brain.py`'s prompt.
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

### 6. `datemath_tool.py` — `date_math`

- `DATEMATH_TOOL_NAME = "date_math"`.
- Seam: `Clock` (`() -> datetime`, the `datetime_tool` shape) — supplies "today"
  so signed "from now" deltas are computable; injected in tests. `python-dateutil`
  is imported lazily.
- Signature: `date_math(date: str, other_date: str | None = None) -> str`, dates
  as ISO `YYYY-MM-DD`.
- **Division of labor (in the tool docstring):** the model is good at *knowing*
  calendar facts and bad at *counting* across them, so the docstring instructs
  it to work out the relevant date(s) itself and pass them as ISO strings; the
  tool does the exact day-counting and weekday. Worked examples in the docstring:
  *"days until Christmas" → `date_math("2026-12-25")`*, *"what weekday is July
  4th" → `date_math("2026-07-04")`*, *"days between March 1 and August 25" →
  `date_math("2026-03-01", "2026-08-25")`*.
- Behavior:
  - One date → its weekday plus a signed distance from today via the `Clock`,
    e.g. *"July 4, 2026 is a Saturday — 12 days from now."* / *"…— 8 days ago."*
    / *"…— that's today."*
  - Two dates → the total days between plus a human breakdown via
    `dateutil.relativedelta`, e.g. *"There are 177 days between March 1 and
    August 25, 2026 — about 5 months and 3 weeks."*
- Failure → apology: an unparseable date (`ValueError` from `dateutil.parser`) →
  *"I couldn't work out those dates."*

### 7. `holiday_tool.py` — `check_holiday`

- `HOLIDAY_TOOL_NAME = "check_holiday"`.
- Seam: `Clock` — supplies "today" for the next-holiday mode; injected in tests.
  The `holidays` library is offline data, imported lazily.
- Signature: `check_holiday(country: str = "US", date: str | None = None) -> str`.
  `country` is an ISO-3166 alpha-2 code (the `holidays` library's key, e.g.
  `US`, `GB`, `DE`); the docstring says so and notes it defaults to the US when
  the user names no country.
- Behavior:
  - With a date → name the holiday on it, or say there isn't one, e.g.
    *"December 25, 2026 is Christmas Day in the US."* / *"March 3, 2026 is not a
    public holiday in the US."*
  - Without a date → the next upcoming public holiday from today (via the
    `Clock`), e.g. *"The next US public holiday is Independence Day on July 4,
    2026 — in 12 days."*
- Failure → apology:
  - Unknown country code (`holidays` raises `NotImplementedError`) → *"I don't
    have holiday data for that country."*
  - Unparseable date → *"I couldn't work out that date."*

### 8. `suntimes_tool.py` — `sun_times`

- `SUNTIMES_TOOL_NAME = "sun_times"`.
- Seams: `Fetcher` (geocoding, via `geocode.geocode`) **and** `Clock` (today's
  date in the target zone) — both injected in tests. `astral` is imported lazily.
- `build_suntimes_tool(fetch=…, now=…)` exposes `sun_times(place: str) -> str`:
  geocode the place → lat/lon + IANA `timezone` → `astral.sun.sun()` for today in
  that zone → sunrise/sunset, plus `astral.moon.phase()` mapped to a phase name
  (new / waxing crescent / first quarter / waxing gibbous / full / …). One
  speakable string, e.g. *"In Paris today the sun rises at 6:01 AM and sets at
  9:45 PM, and the moon is a waxing gibbous."*
- Reuses the shared `geocode.py` — no second network call beyond geocoding;
  astral computes the rest offline.
- Failure → apology:
  - No geocoding match → *"I couldn't find a place called '<place>'."*
  - Astral/timezone error or fetch error → *"I couldn't get the sun times there
    right now."*

## Wiring into `brain.py`

All eight tools converge on three additive edits (the one shared file, edited
once):

1. `build_live_tools()` appends all eight. All are keyless ⇒ always present (no
   `FIRECRAWL_API_KEY`-style gate); the new libraries add no key.
2. `_tool_capabilities()` adds a spoken-capability phrase per tool, each gated on
   the tool's name being present in the bound set:
   - `look_up_topic` → *"look up facts about people, places, and topics"*
   - `calculate` → *"do arithmetic, percentages, and math like square roots and averages"*
   - `convert_units` → *"convert units and currencies"*
   - `define_word` → *"define words and give synonyms"*
   - `get_time_in` → *"tell the current time in a place"*
   - `date_math` → *"do date math, like days until a date or the weekday of a date"*
   - `check_holiday` → *"tell you about public holidays"*
   - `sun_times` → *"tell you sunrise, sunset, and the moon phase for a place"*
3. `_TOOL_LABELS` gains a present-tense affordance label per tool:
   - `look_up_topic` → *"Looking that up"*
   - `calculate` → *"Calculating"*
   - `convert_units` → *"Converting units"*
   - `define_word` → *"Looking up a definition"*
   - `get_time_in` → *"Checking the time there"*
   - `date_math` → *"Working out dates"*
   - `check_holiday` → *"Checking holidays"*
   - `sun_times` → *"Checking sun times"*

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
  `87 / 3` → `29`, asserting no float artifact leaks); a **parametrized case per
  whitelisted function and constant** (so a mutated function-table name is
  caught); and the apology for each failure mode — invalid syntax, a disallowed
  name (`foo + 1`), an **un-whitelisted function** (`factorial(5)` — proving the
  DoS exclusion holds), division by zero, and an over-`MAX_POWER` exponent.
- **`units_tool`:** a physical conversion via `pint` (e.g. miles→km, °F→°C), a
  currency conversion via a fake `fetch`, the unit-error apology, and the
  currency-fetch-error apology; the currency-vs-unit path selection.
- **`define_tool`:** happy path with synonyms and without, the not-found
  (object response) apology, and the fetch-error apology.
- **`worldclock_tool`:** happy path with an injected clock + fake geocode
  (assert the place, weekday/date/time, and tz abbreviation), the no-match
  apology, and the bad-timezone/fetch-error apology.
- **`datemath_tool`:** one-date mode (weekday + signed delta past/future/today,
  via an injected clock), two-date mode (total days + `relativedelta`
  breakdown), and the unparseable-date apology.
- **`holiday_tool`:** date-on-a-holiday and date-not-a-holiday, next-holiday mode
  (injected clock), the unknown-country apology, and the bad-date apology.
- **`suntimes_tool`:** happy path with fake geocode + injected clock (assert
  sunrise/sunset and the mapped moon-phase name), the no-match apology, and the
  astral/fetch-error apology.
- **`brain` wiring:** `build_live_tools()` includes each new `*_TOOL_NAME`;
  `_tool_capabilities()` / `build_system_prompt` advertises each; `_tool_label`
  returns each new label. These assert behavior (the exact phrase/label), not
  mere execution, to kill mutants.

No new env vars or commands ⇒ the docs-consistency gate stays green (verify
during implementation).

## PR sequence

- **PR-A (dependencies only):** add `pint`, `simpleeval`, `python-dateutil`,
  `holidays`, and `astral` to `pyproject.toml` + `uv.lock`. No feature code.
- **PR-B (feature):** `geocode.py` + the eight tool modules + the `weather_tool`
  geocode refactor + the `brain.py` wiring + all tests. Lands after PR-A so the
  new libraries are available.
