# Read-a-URL tool (web pages + PDFs) for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Give the `assembly live` voice agent (the `agent-cascade` command) a keyless,
always-available tool that fetches a **web page or PDF by URL** and returns its
readable text. The agent can then read an article or document the user names, or
follow a link surfaced by web search — bringing the live agent closer to the
"talk to a multimodal assistant" experience with no API-key setup.

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Its only built-in tool today is Firecrawl
web *search*, bound only when `FIRECRAWL_API_KEY` is set — so an unkeyed session
runs tool-free. Reading a specific URL is a distinct capability from searching:
search finds pages, this one reads one.

The CLI already has a purpose-built reader: `core/webpage.py:fetch_article()`.
It fetches a URL with the project's pinned `httpx2` client, then narrows the body
to readable text — **HTML** via trafilatura (nav/sidebars/footers stripped),
**PDF** via pypdf (text layer of every page, detected by Content-Type or the
`%PDF-` magic bytes). It already backs `assembly speak --url`, so its output is
narration-oriented — exactly what a spoken agent wants. It rejects non-http(s)
URLs and raises on an empty/failed fetch.

> Note: `transcribe` itself uses **no** PDF/webpage tools — it pipes transcript
> text through the LLM Gateway. The reusable reader is `fetch_article` (powers
> `speak --url`); the coding agent's `fetch_tool.py:fetch_url` returns *raw*
> truncated HTML with no PDF extraction and is the wrong fit here.

The established pattern for a live tool is `aai_cli/agent_cascade/weather_tool.py`:
pure/directly-testable helpers plus a single thin network seam (a `Callable`)
injected in tests so the suite needs no sockets, and best-effort error handling
that returns a short spoken string rather than raising into the graph.

## Scope

- **Live-only.** The tool lives in `aai_cli/agent_cascade/` and is bound only in
  the live voice agent. The coding agent's toolset is unchanged.
- **Reader: reuse `core/webpage.py:fetch_article`** (HTML + PDF, keyless). No new
  fetching/parsing logic.
- **Always present.** `fetch_article` needs no API key, so the tool is bound in
  every live session — the first built-in tool that is *always* available, so an
  unkeyed session is no longer tool-free.

### Out of scope (YAGNI)

- No local-filesystem read/write — that is the separate `live-file-readwrite`
  design. This tool reads **remote URLs only**.
- No approval gate. A spoken turn can't pause for a keyboard confirmation, so
  live tools are read-only and auto-approved (the existing stance). See the
  security note below.
- No `--no-read` opt-out flag; no per-call content-type selection (the reader
  auto-detects HTML vs PDF).

## Architecture

A new module `aai_cli/agent_cascade/webpage_tool.py`, beside `weather_tool.py`.

```
read_url(url)  ──▶  read(url)        ──▶ core.webpage.fetch_article  ──▶ Article(text, title, url)
               └──▶  _format(article) ──▶ truncated "title + text" string for the model
```

`agent_cascade` → `core` is an allowed import direction (the layers contract
forbids feature slices from importing `commands`, not `core`).

### Components

- `READ_URL_TOOL_NAME = "read_url"` — the registered tool name. `brain.py`
  detects availability and labels the live-UI affordance by this name, so a test
  pins it.
- `Reader = Callable[[str], Article]`, default `fetch_article` — **the only
  network seam**. Tests inject a fake returning a canned `Article` (happy path)
  or raising a `CLIError` (failure paths), so the whole flow runs with no
  sockets.
- `_MAX_CHARS` — truncation cap (~16000), so a long article or multi-page PDF
  can't blow the model's context budget. A `±` shift is behaviorally equivalent,
  so the constant line is `# pragma: no mutate`.
- `_format(article) -> str` — pure. Leads with the title (when present) then the
  readable body, truncated to `_MAX_CHARS` with a trailing `…[truncated]` marker
  when it overflows. The body is *source text for the model to summarize aloud*,
  not spoken verbatim, so it needn't be "speakable" — only bounded.
- `build_read_url_tool(read=fetch_article) -> BaseTool` — the
  `@tool(READ_URL_TOOL_NAME)` wrapper exposing `read_url(url: str) -> str`. The
  `read` seam is injectable for hermetic tests. Plus `READ_URL_TOOL_NAME`, these
  are the module's only public names.

### Data flow per call

1. The model calls `read_url` with a URL string (from the conversation or a
   prior web-search result).
2. `read` (`fetch_article`) fetches and extracts readable text — HTML via
   trafilatura, PDF via pypdf — returning an `Article(text, title, url)`.
3. `_format` renders `title + text`, truncated, for the model to read and
   summarize aloud.

## Wiring into `brain.py`

The three spots a built-in tool touches:

- `build_live_tools()` — **always** includes the read-url tool (keyless), so even
  an unkeyed session has a real capability. Firecrawl search stays key-gated and
  is appended alongside it when present.
- `_tool_capabilities()` — restructured to collect *multiple* built-in capability
  phrases (today it returns at most one). Adds *"read a web page or PDF you have
  the URL for"* when the read-url tool is present; web search's phrase is
  appended when that tool is present. `_join_clause` already renders a list.
- `_TOOL_LABELS[READ_URL_TOOL_NAME] = "Reading the page"` so the live UI shows a
  meaningful affordance while the tool runs (matching `"Searching the web"`).

The `_NO_TOOLS_GUIDANCE` path still works: it is reached only when
`build_system_prompt` is handed an explicitly empty toolset (which tests do),
not in a normal live session (which now always has ≥1 tool).

The committed-but-dormant `weather_tool.py` is **left untouched** by this change.

## Error handling

The tool is best-effort and **never raises** into the graph — a fetch failure
must not trip `brain`'s "the agent couldn't complete the turn" path or sink a
live turn. `fetch_article` raises `UsageError` (not an http(s) URL, or no
readable text — e.g. a scanned/image-only PDF or a paywalled/JS page) and
`APIError` (DNS/timeout/non-2xx), both `CLIError`. `read_url` catches its own
failures and returns a short speakable string instead:

- No readable text / bad URL (`UsageError`) → *"I couldn't find readable text on
  that page."*
- Fetch failed (`APIError`, or any other exception) → *"I couldn't read that
  page right now."*

## Security note (accepted)

An un-gated URL fetch can reach internal/SSRF targets. The **coding** agent
gates its `fetch_url` for exactly this reason, but it can pause for keyboard
approval; a spoken live turn cannot. Live therefore auto-approves read-only
tools — and already exposes an un-gated web-search tool that returns content
from arbitrary URLs — so reading a URL is consistent with the existing posture,
not a new class of exposure. Recorded here as a known, accepted trade-off rather
than a gate.

## Testing

Targets the gate's 100% patch-coverage + diff-scoped mutation requirements:
assertions must *fail* if a changed line breaks, not merely execute it. All
tests are hermetic — no real network — via the injected `read` seam, in keeping
with the rest of the cascade's STT/LLM/TTS fakes.

- `_format` tested directly:
  - title present → leads with the title, then the body.
  - title absent → body only.
  - body over `_MAX_CHARS` → truncated to the cap with the `…[truncated]` marker;
    a short body is returned untruncated.
- The tool driven end-to-end with a fake `read`:
  - Happy path: canned `Article` → `_format`'s output.
  - `UsageError` raised → the "couldn't find readable text" message.
  - `APIError` raised → the "couldn't read that page" message.
- `brain` wiring:
  - `build_live_tools()` includes a tool named `READ_URL_TOOL_NAME` (and still
    includes web search when keyed).
  - `_tool_capabilities()` / `build_system_prompt` advertises the read-url
    capability (and both capabilities together when search is also present).
  - `_tool_label(READ_URL_TOOL_NAME)` returns "Reading the page".
