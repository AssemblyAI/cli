# Read-a-URL tool (web pages + PDFs) for `assembly live` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `assembly live` voice agent a keyless, always-present `read_url` tool that fetches a web page or PDF by URL and returns its readable text.

**Architecture:** A new `aai_cli/agent_cascade/webpage_tool.py` wraps the existing `core/webpage.py:fetch_article` (HTML via trafilatura, PDF via pypdf — the same reader behind `assembly speak --url`) as a LangChain `BaseTool`, mirroring the sibling `weather_tool.py` (pure helpers + one injected network seam + best-effort error handling). It is wired into the live deepagents graph through the three existing tool hooks in `brain.py`.

**Tech Stack:** Python 3.12+, LangChain `@tool`, deepagents, pytest. Tests are hermetic via the injected `read` seam (pytest-socket stays armed).

## Global Constraints

- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- Internal helper docstrings keep trailing periods (mutation gate flags a changed docstring line); the `read_url` tool's own docstring is its model-facing description.
- **Lazy-import the heavy reader.** `core/webpage.py` imports `httpx2` at module top, so `webpage_tool.py` must NOT import `fetch_article` at module scope — defer it inside `_read` (mirrors `weather_tool._get_json` deferring `httpx`). Keep `Article` under `TYPE_CHECKING` only.
- The single network seam is an injected `Reader` callable; tests pass a fake — no sockets, no real `fetch_article`.
- The tool is **best-effort and never raises** into the graph (a spoken turn can't surface a traceback).
- Gate rules: iterate with targeted `uv run pytest`, then run `./scripts/check.sh` to completion before the final commit (it must print `All checks passed.`). The branch `live-tool-call-ux` carries **unrelated in-flight working-tree changes** — `git add` only this feature's files for each commit, and use `AAI_ALLOW_COMMIT=1 git commit …` for the intermediate (Task 1) commit since the full tree can't be cleanly gated mid-flight.

---

### Task 1: The `webpage_tool.py` module (read_url tool, standalone)

**Files:**
- Create: `aai_cli/agent_cascade/webpage_tool.py`
- Test: `tests/test_agent_cascade_webpage.py`

**Interfaces:**
- Consumes: `aai_cli.core.webpage.fetch_article` (`(url: str) -> Article`, `Article` has `.text: str` and `.title: str | None`); `aai_cli.core.errors.UsageError` (raised for a non-http(s) URL or no readable text), `APIError` (raised for fetch failure) — both `CLIError` subclasses.
- Produces (later tasks rely on these exact names):
  - `READ_URL_TOOL_NAME = "read_url"` (str constant)
  - `Reader = Callable[[str], Article]` (type alias)
  - `build_read_url_tool(read: Reader = _read) -> BaseTool` — returns a tool named `read_url` taking `url: str` returning `str`
  - `_format(article: Article) -> str`, `_read(url: str) -> Article` (module-private)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_cascade_webpage.py`:

```python
"""Tests for the keyless read-a-URL tool behind `assembly live`.

The tool's only network seam is the injected ``read`` callable, so the whole
fetch -> format flow runs with no sockets (pytest-socket stays armed).
"""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade import webpage_tool
from aai_cli.core.errors import APIError, UsageError
from aai_cli.core.webpage import Article


def _article(text: str = "Body text.", title: str | None = "Title") -> Article:
    return Article(text=text, title=title, url="https://example.com/post")


# --- _format -----------------------------------------------------------------


def test_format_leads_with_title_then_body():
    out = webpage_tool._format(_article(text="Hello world.", title="My Post"))
    assert out == "My Post\n\nHello world."


def test_format_without_title_is_body_only():
    out = webpage_tool._format(_article(text="Just the body.", title=None))
    assert out == "Just the body."


def test_format_truncates_long_body_with_marker():
    long = "x" * (webpage_tool._MAX_CHARS + 50)
    out = webpage_tool._format(_article(text=long, title=None))
    assert out == "x" * webpage_tool._MAX_CHARS + "\n…[truncated]"


def test_format_keeps_short_body_untruncated():
    out = webpage_tool._format(_article(text="short", title=None))
    assert "[truncated]" not in out
    assert out == "short"


# --- _read (default seam delegates to core.webpage.fetch_article) ------------


def test_read_delegates_to_fetch_article(monkeypatch):
    captured = {}

    def fake_fetch_article(url: str) -> Article:
        captured["url"] = url
        return _article()

    monkeypatch.setattr("aai_cli.core.webpage.fetch_article", fake_fetch_article)
    result = webpage_tool._read("https://example.com/post")
    assert captured["url"] == "https://example.com/post"
    assert result.title == "Title"


# --- build_read_url_tool -----------------------------------------------------


def test_tool_is_named_read_url():
    tool = webpage_tool.build_read_url_tool(read=lambda url: _article())
    assert tool.name == webpage_tool.READ_URL_TOOL_NAME


def test_read_url_happy_path_returns_formatted_text():
    tool = webpage_tool.build_read_url_tool(read=lambda url: _article(text="Article.", title="T"))
    assert tool.invoke({"url": "https://example.com"}) == "T\n\nArticle."


def test_read_url_usage_error_returns_no_readable_text_message():
    def read(url: str) -> Article:
        raise UsageError("Couldn't find readable text.")

    tool = webpage_tool.build_read_url_tool(read=read)
    assert tool.invoke({"url": "https://example.com"}) == (
        "I couldn't find readable text on that page."
    )


def test_read_url_fetch_failure_returns_could_not_read_message():
    def read(url: str) -> Article:
        raise APIError("DNS boom")

    tool = webpage_tool.build_read_url_tool(read=read)
    assert tool.invoke({"url": "https://example.com"}) == (
        "I couldn't read that page right now."
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_webpage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.agent_cascade.webpage_tool'` (collection error).

- [ ] **Step 3: Write the module**

Create `aai_cli/agent_cascade/webpage_tool.py`:

```python
"""A keyless read-a-URL tool for the `assembly live` voice agent.

Reads a web page or PDF the agent has a URL for and returns its readable text, so
the live agent can read an article the user names or a link surfaced by web search.
It reuses :func:`aai_cli.core.webpage.fetch_article` — the same trafilatura HTML
extraction and pypdf PDF text extraction that backs ``assembly speak --url`` — so no
API key is needed and every live session has this capability.

The only network seam is :data:`Reader` (a ``url -> Article`` callable), injected in
tests so the whole flow runs with no sockets — the same shape ``weather_tool`` uses.
Failures never raise out to the graph: ``read_url`` catches them and returns a short
spoken apology so a fetch outage can't sink a live turn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aai_cli.core.errors import UsageError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from aai_cli.core.webpage import Article

# The registered tool name. ``brain.py`` detects availability and labels the live-UI
# affordance by this name, so a test pins it.
READ_URL_TOOL_NAME = "read_url"

# A reader GETs a URL and returns the extracted Article. Injected in tests (the only net seam).
Reader = Callable[[str], "Article"]

# Cap the returned text so a long article or multi-page PDF can't blow the model's context
# budget. The body is source for the model to summarize aloud, so the exact cap is a tuning
# knob — a +-1 shift is behaviorally equivalent, so no test can kill that mutant.
_MAX_CHARS = 16000  # pragma: no mutate


def _read(url: str) -> Article:
    """Fetch and extract ``url`` via core.webpage (imported lazily to stay off startup)."""
    from aai_cli.core.webpage import fetch_article

    return fetch_article(url)


def _format(article: Article) -> str:
    """Render the article as ``title + readable text``, truncated to ``_MAX_CHARS``."""
    body = article.text
    if len(body) > _MAX_CHARS:
        body = body[:_MAX_CHARS] + "\n…[truncated]"
    if article.title:
        return f"{article.title}\n\n{body}"
    return body


def build_read_url_tool(read: Reader = _read) -> BaseTool:
    """Wrap the URL reader as the ``read_url`` tool (``read`` injectable for tests)."""
    from langchain_core.tools import tool

    @tool(READ_URL_TOOL_NAME)
    def read_url(url: str) -> str:
        """Read a web page or PDF by URL and return its text. Use to read an article,
        document, or page you have the URL for (e.g. from a web-search result)."""
        try:
            return _format(read(url))
        except UsageError:
            # Bad URL or no readable text (scanned/image-only PDF, paywalled/JS page).
            return "I couldn't find readable text on that page."
        except Exception:
            # Any fetch failure (APIError: DNS/timeout/non-2xx, or anything else) must not
            # bubble into brain's "couldn't complete the turn" path. Mirrors weather_tool.
            return "I couldn't read that page right now."

    return read_url
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_webpage.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/webpage_tool.py tests/test_agent_cascade_webpage.py
AAI_ALLOW_COMMIT=1 git commit -m "feat: read-a-URL (web + PDF) tool module for assembly live"
```

---

### Task 2: Wire `read_url` into the live agent (`brain.py`)

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (import line ~24, `_TOOL_LABELS`, `_tool_capabilities`, `build_live_tools`)
- Test: `tests/test_agent_cascade_brain.py` (update two existing `build_live_tools` tests; add three wiring tests)

**Interfaces:**
- Consumes from Task 1: `webpage_tool.READ_URL_TOOL_NAME`, `webpage_tool.build_read_url_tool`.
- Produces: `build_live_tools()` returns `[weather, read_url, (web_search if keyed)]` in that order; `_tool_label("read_url")` returns `"Reading the page"`; `build_system_prompt` advertises reading a page/PDF.

- [ ] **Step 1: Update the two existing `build_live_tools` tests + add wiring tests (failing)**

In `tests/test_agent_cascade_brain.py`, change the import (line ~18) from:

```python
from aai_cli.agent_cascade import brain, weather_tool
```

to:

```python
from aai_cli.agent_cascade import brain, weather_tool, webpage_tool
```

Replace the existing `test_build_live_tools_has_weather_and_web_search_when_keyed` body (lines ~378-384) so it also asserts read_url:

```python
def test_build_live_tools_has_weather_and_web_search_when_keyed(monkeypatch):
    search = _NamedTool(brain.WEB_SEARCH_TOOL_NAME)
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: search)
    names = [tool.name for tool in brain.build_live_tools()]
    # Web search is the optional keyed leg; the keyless weather + read-url tools are always present.
    assert brain.WEB_SEARCH_TOOL_NAME in names
    assert weather_tool.WEATHER_TOOL_NAME in names
    assert webpage_tool.READ_URL_TOOL_NAME in names
```

Replace `test_build_live_tools_is_just_weather_without_firecrawl_key` (lines ~387-391) with the keyless-pair assertion (renamed):

```python
def test_build_live_tools_has_weather_and_read_url_without_firecrawl_key(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: None)
    # No FIRECRAWL_API_KEY -> no web search, but the keyless weather + read-url tools still load.
    names = [tool.name for tool in brain.build_live_tools()]
    assert names == [weather_tool.WEATHER_TOOL_NAME, webpage_tool.READ_URL_TOOL_NAME]
```

Add three new tests at the end of the file (after `test_tool_label_maps_weather`):

```python
def test_read_url_tool_advertised_in_system_prompt():
    prompt = brain.build_system_prompt(
        "persona", tools=[_NamedTool(webpage_tool.READ_URL_TOOL_NAME)]
    )
    assert "read a web page or PDF" in prompt


def test_tool_label_maps_read_url():
    assert brain._tool_label(webpage_tool.READ_URL_TOOL_NAME) == "Reading the page"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: FAIL — `test_build_live_tools_has_weather_and_read_url_without_firecrawl_key` (read_url not yet built), `test_read_url_tool_advertised_in_system_prompt` (phrase absent), `test_tool_label_maps_read_url` (falls back to `"Using read_url"`), and the keyed test's new `in` assertion.

- [ ] **Step 3: Wire `brain.py` — import**

Change the import line (~24) from:

```python
from aai_cli.agent_cascade import weather_tool
```

to:

```python
from aai_cli.agent_cascade import weather_tool, webpage_tool
```

- [ ] **Step 4: Wire `brain.py` — `_TOOL_LABELS`**

Add the read_url label to the `_TOOL_LABELS` dict (after the weather entry):

```python
_TOOL_LABELS = {
    WEB_SEARCH_TOOL_NAME: "Searching the web",
    weather_tool.WEATHER_TOOL_NAME: "Checking the weather",
    webpage_tool.READ_URL_TOOL_NAME: "Reading the page",
}
```

- [ ] **Step 5: Wire `brain.py` — `_tool_capabilities`**

In `_tool_capabilities`, after the weather `if` block, add the read-url capability so it is advertised when present:

```python
    if weather_tool.WEATHER_TOOL_NAME in names:
        capabilities.append("tell someone the current weather and short forecast for a place")
    if webpage_tool.READ_URL_TOOL_NAME in names:
        capabilities.append("read a web page or PDF you have the URL for")
    return capabilities
```

- [ ] **Step 6: Wire `brain.py` — `build_live_tools`**

In `build_live_tools`, add the lazy import and include the read-url tool in the always-present list. The body becomes:

```python
    from aai_cli.agent_cascade.weather_tool import build_weather_tool
    from aai_cli.agent_cascade.webpage_tool import build_read_url_tool
    from aai_cli.code_agent.firecrawl_search import build_web_search_tool

    tools: list[BaseTool] = [build_weather_tool(), build_read_url_tool()]
    search = build_web_search_tool()
    if search is not None:
        tools.append(search)
    return tools
```

Also update its docstring's first sentence to name the new tool, e.g. change *"the keyless weather tool, plus Firecrawl web search…"* to *"the keyless weather and read-a-URL tools, plus Firecrawl web search when ``FIRECRAWL_API_KEY`` is set."*

- [ ] **Step 7: Run the brain tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py tests/test_agent_cascade_webpage.py -q`
Expected: PASS (all green).

- [ ] **Step 8: Run the full gate**

Run: `./scripts/check.sh`
Expected: finishes with `All checks passed.` (covers ruff, mypy/pyright, vulture, import-linter, 100% patch coverage vs origin/main, and the diff-scoped mutation gate on the changed lines). Fix any finding before committing.

- [ ] **Step 9: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
git commit -m "feat: wire read_url tool into assembly live"
```

---

## Self-Review

**Spec coverage:**
- New module `webpage_tool.py` reusing `fetch_article` (HTML + PDF) → Task 1.
- `READ_URL_TOOL_NAME`, injected `Reader` seam, `_MAX_CHARS` truncation, `_format`, `build_read_url_tool` → Task 1 Step 3.
- Best-effort error handling (UsageError vs other → two apology strings) → Task 1 Steps 1 & 3.
- Always-present (keyless) wiring in `build_live_tools` → Task 2 Step 6.
- `_tool_capabilities` advertises read-url → Task 2 Step 5.
- `_TOOL_LABELS` "Reading the page" → Task 2 Step 4.
- Weather tool left untouched → confirmed (no weather edits in any task).
- Testing matrix (format title/no-title/truncation/short, happy, UsageError, APIError, brain wiring) → Tasks 1 & 2 test steps.
- Security note (no approval gate) → design-doc only; no code, correctly nothing to implement.

**Placeholder scan:** none — every step carries full code/commands.

**Type consistency:** `READ_URL_TOOL_NAME`, `Reader`, `build_read_url_tool(read=…)`, `_format`, `_read` names match between Task 1 (definition) and Task 2 (consumption); `build_live_tools` order `[weather, read_url, search?]` matches the updated keyless test assertion.
