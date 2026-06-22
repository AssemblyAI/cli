# `assembly live` Streaming Reply Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap LLM token generation, TTS synthesis, and playback in a live cascade turn so audio starts on the first clause instead of after the whole reply is generated.

**Architecture:** The reply leg becomes a stream. `brain.build_streamer` drives the deepagents graph with `stream_mode="messages"` and yields `SpeechDelta`/`ToolNotice` events. The engine consumes them on the reply-worker thread (fed by a throwaway daemon producer thread + a `queue.Queue` that preserves today's wall-clock timeout), buffers token deltas, flushes complete clauses via a new `text.pop_clauses`, and synthesizes each clause with **streaming TTS** (`tts_session.synthesize(..., on_audio=...)`) so playback begins on its first audio frame.

**Tech Stack:** Python 3.12–3.13, deepagents/langgraph (`graph.stream`), `langchain_core` message chunks, the streaming-TTS WebSocket in `aai_cli/tts/session.py`, `threading`/`queue`, pytest + syrupy.

## Global Constraints

Copied verbatim from the repo invariants (every task's requirements include these):

- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- Errors → stderr, data → stdout. Help/option copy is terse, imperative, sentence-case, **no trailing period** (not relevant here — no new flags — but keep docstrings periodful).
- **The gate is the source of truth.** `./scripts/check.sh` must print `All checks passed.` Notable diff-scoped gates: **100% patch coverage** vs `origin/main`, a **diff-scoped mutation gate** (a changed line needs a test that *fails* if the line breaks — not just coverage), **vulture** (no dead code — flags both unused new functions *and* code that becomes unused), **xenon** (function complexity ≤ B), the **500-line max file length**, and a **"no new escape hatches"** count gate (`# pragma: no mutate` / `# noqa` / `pragma: no cover` / `cast(` / `Any` counted against merge-base — a *net-new* one fails). The **Textual-module ≥90% coverage floor** also applies, but this change does not touch `tui.py`.
- **Commit discipline:** iterate with fast targeted `uv run pytest …`, then gate once at the end. Use `AAI_ALLOW_COMMIT=1 git commit …` for intermediate per-task WIP commits; the **final** commit must follow a full green `./scripts/check.sh` (the PreToolUse hook enforces this). End every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Run every tool through `uv run`.
- **Escape-hatch budget is net-neutral by design:** Task 3 removes `_complete_within`'s `daemon=True  # pragma: no mutate`; the new producer thread adds exactly one back. `_MIN_CLAUSE_CHARS` is pinned by a test (Task 1/3), so it needs **no** pragma. Do not introduce other new pragmas.
- **Do not touch `uv.lock`** — no dependency changes in this work.

---

## File Structure

- `aai_cli/agent_cascade/text.py` — add `pop_clauses` (pure incremental clause splitter). Keep `split_sentences`/`trim_history`.
- `aai_cli/agent_cascade/brain.py` — add `SpeechDelta`/`ToolNotice` + `build_streamer` + `_stream_graph` (messages-mode iteration with verbose logging). Task 4 removes the now-dead `build_completer`/`_run_graph`/`_drive_graph`/`_log_flow`/`_surface_event`/`_reply_text`. Keep `_clip`/`_tool_label`/`_content_text`/`build_graph`/`build_system_prompt`/`build_live_tools`.
- `aai_cli/agent_cascade/engine.py` — change `CascadeDeps` seam (`complete_reply: str` → `stream_reply: Iterable[event]`; `synthesize: (str)->bytes` → `synthesize: (str, sink)->None`), rewrite `greet`/`_generate_reply`, add producer/queue/clause helpers, remove `_complete_within`.
- `aai_cli/AGENTS.md` — update the `agent_cascade/` bullet describing the `-v` behavior and "per-sentence TTS".
- `tests/_cascade_fakes.py` — `make_session` seam: `stream_reply` + streaming `synthesize`.
- `tests/test_agent_cascade_engine.py` — rewrite reply-generation/timeout tests for the streaming seam.
- `tests/test_agent_cascade_brain.py` — add `build_streamer` tests (Task 2); remove `build_completer`/`_run_graph` tests (Task 4).
- `tests/test_agent_cascade_command.py` — update the two `CascadeDeps.real` leg tests + the `fake_real` constructions.

---

## Task 1: `pop_clauses` incremental clause splitter

**Files:**
- Modify: `aai_cli/agent_cascade/text.py`
- Test: `tests/test_agent_cascade_text.py` (new file)

**Interfaces:**
- Produces: `pop_clauses(buffer: str, *, min_chars: int) -> tuple[list[str], str]` — returns complete speakable clauses pulled off the front of `buffer` plus the unflushed remainder. Hard terminators `.!?` (followed by whitespace) always end a clause; soft separators `,;:` (followed by whitespace) end one only when the pending clause is at least `min_chars` long.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_cascade_text.py`:

```python
"""Tests for the cascade's pure text helpers (sentence/clause splitting)."""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade.text import pop_clauses


def test_pop_clauses_flushes_hard_terminators_and_keeps_tail():
    chunks, remainder = pop_clauses("One. Two! Three", min_chars=1)
    assert chunks == ["One.", "Two!"]
    assert remainder == " Three"  # no terminator yet -> stays buffered


def test_pop_clauses_flushes_soft_separator_only_past_min_chars():
    # The clause before the comma is long enough, so the comma ends a clause.
    chunks, remainder = pop_clauses("the weather today is, in fact ", min_chars=10)
    assert chunks == ["the weather today is,"]
    assert remainder == " in fact "


def test_pop_clauses_holds_short_soft_clause_to_avoid_choppy_tts():
    # "Yes," is shorter than min_chars, so it is NOT flushed on the comma.
    chunks, remainder = pop_clauses("Yes, it is sunny", min_chars=10)
    assert chunks == []
    assert remainder == "Yes, it is sunny"


def test_pop_clauses_does_not_fragment_a_decimal_or_stacked_terminators():
    # A '.' inside $3.50 (no following space) and stacked '...'/'?!' are not boundaries.
    chunks, remainder = pop_clauses("It costs $3.50 total... ", min_chars=1)
    assert chunks == ["It costs $3.50 total..."]
    assert remainder == " "


def test_pop_clauses_returns_nothing_for_an_unterminated_buffer():
    chunks, remainder = pop_clauses("still going", min_chars=1)
    assert chunks == []
    assert remainder == "still going"


def test_pop_clauses_strips_whitespace_from_each_flushed_clause():
    chunks, _remainder = pop_clauses("  Hi there.  Next.", min_chars=1)
    assert chunks == ["Hi there.", "Next."]


@pytest.mark.parametrize("min_chars", [1, 25])
def test_pop_clauses_flushes_hard_terminator_regardless_of_min_chars(min_chars):
    # min_chars only gates SOFT separators; a sentence terminator always flushes.
    chunks, remainder = pop_clauses("Hi. ", min_chars=min_chars)
    assert chunks == ["Hi."]
    assert remainder == " "
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_text.py -q`
Expected: FAIL — `ImportError: cannot import name 'pop_clauses'`.

- [ ] **Step 3: Implement `pop_clauses`**

Add to `aai_cli/agent_cascade/text.py` (below `_TERMINATORS`):

```python
# Soft clause separators: a comma/semicolon/colon ends a *speakable* chunk too, but only
# once the pending clause is long enough (see pop_clauses) — flushing "Yes," on its own
# makes choppy TTS. Hard terminators (_TERMINATORS) always end a clause.
_SOFT_SEPARATORS = ",;:"


def _is_boundary(text: str, index: int) -> bool:
    """True when the char at ``index`` ends a clause: a terminator/separator that is the
    last char or is followed by whitespace (so a '.' inside "$3.50" never splits)."""
    return index + 1 == len(text) or text[index + 1].isspace()


def pop_clauses(buffer: str, *, min_chars: int) -> tuple[list[str], str]:
    """Pull complete speakable clauses off the front of ``buffer`` for incremental TTS.

    A hard terminator (``.``/``!``/``?``) followed by whitespace (or end-of-buffer) always
    ends a clause; a soft separator (``,``/``;``/``:``) ends one only when the clause built
    since the last boundary is at least ``min_chars`` long, so a tiny fragment ("Yes,")
    isn't synthesized on its own. Returns the flushed clauses (each stripped, never blank)
    and the still-incomplete remainder to keep buffering. The caller flushes the final tail
    at end-of-stream.
    """
    clauses: list[str] = []
    start = 0
    for index, char in enumerate(buffer):
        is_hard = char in _TERMINATORS
        is_soft = char in _SOFT_SEPARATORS
        if not (is_hard or is_soft) or not _is_boundary(buffer, index):
            continue
        clause = buffer[start : index + 1].strip()
        if is_soft and len(clause) < min_chars:
            continue  # too short to speak on its own — keep accumulating
        if clause:
            clauses.append(clause)
        start = index + 1
    return clauses, buffer[start:]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_text.py -q`
Expected: PASS (all 8 cases).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/text.py tests/test_agent_cascade_text.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add pop_clauses incremental clause splitter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: brain — `build_streamer` reply event stream (added alongside `build_completer`)

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py`
- Test: `tests/test_agent_cascade_brain.py`

**Interfaces:**
- Produces:
  - `class SpeechDelta` — frozen dataclass, field `text: str` (a top-level assistant-text token delta).
  - `class ToolNotice` — frozen dataclass, field `label: str` (the speakable tool affordance label).
  - `build_streamer(api_key: str, config: CascadeConfig, *, graph: CompiledAgent | None = None) -> Callable[[list[ChatCompletionMessageParam]], Iterator[SpeechDelta | ToolNotice]]`. The returned `stream_reply(messages)` drops the prepended `system` message, then iterates `graph.stream({"messages": conversation}, None, stream_mode="messages")` yielding `SpeechDelta`/`ToolNotice`. Graph exceptions wrap into `CLIError` (`agent_brain_error`); a `CLIError` passes through unchanged. Under `-v` it logs accumulated assistant text, tool calls, and tool results to `_FLOW_LOG`.
- Consumes: existing `_tool_label`, `_clip`, `_content_text`, `build_graph`, `debuglog`, `CLIError`.

> NOTE: `build_completer` and its helpers stay untouched in this task (still used by `engine`); Task 4 removes them once the engine has switched. The new `build_streamer` is referenced by the tests below, so vulture sees it as used.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_cascade_brain.py` (new imports at top: `from langchain_core.messages import AIMessageChunk`; reuse existing `AIMessage`/`ToolMessage`/`logging`/`pytest`):

```python
# --- build_streamer (token streaming -> SpeechDelta / ToolNotice) ------------


class _MessageStreamGraph:
    """A graph whose .stream yields (message_chunk, metadata) pairs — the shape
    langgraph emits under stream_mode='messages'. Records the stream_mode it saw."""

    def __init__(self, items):
        self._items = items
        self.stream_mode = None

    def stream(self, graph_input, config, *, stream_mode):
        del graph_input, config
        self.stream_mode = stream_mode
        yield from self._items


def _collect(graph, messages, **kwargs):
    streamer = brain.build_streamer("k", CascadeConfig(), graph=graph)
    return list(streamer(messages, **kwargs)) if kwargs else list(streamer(messages))


def test_streamer_yields_speech_deltas_for_assistant_tokens():
    graph = _MessageStreamGraph(
        [
            (AIMessageChunk(content="Hello "), {}),
            (AIMessageChunk(content="there."), {}),
        ]
    )
    events = _collect(graph, [{"role": "user", "content": "hi"}])
    assert [e.text for e in events if isinstance(e, brain.SpeechDelta)] == ["Hello ", "there."]
    assert graph.stream_mode == "messages"


def test_streamer_strips_system_message_before_streaming():
    captured = {}

    class _Capture(_MessageStreamGraph):
        def stream(self, graph_input, config, *, stream_mode):
            captured["roles"] = [m["role"] for m in graph_input["messages"]]
            return super().stream(graph_input, config, stream_mode=stream_mode)

    graph = _Capture([(AIMessageChunk(content="ok"), {})])
    _collect(graph, [{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}])
    assert captured["roles"] == ["user"]


def test_streamer_emits_a_tool_notice_when_a_tool_call_starts():
    call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": brain.WEB_SEARCH_TOOL_NAME, "args": "", "id": "c1", "index": 0}
        ],
    )
    graph = _MessageStreamGraph(
        [(call_chunk, {}), (AIMessageChunk(content="Here it is."), {})]
    )
    events = _collect(graph, [{"role": "user", "content": "news?"}])
    notices = [e.label for e in events if isinstance(e, brain.ToolNotice)]
    deltas = [e.text for e in events if isinstance(e, brain.SpeechDelta)]
    assert notices == ["Searching the web"]
    assert deltas == ["Here it is."]


def test_streamer_emits_one_notice_per_call_ignoring_arg_only_chunks():
    # The first tool-call chunk carries the name; later arg-only chunks (name=None) must NOT
    # re-fire the affordance.
    first = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "get_time", "args": "", "id": "c1", "index": 0}]
    )
    rest = AIMessageChunk(
        content="", tool_call_chunks=[{"name": None, "args": '{"tz":1}', "id": "c1", "index": 0}]
    )
    graph = _MessageStreamGraph([(first, {}), (rest, {})])
    events = _collect(graph, [{"role": "user", "content": "time?"}])
    assert [e.label for e in events if isinstance(e, brain.ToolNotice)] == ["Using get_time"]


def test_streamer_wraps_graph_errors_in_cli_error():
    class _Boom:
        def stream(self, graph_input, config, *, stream_mode):
            del graph_input, config, stream_mode
            raise ValueError("gateway said no")
            yield  # pragma: no cover  (make it a generator)

    streamer = brain.build_streamer("k", CascadeConfig(), graph=_Boom())
    with pytest.raises(CLIError) as excinfo:
        list(streamer([{"role": "user", "content": "hi"}]))
    assert "couldn't complete the turn" in excinfo.value.message
    assert "gateway said no" in excinfo.value.message


def test_streamer_passes_cli_error_through():
    class _CliBoom:
        def stream(self, graph_input, config, *, stream_mode):
            del graph_input, config, stream_mode
            raise CLIError("already clean", error_type="x")
            yield  # pragma: no cover

    streamer = brain.build_streamer("k", CascadeConfig(), graph=_CliBoom())
    with pytest.raises(CLIError, match="already clean"):
        list(streamer([{"role": "user", "content": "hi"}]))


def test_streamer_logs_flow_when_verbose(monkeypatch, caplog, preserve_logging_state):
    monkeypatch.setattr(brain.debuglog, "active", lambda: True)
    call_chunk = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "tavily_search", "args": "", "id": "c1", "index": 0}]
    )
    items = [
        (AIMessageChunk(content="Let me "), {}),
        (AIMessageChunk(content="search."), {}),
        (call_chunk, {}),
        (ToolMessage(content="rainy, 52F", name="tavily_search", tool_call_id="c1"), {}),
        (AIMessageChunk(content="It's rainy."), {}),
    ]
    graph = _MessageStreamGraph(items)
    with caplog.at_level(logging.INFO, logger="aai_cli.agent_cascade.brain"):
        _collect(graph, [{"role": "user", "content": "weather?"}])
    messages = [r.getMessage() for r in caplog.records]
    # Accumulated assistant text is logged as one line per assistant turn, around the
    # tool call and its result.
    assert messages == [
        "llm: Let me search.",
        "tool call tavily_search",
        "tool result tavily_search -> rainy, 52F",
        "llm: It's rainy.",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k streamer -q`
Expected: FAIL — `AttributeError: module 'aai_cli.agent_cascade.brain' has no attribute 'build_streamer'`.

- [ ] **Step 3: Add the event dataclasses**

In `aai_cli/agent_cascade/brain.py`, add `from dataclasses import dataclass` to the imports, and define (just below `_TOOL_LABELS`/`_tool_label`):

```python
@dataclass(frozen=True)
class SpeechDelta:
    """A top-level assistant-text token delta to be spoken (one piece of the reply)."""

    text: str


@dataclass(frozen=True)
class ToolNotice:
    """A speakable affordance label emitted when the agent starts a tool call mid-turn."""

    label: str
```

- [ ] **Step 4: Add `build_streamer` and `_stream_graph`**

Add `Iterator` to the `collections.abc` import (`from collections.abc import Callable, Iterator, Sequence`). Add these functions to `brain.py` (place near `build_completer`):

```python
def build_streamer(
    api_key: str, config: CascadeConfig, *, graph: CompiledAgent | None = None
) -> Callable[..., Iterator[SpeechDelta | ToolNotice]]:
    """A streaming reply leg for the cascade engine, backed by the deepagents graph.

    The cascade prepends its own ``system`` message each turn; the graph owns the system
    prompt, so it is dropped before streaming. The graph is driven with
    ``stream_mode="messages"`` and each top-level assistant token delta is yielded as a
    :class:`SpeechDelta`, each started tool call as a :class:`ToolNotice` (the live UI's
    affordance). Under ``-v`` the flow is logged. ``graph`` is injected in tests so the
    per-turn wiring runs against a fake with no network.
    """
    resolved = build_graph(api_key, config) if graph is None else graph

    def stream_reply(
        messages: list[ChatCompletionMessageParam],
    ) -> Iterator[SpeechDelta | ToolNotice]:
        conversation = [message for message in messages if message.get("role") != "system"]
        return _stream_graph(resolved, conversation)

    return stream_reply


def _stream_graph(
    graph: CompiledAgent, conversation: list[ChatCompletionMessageParam]
) -> Iterator[SpeechDelta | ToolNotice]:
    """Stream one turn through the graph token-by-token, yielding speech/tool events.

    Wraps any graph failure as a CLIError (a clean ``CLIError`` passes through) so the
    cascade surfaces it instead of the reply worker dying silently — the same contract the
    old ``_run_graph`` had. Under ``-v`` the accumulated assistant text, each tool call,
    and each tool result are logged to ``_FLOW_LOG``.
    """
    verbose = debuglog.active()
    pending: list[str] = []  # assistant deltas accumulated for one verbose "llm:" line

    def flush_log() -> None:
        if verbose and pending:
            _FLOW_LOG.info("llm: %s", "".join(pending))
        pending.clear()

    try:
        for chunk, _meta in graph.stream({"messages": conversation}, None, stream_mode="messages"):
            yield from _events_from_chunk(chunk, verbose, pending, flush_log)
        flush_log()
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(
            f"the agent couldn't complete the turn: {exc}", error_type="agent_brain_error"
        ) from exc


def _events_from_chunk(
    chunk: object, verbose: bool, pending: list[str], flush_log: Callable[[], None]
) -> Iterator[SpeechDelta | ToolNotice]:
    """Translate one streamed message chunk into speech/tool events (and verbose logs)."""
    if type(chunk).__name__ == "ToolMessage":
        flush_log()
        if verbose:
            content = _content_text(getattr(chunk, "content", ""))
            _FLOW_LOG.info("tool result %s -> %s", getattr(chunk, "name", ""), _clip(content))
        return
    for call in getattr(chunk, "tool_call_chunks", None) or []:
        name = call.get("name")
        if name:
            flush_log()
            if verbose:
                _FLOW_LOG.info("tool call %s", name)
            yield ToolNotice(_tool_label(name))
    text = _content_text(getattr(chunk, "content", ""))
    if text:
        pending.append(text)
        yield SpeechDelta(text)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k streamer -q`
Expected: PASS (7 cases).

- [ ] **Step 6: Run the whole brain suite (build_completer still present and green)**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: PASS (old `build_completer`/`_run_graph` tests untouched).

- [ ] **Step 7: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): add build_streamer token-streaming reply leg

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: engine — streaming seam + `_generate_reply` rewrite

**Files:**
- Modify: `aai_cli/agent_cascade/engine.py`
- Modify: `tests/_cascade_fakes.py`
- Modify: `tests/test_agent_cascade_engine.py`
- Modify: `tests/test_agent_cascade_command.py`

**Interfaces:**
- Consumes: `brain.build_streamer`, `brain.SpeechDelta`, `brain.ToolNotice` (Task 2); `text.pop_clauses` (Task 1).
- Produces (new `CascadeDeps` shape):
  - `stream_reply: Callable[..., Iterable[SpeechDelta | ToolNotice]]` — `(messages) -> iterable of reply events`.
  - `synthesize: Callable[[str, Callable[[bytes], None]], None]` — `(text, sink)`; calls `sink(pcm)` per audio frame.
  - `run_stt`, `spawn` unchanged.
- `CascadeDeps.real(api_key, config, *, audio, stt_params)` signature is **unchanged** (so `commands/agent_cascade/_exec.py` needs no edit) — only the legs it builds change.

- [ ] **Step 1: Update the shared fakes**

In `tests/_cascade_fakes.py`, replace `make_session` (and only it) so the seam matches. New `make_session`:

```python
def make_session(
    *,
    stream_reply=None,
    synthesize=lambda text, sink: sink(b"pcm:" + text.encode()),
    spawn=sync_spawn,
    run_stt=lambda on_turn: None,
    config=None,
):
    from aai_cli.agent_cascade.brain import SpeechDelta

    if stream_reply is None:
        stream_reply = lambda messages: [SpeechDelta("Hello there.")]
    deps = CascadeDeps(
        run_stt=run_stt, stream_reply=stream_reply, synthesize=synthesize, spawn=spawn
    )
    renderer = FakeRenderer()
    player = FakePlayer()
    session = CascadeSession(
        deps=deps, renderer=renderer, player=player, config=config or CascadeConfig()
    )
    return session, renderer, player
```

Add a small helper just below it so tests can script a reply as a list of deltas/notices:

```python
def deltas(*texts):
    """A stream_reply that yields the given strings as SpeechDelta events."""
    from aai_cli.agent_cascade.brain import SpeechDelta

    return lambda messages: [SpeechDelta(t) for t in texts]
```

- [ ] **Step 2: Rewrite the engine's `CascadeDeps`, `greet`, and reply path tests**

Replace the reply-generation/timeout/greeting tests in `tests/test_agent_cascade_engine.py`. Delete `test_complete_within_*` (3 tests) and rewrite the reply tests against the streaming seam. New/changed tests (keep the barge-in/shutdown/`_is_final_turn`/`run_cascade` tests but update their `CascadeDeps(...)` constructions — see Step 7):

```python
from tests._cascade_fakes import deltas as _deltas
from aai_cli.agent_cascade.brain import SpeechDelta, ToolNotice


def test_greet_speaks_and_seeds_history():
    session, renderer, player = make_session()
    session.greet()
    assert session.history == [{"role": "assistant", "content": session.config.greeting}]
    assert ("agent_transcript", session.config.greeting, False) in renderer.calls
    assert player.enqueued == [b"pcm:" + session.config.greeting.encode()]


def test_greet_records_tts_failure():
    def boom(text, sink):
        raise APIError("tts down")

    session, _renderer, player = make_session(synthesize=boom)
    session.greet()
    assert isinstance(session.error, APIError)
    assert session.error.message == "tts down"
    assert player.enqueued == []


def test_generate_reply_speaks_each_clause_as_it_streams():
    spoken = []
    session, renderer, player = make_session(
        stream_reply=_deltas("One. ", "Two! ", "Three?"),
        synthesize=lambda text, sink: spoken.append(text) or sink(text.encode()),
    )
    session._generate_reply()
    assert spoken == ["One.", "Two!", "Three?"]
    assert player.enqueued == [b"One.", b"Two!", b"Three?"]
    assert ("reply_started",) in renderer.calls
    assert ("agent_transcript", "One.", False) in renderer.calls
    assert session.history[-1] == {"role": "assistant", "content": "One. Two! Three?"}
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_forwards_tool_notice_and_drops_unspoken_preamble():
    # A ToolNotice surfaces the affordance AND clears any buffered-but-unspoken text, so a
    # half-streamed preamble before a tool call is never spoken.
    spoken = []

    def stream(messages):
        yield SpeechDelta("Let me check")  # incomplete clause, not yet flushed
        yield ToolNotice("Searching the web")
        yield SpeechDelta("It is sunny today.")

    session, renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert ("tool_call", "Searching the web") in renderer.calls
    assert spoken == ["It is sunny today."]  # the preamble was dropped, never synthesized
    assert session.history[-1] == {"role": "assistant", "content": "It is sunny today."}


def test_generate_reply_marks_speaking_on_first_delta_then_clears():
    observed = []
    session, _renderer, _player = make_session(stream_reply=_deltas("Hi. ", "Yes."))
    session.deps.synthesize = lambda text, sink: observed.append(session._speaking.is_set())
    session._generate_reply()
    assert observed == [True, True]
    assert not session._speaking.is_set()


def test_generate_reply_threads_system_prompt_and_history():
    captured = {}

    def capture(messages):
        captured["messages"] = messages
        return [SpeechDelta("Ok.")]

    session, _renderer, _player = make_session(
        stream_reply=capture, config=CascadeConfig(system_prompt="be terse")
    )
    session.history.append({"role": "user", "content": "prior"})
    session._generate_reply()
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert {"role": "user", "content": "prior"} in captured["messages"]


def test_generate_reply_trims_history_window():
    session, _renderer, _player = make_session(
        stream_reply=_deltas("a. b."), config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "user", "content": "hi"})
    session._generate_reply()
    assert session.history == [{"role": "assistant", "content": "a. b."}]


def test_generate_reply_stop_after_first_clause_records_partial():
    def synth(text, sink):
        if text == "Two.":
            session._stop.set()
        sink(text.encode())

    session, renderer, player = make_session(stream_reply=_deltas("One. Two. Three."))
    session.deps.synthesize = synth
    session._generate_reply()
    assert player.enqueued == [b"One."]
    assert session.history[-1] == {"role": "assistant", "content": "One."}
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_stop_before_first_clause_speaks_nothing():
    session, renderer, player = make_session(stream_reply=_deltas("One. Two."))
    session._stop.set()
    session._generate_reply()
    assert player.enqueued == []
    assert all(item.get("role") != "assistant" for item in session.history)
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_times_out_via_the_backstop(monkeypatch):
    release = threading.Event()

    def hang(messages):
        release.wait(timeout=2.0)  # self-releases so no mutated deadline can wedge the suite
        yield SpeechDelta("late")

    monkeypatch.setattr(engine, "_REPLY_TIMEOUT_SECONDS", 0.05)
    session, renderer, player = make_session(stream_reply=hang)
    try:
        session._generate_reply()
        assert isinstance(session.error, CLIError)
        assert session.error.error_type == "agent_timeout"
        assert any(c[0] == "agent_transcript" and "longer than" in c[1] for c in renderer.calls)
        assert ("reply_done", False) in renderer.calls
        assert player.enqueued == []
    finally:
        release.set()


def test_generate_reply_llm_failure_is_recorded_and_surfaced():
    def boom(messages):
        raise APIError("gateway down")
        yield  # pragma: no cover

    session, renderer, player = make_session(stream_reply=boom)
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert ("agent_transcript", "(error: gateway down)", False) in renderer.calls
    assert ("reply_done", False) in renderer.calls
    assert player.enqueued == []


def test_generate_reply_tts_failure_midway_is_recorded():
    def boom(text, sink):
        raise APIError("tts down")

    session, renderer, player = make_session(stream_reply=_deltas("Hi."), synthesize=boom)
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert player.enqueued == []
    assert ("reply_started",) in renderer.calls
    assert ("reply_done", False) in renderer.calls
```

Also update `test_on_turn_final_renders_and_replies`, `test_reply_forwards_tool_calls_to_the_renderer`, `test_on_turn_interim_shows_partial_and_does_not_reply`, and `test_on_turn_trims_history_window` to the new seam:

```python
def test_on_turn_final_renders_and_replies():
    session, renderer, player = make_session(stream_reply=_deltas("Sure thing."))
    session.on_turn(_turn("what time is it"))
    assert ("user_final", "what time is it") in renderer.calls
    assert {"role": "user", "content": "what time is it"} in session.history
    assert {"role": "assistant", "content": "Sure thing."} in session.history
    assert player.enqueued == [b"pcm:Sure thing."]
    assert ("reply_done", False) in renderer.calls


def test_reply_forwards_tool_calls_to_the_renderer():
    def stream(messages):
        yield ToolNotice("Searching the web")
        yield SpeechDelta("Found it.")

    session, renderer, _player = make_session(stream_reply=stream)
    session.on_turn(_turn("what's the news"))
    assert ("tool_call", "Searching the web") in renderer.calls


def test_on_turn_interim_shows_partial_and_does_not_reply():
    streamed = []
    session, renderer, _player = make_session(
        stream_reply=lambda m: streamed.append(m) or [SpeechDelta("x")]
    )
    session.on_turn(_turn("partial words", end_of_turn=False))
    assert ("user_partial", "partial words") in renderer.calls
    assert streamed == []
    assert session.history == []


def test_on_turn_trims_history_window():
    session, _renderer, _player = make_session(
        stream_reply=_deltas(""), config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "assistant", "content": "old"})
    session.on_turn(_turn("newest"))
    assert session.history == [{"role": "user", "content": "newest"}]
```

- [ ] **Step 3: Run the engine tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_engine.py -q`
Expected: FAIL — the engine still has the old `complete_reply`/`synthesize` seam (and `_complete_within`), so the new tests error on the changed `CascadeDeps` fields / removed method.

- [ ] **Step 4: Rewrite the engine's seam, imports, and constants**

In `aai_cli/agent_cascade/engine.py`:

Add imports near the top (after `import threading`):

```python
import queue
import time
```

Update the `text` import to include the splitter:

```python
from aai_cli.agent_cascade.text import pop_clauses, trim_history
```

Replace the `_REPLY_TIMEOUT_SECONDS` comment/const block with the streaming rationale and add the clause threshold:

```python
# Wall-clock backstop for one reply turn. The reply is streamed on a throwaway producer
# thread feeding a queue; a stalled gateway can block inside a token read the worker can't
# observe, so the consumer's queue.get is bounded by a monotonic deadline. After this long
# we stop waiting and surface a timeout so the session stays usable. Generous on purpose.
_REPLY_TIMEOUT_SECONDS = 60.0  # pragma: no mutate

# A clause is flushed to TTS on a soft separator (comma/semicolon/colon) only once it is at
# least this long, so we don't synthesize a choppy two-word fragment. Pinned by a text test.
_MIN_CLAUSE_CHARS = 25
```

Replace the `CascadeDeps` docstring comment + the two changed fields:

```python
    run_stt: Callable[[Callable[[object], None]], None]
    # stream_reply(messages) -> iterable of SpeechDelta/ToolNotice events. The reply is
    # streamed token-by-token so the engine can speak each clause as it lands; a ToolNotice
    # surfaces the "Searching the web…" affordance (brain.build_streamer).
    stream_reply: Callable[..., Iterable[object]]
    # synthesize(text, sink): streaming TTS — sink is called with each PCM frame as it
    # arrives so playback starts on the first frame instead of after the whole clause.
    synthesize: Callable[[str, Callable[[bytes], None]], None]
    spawn: Callable[[Callable[[], None]], _Worker] = _spawn_thread
```

Replace `CascadeDeps.real`'s body legs:

```python
        def run_stt(on_turn: Callable[[object], None]) -> None:
            client.stream_audio(api_key, audio, params=stt_params, on_turn=on_turn)

        # The LLM leg is a deepagents graph (web search / MCP tools), streamed token-by-token
        # so a spoken turn can transparently use tools and start speaking sooner.
        stream_reply = brain.build_streamer(api_key, config)

        def synthesize(text: str, sink: Callable[[bytes], None]) -> None:
            spec = SpeakConfig(
                text=text,
                voice=config.voice,
                language=config.language,
                sample_rate=TTS_SAMPLE_RATE,
                extra=config.tts_extra,
            )
            tts_session.synthesize(api_key, spec, on_audio=lambda chunk, _rate: sink(chunk))

        return cls(run_stt=run_stt, stream_reply=stream_reply, synthesize=synthesize)
```

Add `Iterable` to the typing import line: `from collections.abc import Callable, Iterable`.

- [ ] **Step 5: Rewrite `greet` and the reply path; remove `_complete_within`**

In `greet`, change the synth call:

```python
        try:
            self.deps.synthesize(greeting, self.player.enqueue)
        except CLIError as exc:
            self._record_error(exc)
```

Delete `_complete_within` entirely. Replace `_generate_reply` with the streaming consumer plus its helpers:

```python
    def _generate_reply(self) -> None:
        """Stream the LLM reply, speak each clause as it lands, and record what was spoken
        (so a barge-in still leaves the history alternating)."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.config.system_prompt},
            *self.history,
        ]
        events: queue.Queue[object] = queue.Queue()
        producer = threading.Thread(  # pragma: no mutate
            target=lambda: self._pump(messages, events), daemon=True
        )
        producer.start()
        deadline = time.monotonic() + _REPLY_TIMEOUT_SECONDS
        buffer = ""
        spoken: list[str] = []
        started = False
        aborted = False
        while True:
            try:
                item = events.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self._fail_leg(_timeout_error(), started)
                return
            if isinstance(item, _Failure):
                self._fail_leg(item.error, started)
                return
            if isinstance(item, _Done):
                break
            if isinstance(item, brain.ToolNotice):
                self.renderer.tool_call(item.label)
                buffer = ""  # drop any unspoken preamble — the answer comes after the tool
                continue
            if self._stop.is_set():
                aborted = True
                break
            if not started:
                self._speaking.set()
                self.renderer.reply_started()
                started = True
            buffer += item.text
            chunks, buffer = pop_clauses(buffer, min_chars=_MIN_CLAUSE_CHARS)
            if not self._speak(chunks, spoken):
                aborted = True
                break
        if not aborted:
            tail = buffer.strip()
            if tail:
                self._speak([tail], spoken)
        self._record_spoken(spoken)
        self._speaking.clear()
        self.renderer.reply_done(interrupted=self._stop.is_set())

    def _pump(self, messages: list[ChatCompletionMessageParam], events: queue.Queue[object]) -> None:
        """Drive the streaming reply leg on a throwaway thread, forwarding events to the
        queue and ending with a _Done (or _Failure on a clean leg error)."""
        try:
            for event in self.deps.stream_reply(messages):
                events.put(event)
            events.put(_Done())
        except CLIError as exc:
            events.put(_Failure(exc))

    def _speak(self, chunks: list[str], spoken: list[str]) -> bool:
        """Render and synthesize each clause, feeding frames to the player. Returns False if
        the turn was cut (barge-in stop or a TTS failure), True if every clause was spoken."""
        for chunk in chunks:
            if self._stop.is_set():
                return False
            self.renderer.agent_transcript(chunk, interrupted=False)
            try:
                self.deps.synthesize(chunk, self._feed)
            except CLIError as exc:
                self._record_error(exc)
                return False
            if self._stop.is_set():
                return False
            spoken.append(chunk)
        return True

    def _feed(self, pcm: bytes) -> None:
        """Enqueue one synthesized PCM frame, unless a barge-in has already landed (then the
        remaining frames of the in-flight clause are dropped)."""
        if not self._stop.is_set():
            self.player.enqueue(pcm)

    def _record_spoken(self, spoken: list[str]) -> None:
        """Append what was actually spoken to the history (kept alternating after a barge-in)."""
        spoken_text = " ".join(spoken).strip()
        if spoken_text:
            self.history.append({"role": "assistant", "content": spoken_text})
            trim_history(self.history, self.config.max_history)

    def _fail_leg(self, exc: CLIError, started: bool) -> None:
        """Surface a reply-leg failure (LLM/timeout) and close the turn. Before any audio,
        the error is shown inline in the transcript so the turn doesn't vanish; mid-speech it
        is only recorded (the spoken text already explains the turn)."""
        self._record_error(exc)
        if not started:
            self.renderer.reply_started()
            self.renderer.agent_transcript(f"(error: {exc.message})", interrupted=False)
        self._speaking.clear()
        self.renderer.reply_done(interrupted=self._stop.is_set())
```

Add the producer item types and the timeout factory near the top of the module (after `_REPLY_TIMEOUT_SECONDS`/`_MIN_CLAUSE_CHARS`), and a `dataclass` import is already present:

```python
@dataclass(frozen=True)
class _Done:
    """Producer sentinel: the reply stream finished normally."""


@dataclass(frozen=True)
class _Failure:
    """Producer sentinel: the reply leg raised a (clean) CLIError."""

    error: CLIError


def _timeout_error() -> CLIError:
    """The backstop error raised when a reply overruns the wall-clock deadline."""
    return CLIError(
        f"the agent took longer than {_REPLY_TIMEOUT_SECONDS:.0f}s to respond and was cut off",
        error_type="agent_timeout",
    )
```

- [ ] **Step 6: Run the engine tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_engine.py -q`
Expected: PASS. If `test_generate_reply_times_out_via_the_backstop` is flaky, confirm `events.get` uses the `max(0.0, deadline - now)` deadline (not a fixed timeout).

- [ ] **Step 7: Fix the `run_cascade` and `command` constructions**

In `tests/test_agent_cascade_engine.py`, every `CascadeDeps(...)` literal in the `run_cascade` tests uses the old field names — update each: `complete_reply=...` → `stream_reply=...` (a function returning a list of `SpeechDelta`), and `synthesize=lambda text: ...` → `synthesize=lambda text, sink: sink(...)`. Example for `test_run_cascade_greets_then_pumps_turns`:

```python
    def stream_reply(messages):
        session_box["messages"] = messages
        return [SpeechDelta("Hi back.")]

    ...
    deps = CascadeDeps(
        run_stt=run_stt,
        stream_reply=stream_reply,
        synthesize=lambda text, sink: sink(text.encode()),
        spawn=_sync_spawn,
    )
```

Apply the same shape to `test_run_cascade_hands_the_session_to_on_session_before_greeting`, `test_run_cascade_shuts_down_inflight_worker`, `test_run_cascade_reraises_recorded_leg_error` (the `boom` becomes a generator that `raise`s an `APIError` then has an unreachable `yield`), and `test_run_cascade_closes_player_when_stt_raises`.

In `tests/test_agent_cascade_command.py`:
- Update the `fake_real` near line 402 to `stream_reply=lambda _m: [], synthesize=lambda _t, _sink: None`.
- Replace `test_deps_real_complete_reply_is_built_by_the_deepagents_brain` with a streamer version:

```python
def test_deps_real_stream_reply_is_built_by_the_deepagents_brain(monkeypatch):
    from aai_cli.agent_cascade.brain import SpeechDelta

    def fake_build_streamer(api_key, config):
        del api_key, config
        return lambda messages: [SpeechDelta("reply to " + messages[-1]["content"])]

    monkeypatch.setattr(engine.brain, "build_streamer", fake_build_streamer)
    cfg = CascadeConfig()
    deps = CascadeDeps.real("k", cfg, audio=[], stt_params=_stt_params())
    events = list(deps.stream_reply([{"role": "user", "content": "hi"}]))
    assert [e.text for e in events] == ["reply to hi"]
```

- Replace `test_deps_real_synthesize_threads_voice_language_and_extra` to drive the streaming `on_audio`:

```python
def test_deps_real_synthesize_streams_frames_and_threads_voice(monkeypatch):
    captured = {}

    def fake_synth(api_key, spec, *, on_audio):
        captured["voice"] = spec.voice
        captured["sample_rate"] = spec.sample_rate
        on_audio(b"AUDIO", spec.sample_rate or 0)
        return engine.tts_session.SpeakResult(b"AUDIO", spec.sample_rate or 0, 0.0)

    monkeypatch.setattr(engine.tts_session, "synthesize", fake_synth)
    cfg = CascadeConfig(voice="luna")
    deps = CascadeDeps.real("k", cfg, audio=[], stt_params=_stt_params())
    frames = []
    deps.synthesize("say this", frames.append)
    assert frames == [b"AUDIO"]
    assert captured["voice"] == "luna"
    assert captured["sample_rate"] == 24000  # TTS always synthesizes at the live player's rate
```

- [ ] **Step 8: Run both touched test files**

Run: `uv run pytest tests/test_agent_cascade_engine.py tests/test_agent_cascade_command.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add aai_cli/agent_cascade/engine.py tests/_cascade_fakes.py tests/test_agent_cascade_engine.py tests/test_agent_cascade_command.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): stream the reply through clause-level streaming TTS

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: brain — remove the now-dead `build_completer` cluster

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py`
- Modify: `tests/test_agent_cascade_brain.py`

**Interfaces:**
- Removes: `build_completer`, `_run_graph`, `_drive_graph`, `_log_flow`, `_surface_event`, `_reply_text`. Keeps `_clip`, `_tool_label`, `_content_text`, `build_graph`, `build_system_prompt`, `build_live_tools`, the new `build_streamer`/`_stream_graph`/`_events_from_chunk`, and `SpeechDelta`/`ToolNotice`.

- [ ] **Step 1: Confirm nothing in `aai_cli/` still imports the old symbols**

Run: `grep -rn "build_completer\|_run_graph\|_drive_graph\|_log_flow\|_surface_event\|_reply_text" aai_cli/`
Expected: no matches (engine now uses `build_streamer`). If any remain, fix them before deleting.

- [ ] **Step 2: Delete the dead functions from `brain.py`**

Remove `build_completer`, `_run_graph`, `_drive_graph`, `_log_flow`, `_surface_event`, and `_reply_text` (the contiguous block from `def build_completer` through `def _reply_text`/its body, excluding `_content_text`, `_clip`, `_tool_label` which stay). Also drop any imports left unused (e.g. if `code_agent.events.message_events` was only used by `_log_flow`/`_surface_event` — verify with `uv run ruff check aai_cli/agent_cascade/brain.py`).

- [ ] **Step 3: Delete the dead tests**

In `tests/test_agent_cascade_brain.py`, remove the tests that exercised the deleted code: `test_completer_*`, `test_run_graph_*`, `test_on_tool_sink_streams_*`, `test_log_flow_ignores_non_list_messages`, and the `_reply_text`/`_content_text` block (`test_reply_text_*`, `test_content_text_coerces_unexpected_content`) — **except** keep one `_content_text` test if `_content_text` survives; re-point it:

```python
def test_content_text_coerces_unexpected_content():
    assert brain._content_text(123) == "123"


def test_content_text_joins_list_content_blocks():
    assert brain._content_text([{"type": "text", "text": "Hello "}, "world"]) == "Hello world"
```

Also remove the now-unused `_StreamingGraph`, `_search_call_message`, `_graph`, and `FakeChatModel`/`ChatGeneration`/`ChatResult` imports **only if** no surviving test (e.g. `test_build_graph_uses_gateway_model_and_runs_offline`) still uses them. `test_build_graph_uses_gateway_model_and_runs_offline` uses `FakeChatModel` and called `build_completer` — re-point it to `build_streamer`:

```python
def test_build_graph_uses_gateway_model_and_runs_offline(monkeypatch):
    captured = {}

    def fake_build_model(api_key, *, model, max_tokens, extra):
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        captured["extra"] = dict(extra)
        return FakeChatModel(responses=[AIMessage(content="hi from the agent")])

    monkeypatch.setattr(model_mod, "build_model", fake_build_model)
    cfg = CascadeConfig(model="claude-x", max_tokens=128, llm_extra={"temperature": 0.2})
    graph = brain.build_graph("k", cfg, tools=[])
    assert captured == {"model": "claude-x", "max_tokens": 128, "extra": {"temperature": 0.2}}
    streamer = brain.build_streamer("k", cfg, graph=graph)
    spoken = "".join(e.text for e in streamer([{"role": "user", "content": "hi"}]))
    assert spoken == "hi from the agent"
```

(`FakeChatModel` streams through the real deepagents graph; `build_streamer`'s messages-mode iteration collects its tokens. Keep `FakeChatModel` and its imports.)

Likewise re-point `test_build_graph_uses_gateway_model_and_runs_offline`'s sibling MCP tests (`test_build_graph_loads_mcp_tools_from_config_when_not_injected` calls `build_graph` only — unaffected).

- [ ] **Step 4: Run the brain suite + a vulture check**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: PASS.
Run: `uv run vulture aai_cli/agent_cascade/brain.py` (or rely on the gate) — expect no unused-code report.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "refactor(live): drop the superseded build_completer reply path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: docs + full gate

**Files:**
- Modify: `aai_cli/AGENTS.md`

- [ ] **Step 1: Update the architecture note**

In `aai_cli/AGENTS.md`, in the `agent_cascade/` bullet, replace the description of the LLM leg and per-sentence TTS to reflect streaming. Change the phrase "the cascade — greeting, per-sentence TTS, barge-in, history window —" to "the cascade — greeting, clause-level streaming TTS, barge-in, history window —", and rewrite the `-v` sentence:

> The LLM leg is a deepagents graph (`brain.py`) streamed token-by-token via `brain.build_streamer` (`stream_mode="messages"`): the engine buffers deltas, flushes complete clauses with `text.pop_clauses`, and synthesizes each with **streaming TTS** (`tts.session.synthesize(on_audio=…)`) so audio starts on the first frame. A `ToolNotice` surfaces the "Searching the web…" affordance and drops any unspoken preamble. Under `-v` (`debuglog.active()`) `brain._stream_graph` logs each accumulated assistant line, tool call, and tool result as it streams.

- [ ] **Step 2: Run the docs consistency gate**

Run: `uv run python scripts/docs_consistency_gate.py`
Expected: PASS (no env-var/exit-code/command drift — this change adds none).

- [ ] **Step 3: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` Likely fixups and how to clear them:
  - **Patch coverage / mutation**: every changed engine branch needs a *failing-on-break* assertion. The new tests cover the clause flush, the `ToolNotice` buffer-clear, the `_speaking` first-delta gate, the `_feed` stop-drop, the timeout, and both error paths. If the mutation gate flags `_MIN_CLAUSE_CHARS` or a boundary in `pop_clauses`, add/adjust a `tests/test_agent_cascade_text.py` case that distinguishes the two values (Task 1 already pins `min_chars=10` vs short fragments).
  - **Escape hatches**: confirm net-neutral — the producer's `daemon=True  # pragma: no mutate` replaces `_complete_within`'s removed one; do not add others. The two `# pragma: no cover` `yield` lines in the error-raising fake generators are test-only and count against the gate — if they tip the budget, rewrite those fakes as a tiny class with a `stream` method that `raise`s (no generator, no pragma needed), mirroring `_Boom` in Task 2.
  - **Textual coverage floor**: unaffected (no `tui.py` change), but `check.sh` runs it anyway — should stay ≥90%.
  - **xenon**: `_generate_reply` must stay ≤ B complexity. If it trips, the `_speak`/`_pump`/`_fail_leg`/`_record_spoken` helpers already factor most branches out; move the queue-item dispatch into a small `_handle_item` helper if needed.

- [ ] **Step 4: Final commit (gated)**

After `check.sh` prints `All checks passed.`, make the final commit normally (the gate marker is now recorded, so the commit hook permits it without `AAI_ALLOW_COMMIT`):

```bash
git add aai_cli/AGENTS.md
git commit -m "docs(live): describe the streaming reply pipeline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 brain reply event stream → Task 2 (`build_streamer`, `SpeechDelta`/`ToolNotice`, verbose logging) + Task 4 (remove old path). ✅
- §2 TTS frame sink → Task 3 (`CascadeDeps.synthesize` signature + `CascadeDeps.real` + `greet` + `_feed`). ✅
- §3 engine streaming `_generate_reply` (producer thread + queue + monotonic deadline + buffer-clear-on-tool + `_speaking` first-delta + error paths) → Task 3. ✅
- §4 incremental clause splitter `pop_clauses` → Task 1. ✅
- §5 testing (engine fake seam, brain fake graph `.stream`, `pop_clauses` table tests) → Tasks 1–4. ✅
- Risks/out-of-scope → no `--no-format-turns` or no-tools-completion work here (out of scope, as specified). ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The full-gate fixups in Task 5 are described with concrete remedies, not "handle errors." ✅

**Type consistency:** `stream_reply` (engine seam) returns an iterable of `brain.SpeechDelta | brain.ToolNotice`; `build_streamer` returns exactly that iterator; the fakes yield those types. `synthesize(text, sink)` is consistent across `CascadeDeps`, `CascadeDeps.real`, `greet`, `_speak`, and every fake. `pop_clauses(buffer, *, min_chars)` matches between Task 1's definition and Task 3's call. `_Done`/`_Failure`/`_timeout_error` are defined and used in Task 3. ✅
