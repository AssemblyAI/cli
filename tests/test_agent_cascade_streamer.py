"""Tests for `build_streamer`, the live brain's token-streaming reply leg.

The brain's only network seam is the compiled graph, so `build_streamer` is driven
against the *real* deepagents graph wired to a fake chat model (pytest-socket stays
armed) — no sockets. Split out of `test_agent_cascade_brain.py` to keep each file under
the 500-line gate; the shared `FakeChatModel` lives in `_cascade_fakes`.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.core.errors import CLIError
from tests._cascade_fakes import FakeChatModel

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


def _collect(graph, messages):
    streamer = brain.build_streamer("k", CascadeConfig(), graph=graph)
    return list(streamer(messages))


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


def test_build_middleware_caps_tool_calls_with_a_graceful_stop():
    # The brain wires a per-turn tool-call budget that forces a graceful answer instead of
    # erroring: a ToolCallLimitMiddleware with the config's run_limit and exit_behavior="continue"
    # (block further tool calls and let the model answer with what it has). The default is 10.
    from langchain.agents.middleware import ToolCallLimitMiddleware

    (default_mw,) = brain._build_middleware(CascadeConfig())
    assert isinstance(default_mw, ToolCallLimitMiddleware)
    assert default_mw.run_limit == 10  # DEFAULT_TOOL_CALL_LIMIT
    assert default_mw.exit_behavior == "continue"  # answer with what it has, never raise

    (custom_mw,) = brain._build_middleware(CascadeConfig(tool_call_limit=3))
    assert isinstance(custom_mw, ToolCallLimitMiddleware)
    assert custom_mw.run_limit == 3


def test_streamer_emits_a_tool_notice_when_a_tool_call_starts():
    call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": brain.WEB_SEARCH_TOOL_NAME, "args": "", "id": "c1", "index": 0}],
    )
    graph = _MessageStreamGraph([(call_chunk, {}), (AIMessageChunk(content="Here it is."), {})])
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

    streamer = brain.build_streamer("k", CascadeConfig(), graph=_CliBoom())
    with pytest.raises(CLIError, match="already clean"):
        list(streamer([{"role": "user", "content": "hi"}]))


def test_streamer_errors_when_graph_cannot_stream():
    # A graph that only implements invoke (no .stream) can't be streamed — the streamer
    # must surface a clean CLIError rather than AttributeError-ing mid-turn.
    class _InvokeOnly:
        def invoke(self, graph_input):
            del graph_input
            return {"messages": []}

    streamer = brain.build_streamer("k", CascadeConfig(), graph=_InvokeOnly())
    with pytest.raises(CLIError) as excinfo:
        list(streamer([{"role": "user", "content": "hi"}]))
    assert "cannot stream" in excinfo.value.message


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


# --- build_streamer write approval (--files) ---------------------------------


def _gated_graph(model: BaseChatModel, root: str):
    """A real deepagents graph that gates write_file/edit_file, rooted at ``root``."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import InMemorySaver

    return create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        interrupt_on={"write_file": True, "edit_file": True},
        checkpointer=InMemorySaver(),
        system_prompt="be a friendly live agent",
    )


def _write_then(reply: str) -> FakeChatModel:
    """A model that calls write_file once, then (after resume) answers with ``reply``."""
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": "/n.txt", "content": "hi"}, "id": "w1"}
        ],
    )
    return FakeChatModel(responses=[call, AIMessage(content=reply)])


def test_streamer_approves_write_then_resumes(tmp_path):
    asked: list[tuple[str, dict]] = []

    def approve(name, args):
        asked.append((name, args))
        return True

    graph = _gated_graph(_write_then("Saved your note."), str(tmp_path))
    streamer = brain.build_streamer("k", CascadeConfig(files=True), graph=graph, approver=approve)
    events = list(streamer([{"role": "user", "content": "save a note"}]))
    spoken = "".join(e.text for e in events if isinstance(e, brain.SpeechDelta))
    assert spoken == "Saved your note."
    # The approver was consulted for the write, and the approved write hit the rooted dir.
    assert asked and asked[0][0] == "write_file"
    assert (tmp_path / "n.txt").read_text() == "hi"


def test_streamer_rejects_write_without_approval(tmp_path):
    graph = _gated_graph(_write_then("Okay, I won't save it."), str(tmp_path))
    streamer = brain.build_streamer(
        "k", CascadeConfig(files=True), graph=graph, approver=lambda name, args: False
    )
    events = list(streamer([{"role": "user", "content": "save a note"}]))
    spoken = "".join(e.text for e in events if isinstance(e, brain.SpeechDelta))
    assert spoken == "Okay, I won't save it."
    # Declined: nothing was written to the rooted directory.
    assert not (tmp_path / "n.txt").exists()


def test_streamer_brackets_write_approval_with_pause_events(tmp_path):
    # The human-think wait is bracketed by ApprovalPause(active=True/False) so the engine can
    # suspend its reply-timeout deadline for exactly that interval. The approver runs between
    # the two markers by construction (the streamer yields True, asks, then yields False).
    asked: list[str] = []
    graph = _gated_graph(_write_then("Done."), str(tmp_path))
    streamer = brain.build_streamer(
        "k",
        CascadeConfig(files=True),
        graph=graph,
        approver=lambda name, args: asked.append(name) or True,
    )
    events = list(streamer([{"role": "user", "content": "save"}]))
    pauses = [event.active for event in events if isinstance(event, brain.ApprovalPause)]
    assert pauses == [True, False]  # the write was bracketed: pause on, then resume
