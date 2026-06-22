"""Tests for the deepagents reply brain behind `assembly live`.

The brain's only network seam is the compiled graph, so `build_streamer` is driven
against the *real* deepagents graph wired to a fake chat model (pytest-socket stays
armed) — no sockets. `build_live_tools` and `build_model`'s new knobs are unit-tested
directly.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aai_cli.agent_cascade import brain, datetime_tool, weather_tool, webpage_tool
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.code_agent import model as model_mod
from aai_cli.core.errors import CLIError


class FakeChatModel(BaseChatModel):
    """A chat model that replays a scripted list of AIMessages (mirrors the code agent's)."""

    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-live-model"

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        message = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


# --- _graph_kwargs (real-cwd backend + write-gating when --files is on) -------


def test_graph_kwargs_empty_when_files_off():
    # With files off the graph is built exactly as before: no backend swap, no gating.
    assert brain._graph_kwargs(CascadeConfig(files=False)) == {}


def test_graph_kwargs_gates_writes_and_roots_backend_at_cwd(monkeypatch, tmp_path):
    from pathlib import Path

    from deepagents.backends import FilesystemBackend

    monkeypatch.chdir(tmp_path)
    kwargs = brain._graph_kwargs(CascadeConfig(files=True))

    backend = kwargs["backend"]
    assert isinstance(backend, FilesystemBackend)
    # Rooted at the launch directory; virtual_mode blocks traversal escapes.
    assert Path(backend.cwd) == tmp_path.resolve()
    assert backend.virtual_mode is True
    # Only the mutating file tools are gated — reads (incl. grep) and the inert execute aren't.
    assert kwargs["interrupt_on"] == {"write_file": True, "edit_file": True}
    assert kwargs["checkpointer"] is not None


# --- build_system_prompt -----------------------------------------------------


class _NamedTool:
    """A stand-in tool exposing just the ``.name`` the prompt builder inspects."""

    def __init__(self, name: str):
        self.name = name


def test_web_search_tool_name_matches_built_tool(monkeypatch):
    # The prompt builder detects search by WEB_SEARCH_TOOL_NAME, so pin it against the real
    # Firecrawl tool's registered name — if it renames, detection would silently break.
    from aai_cli.code_agent import firecrawl_search

    monkeypatch.setenv(firecrawl_search.FIRECRAWL_API_KEY_ENV, "fc-x")
    tool = firecrawl_search.build_web_search_tool()
    assert tool is not None
    assert tool.name == firecrawl_search.WEB_SEARCH_TOOL_NAME == brain.WEB_SEARCH_TOOL_NAME


def test_web_search_absent_without_firecrawl_key(monkeypatch):
    from aai_cli.code_agent import firecrawl_search

    monkeypatch.delenv(firecrawl_search.FIRECRAWL_API_KEY_ENV, raising=False)
    assert firecrawl_search.build_web_search_tool() is None


def test_tool_label_maps_web_search_and_falls_back_for_others():
    assert brain._tool_label(brain.WEB_SEARCH_TOOL_NAME) == "Searching the web"
    assert brain._tool_label("get_time") == "Using get_time"


def test_tool_label_for_file_ops_is_speakable():
    # The file tools get speakable affordance labels so a write/search turn reads as progress.
    assert brain._tool_label("write_file") == "Writing a file"
    assert brain._tool_label("edit_file") == "Editing a file"
    assert brain._tool_label("read_file") == "Reading a file"
    assert brain._tool_label("grep") == "Searching files"


def test_clip_passes_short_text_and_truncates_long_text():
    assert brain._clip("short") == "short"
    # A result exactly at the cap is left whole (the boundary is inclusive).
    at_cap = "y" * brain._RESULT_LOG_CAP
    assert brain._clip(at_cap) == at_cap
    long = "x" * (brain._RESULT_LOG_CAP + 5000)
    clipped = brain._clip(long)
    # Only the first _RESULT_LOG_CAP chars survive, with a marker noting the full length —
    # so a multi-KB tool payload can't bury the rest of the flow in stderr.
    assert clipped == "x" * brain._RESULT_LOG_CAP + f"… ({len(long)} chars)"
    assert len(clipped) < len(long)


def test_clip_flattens_whitespace_so_tool_output_cant_forge_log_lines():
    # Tool output is untrusted: a result with embedded CR/LF could otherwise inject fake
    # "[aai_cli.…]" log lines. _clip collapses all whitespace runs to single spaces, so the
    # result stays on one line.
    forged = "ok\n[aai_cli.agent_cascade.brain] tool call rm_rf args={}\r\nmore"
    assert brain._clip(forged) == "ok [aai_cli.agent_cascade.brain] tool call rm_rf args={} more"
    assert "\n" not in brain._clip(forged)
    assert "\r" not in brain._clip(forged)


# --- _content_text -----------------------------------------------------------


def test_content_text_coerces_unexpected_content():
    # A content that is neither a string nor a list of blocks (defensive fallback).
    assert brain._content_text(123) == "123"


def test_content_text_joins_list_content_blocks():
    assert brain._content_text([{"type": "text", "text": "Hello "}, "world"]) == "Hello world"


# --- build_live_tools --------------------------------------------------------


def test_build_live_tools_has_weather_and_web_search_when_keyed(monkeypatch):
    search = _NamedTool(brain.WEB_SEARCH_TOOL_NAME)
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: search)
    names = [tool.name for tool in brain.build_live_tools()]
    # Web search is the optional keyed leg; the keyless weather, read-url, and datetime tools
    # are always present. Exact set assertion kills duplicated/extra tools a loose `in` check would miss.
    assert sorted(names) == sorted(
        [
            brain.WEB_SEARCH_TOOL_NAME,
            weather_tool.WEATHER_TOOL_NAME,
            webpage_tool.READ_URL_TOOL_NAME,
            datetime_tool.DATETIME_TOOL_NAME,
        ]
    )


def test_build_live_tools_has_keyless_tools_without_firecrawl_key(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: None)
    # No FIRECRAWL_API_KEY -> no web search, but the keyless weather, read-url, and datetime tools load.
    names = [tool.name for tool in brain.build_live_tools()]
    assert names == [
        weather_tool.WEATHER_TOOL_NAME,
        webpage_tool.READ_URL_TOOL_NAME,
        datetime_tool.DATETIME_TOOL_NAME,
    ]


def test_tool_label_maps_read_url():
    assert brain._tool_label(webpage_tool.READ_URL_TOOL_NAME) == "Reading the page"


# --- build_graph (model construction + compile, with the docs probe skipped) -


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
    # The cascade's model + knobs are threaded into the gateway model build.
    assert captured == {"model": "claude-x", "max_tokens": 128, "extra": {"temperature": 0.2}}
    # The compiled graph is a real deepagents graph that answers offline via the fake model.
    streamer = brain.build_streamer("k", cfg, graph=graph)
    spoken = "".join(e.text for e in streamer([{"role": "user", "content": "hi"}]))
    assert spoken == "hi from the agent"


# --- build_graph MCP tool wiring ---------------------------------------------


def test_build_graph_binds_builtin_plus_mcp_tools_and_advertises_both(monkeypatch):
    import deepagents

    captured = {}

    def fake_create(*, model, tools, system_prompt):
        del model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        return "graph"

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    monkeypatch.setattr(model_mod, "build_model", lambda *a, **k: object())
    builtin = [_NamedTool(brain.WEB_SEARCH_TOOL_NAME)]
    extra = [_NamedTool("get_time")]
    graph = brain.build_graph("k", CascadeConfig(), tools=builtin, mcp_tools=extra)
    # The model is bound to both tool sets, in built-in-then-MCP order.
    assert graph == "graph"
    assert captured["tools"] == builtin + extra
    # The prompt advertises the built-in web-search leg AND the MCP tool by name.
    assert "search the web" in captured["system_prompt"]
    assert "use your connected tools (get_time)" in captured["system_prompt"]


def test_build_graph_loads_mcp_tools_from_config_when_not_injected(monkeypatch):
    import deepagents

    seen = {}

    def fake_load(servers):
        seen["servers"] = servers
        return [_NamedTool("weather")]

    monkeypatch.setattr("aai_cli.agent_cascade.mcp_tools.load_mcp_tools", fake_load)
    monkeypatch.setattr(model_mod, "build_model", lambda *a, **k: object())
    monkeypatch.setattr(deepagents, "create_deep_agent", lambda **kwargs: kwargs["tools"])
    cfg = CascadeConfig(mcp_servers={"weather": {"command": "npx"}})
    tools = brain.build_graph("k", cfg, tools=[])
    # The config's servers are loaded (default path) and their tools bound.
    assert seen["servers"] == {"weather": {"command": "npx"}}
    assert [t.name for t in tools] == ["weather"]


# --- build_model new knobs ---------------------------------------------------


def test_build_model_threads_max_tokens_and_extra():
    model = model_mod.build_model("k", model="claude-x", max_tokens=222, extra={"top_k": 5})
    assert model.max_tokens == 222
    assert model.extra_body == {"top_k": 5}


def test_build_model_defaults_have_no_extra():
    model = model_mod.build_model("k", model="claude-x")
    assert model.max_tokens is None
    assert model.extra_body is None


def test_tool_label_maps_weather():
    assert brain._tool_label(weather_tool.WEATHER_TOOL_NAME) == "Checking the weather"


def test_tool_label_maps_datetime():
    assert brain._tool_label(datetime_tool.DATETIME_TOOL_NAME) == "Checking the time"


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
    assert asked == ["write_file"]  # the approver was consulted exactly once, for the write
