"""Tests for the deepagents reply brain behind `assembly live`.

Covers graph assembly (`_graph_kwargs`/`build_graph`), the built-in tool set
(`build_live_tools`), tool labels, and `build_model`'s knobs — all unit-tested directly.
The token-streaming reply leg (`build_streamer`) lives in
`test_agent_cascade_streamer.py`; the shared `FakeChatModel` is in `_cascade_fakes`.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from aai_cli.agent_cascade import brain, datetime_tool, plan, streamer, weather_tool, webpage_tool
from aai_cli.agent_cascade import model as model_mod
from aai_cli.agent_cascade.config import CascadeConfig
from tests._cascade_fakes import FakeChatModel

# --- _graph_kwargs (real-cwd backend + write-gating when --files is on) -------


def test_graph_kwargs_empty_when_files_off():
    # With files off the graph is built exactly as before: no backend swap, no gating.
    assert brain._graph_kwargs(CascadeConfig(files=False)) == {}


def test_graph_kwargs_gates_writes_and_execute_and_sets_memory(monkeypatch, tmp_path):
    from aai_cli.agent_cascade import sandbox

    monkeypatch.chdir(tmp_path)
    kwargs = brain._graph_kwargs(CascadeConfig(files=True))

    backend = kwargs["backend"]
    assert isinstance(backend, sandbox.SandboxedShellBackend)
    assert Path(backend.cwd) == tmp_path.resolve()
    assert backend.virtual_mode is True
    # execute now joins the write gate.
    assert kwargs["interrupt_on"] == {"write_file": True, "edit_file": True, "execute": True}
    assert kwargs["checkpointer"] is not None
    # Durable per-project memory is turned on.
    assert kwargs["memory"] == ["./.deepagents/AGENTS.md"]


def test_sandboxed_backend_implements_sandbox_protocol(monkeypatch, tmp_path):
    from deepagents.backends.protocol import SandboxBackendProtocol

    monkeypatch.chdir(tmp_path)
    backend = brain._build_fs_backend()
    assert isinstance(backend, SandboxBackendProtocol)


# --- build_system_prompt -----------------------------------------------------


class _NamedTool:
    """A stand-in tool exposing just the ``.name`` the prompt builder inspects."""

    def __init__(self, name: str):
        self.name = name


def test_web_search_tool_name_matches_built_tool(monkeypatch):
    # The prompt builder detects search by WEB_SEARCH_TOOL_NAME, so pin it against the real
    # Firecrawl tool's registered name — if it renames, detection would silently break.
    from aai_cli.agent_cascade import firecrawl_search

    monkeypatch.setenv(firecrawl_search.FIRECRAWL_API_KEY_ENV, "fc-x")
    tool = firecrawl_search.build_web_search_tool()
    assert tool is not None
    assert tool.name == firecrawl_search.WEB_SEARCH_TOOL_NAME == brain.WEB_SEARCH_TOOL_NAME


def test_web_search_absent_without_firecrawl_key(monkeypatch):
    from aai_cli.agent_cascade import firecrawl_search

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


def test_tool_label_execute_is_running_code():
    assert brain._tool_label("execute") == "Running code"


def test_clip_passes_short_text_and_truncates_long_text():
    assert streamer._clip("short") == "short"
    # A result exactly at the cap is left whole (the boundary is inclusive).
    at_cap = "y" * streamer._RESULT_LOG_CAP
    assert streamer._clip(at_cap) == at_cap
    long = "x" * (streamer._RESULT_LOG_CAP + 5000)
    clipped = streamer._clip(long)
    # Only the first _RESULT_LOG_CAP chars survive, with a marker noting the full length —
    # so a multi-KB tool payload can't bury the rest of the flow in stderr.
    assert clipped == "x" * streamer._RESULT_LOG_CAP + f"… ({len(long)} chars)"
    assert len(clipped) < len(long)


def test_clip_flattens_whitespace_so_tool_output_cant_forge_log_lines():
    # Tool output is untrusted: a result with embedded CR/LF could otherwise inject fake
    # "[aai_cli.…]" log lines. _clip collapses all whitespace runs to single spaces, so the
    # result stays on one line.
    forged = "ok\n[aai_cli.agent_cascade.brain] tool call rm_rf args={}\r\nmore"
    assert streamer._clip(forged) == "ok [aai_cli.agent_cascade.brain] tool call rm_rf args={} more"
    assert "\n" not in streamer._clip(forged)
    assert "\r" not in streamer._clip(forged)


# --- _content_text -----------------------------------------------------------


def test_content_text_coerces_unexpected_content():
    # A content that is neither a string nor a list of blocks (defensive fallback).
    assert streamer._content_text(123) == "123"


def test_content_text_joins_list_content_blocks():
    assert streamer._content_text([{"type": "text", "text": "Hello "}, "world"]) == "Hello world"


# --- build_live_tools --------------------------------------------------------


def test_build_live_tools_has_weather_and_web_search_when_keyed(monkeypatch):
    search = _NamedTool(brain.WEB_SEARCH_TOOL_NAME)
    monkeypatch.setattr(
        "aai_cli.agent_cascade.firecrawl_search.build_web_search_tool", lambda: search
    )
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
    monkeypatch.setattr(
        "aai_cli.agent_cascade.firecrawl_search.build_web_search_tool", lambda: None
    )
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
    stream_reply = streamer.build_streamer("k", cfg, graph=graph)
    spoken = "".join(e.text for e in stream_reply([{"role": "user", "content": "hi"}]))
    assert spoken == "hi from the agent"


# --- build_graph MCP tool wiring ---------------------------------------------


def test_build_graph_binds_builtin_plus_mcp_tools_and_advertises_both(monkeypatch):
    import deepagents

    captured = {}

    def fake_create(*, model, tools, system_prompt, middleware):
        del model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["middleware"] = middleware
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
    # The per-turn tool-call budget is wired into the deepagents middleware stack.
    from langchain.agents.middleware import ToolCallLimitMiddleware

    assert any(isinstance(mw, ToolCallLimitMiddleware) for mw in captured["middleware"])


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


# --- write_todos plan parsing ------------------------------------------------


def test_parse_todos_shapes_a_valid_blob_and_rejects_junk():
    update = plan._parse_todos('{"todos":[{"content":"A","status":"pending"}]}')
    assert update == plan.TodoUpdate((plan.TodoItem(content="A", status="pending"),))
    # Not-JSON, valid-JSON-without-todos, and an empty list all yield no plan (None).
    assert plan._parse_todos("{not json") is None
    assert plan._parse_todos('{"other":1}') is None
    assert plan._parse_todos('{"todos":[]}') is None


def test_todos_from_list_drops_non_dict_items_and_defaults_missing_fields():
    update = plan._todos_from_list([{"content": "A"}, "junk", {"status": "completed"}])
    # The string item is dropped; missing content/status default to empty strings.
    assert update == plan.TodoUpdate(
        (
            plan.TodoItem(content="A", status=""),
            plan.TodoItem(content="", status="completed"),
        )
    )
    assert plan._todos_from_list("not a list") is None


def test_todo_collector_resets_after_take():
    collector = plan.TodoCollector()
    chunk = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_todos", "args": {"todos": [{"content": "A", "status": "x"}]}, "id": "w"}
        ],
    )
    collector.note_chunk(chunk)
    assert collector.take() == plan.TodoUpdate((plan.TodoItem(content="A", status="x"),))
    # take() consumed the buffer: a second take with no further chunks yields nothing.
    assert collector.take() is None


def test_todo_collector_prefers_complete_buffer_over_partial_tool_calls():
    # A streaming chunk carries both the full JSON fragment AND langchain's partial-parse
    # .tool_calls; the complete buffer must win so the plan isn't truncated to the partial parse.
    from langchain_core.messages import AIMessageChunk

    collector = plan.TodoCollector()
    collector.note_chunk(
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "write_todos", "args": '{"todos":[{"content":"A",', "id": "w", "index": 0}
            ],
        )
    )
    collector.note_chunk(
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": '"status":"done"}]}', "id": "w", "index": 0}],
        )
    )
    assert collector.take() == plan.TodoUpdate((plan.TodoItem(content="A", status="done"),))
