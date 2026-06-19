"""Tests for the deepagents reply brain behind `assembly live`.

The brain's only network seam is the compiled graph, so `build_completer` is driven
against the *real* deepagents graph wired to a fake chat model (pytest-socket stays
armed) — no sockets. `build_live_tools` and `build_model`'s new knobs are unit-tested
directly.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aai_cli.agent_cascade import brain
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


def _graph(model: BaseChatModel):
    from deepagents import create_deep_agent

    return create_deep_agent(model=model, tools=[], system_prompt="be a friendly live agent")


# --- build_system_prompt -----------------------------------------------------


class _NamedTool:
    """A stand-in tool exposing just the ``.name`` the prompt builder inspects."""

    def __init__(self, name: str):
        self.name = name


def test_system_prompt_advertises_web_search_when_present():
    prompt = brain.build_system_prompt(
        "You are a pirate.", tools=[_NamedTool(brain.WEB_SEARCH_TOOL_NAME)]
    )
    # The persona is preserved, and the guidance advertises the web-search capability the
    # present tool backs (the plain cascade persona never mentions tools).
    assert prompt.startswith("You are a pirate.")
    assert "search the web" in prompt


def test_system_prompt_omits_web_search_when_search_tool_absent():
    # Without the Firecrawl search tool the guidance must NOT promise web search — announcing
    # a missing tool makes the agent narrate "I'll search…" and then stall with no answer. A
    # non-search tool name must not falsely trigger the web-search capability.
    prompt = brain.build_system_prompt("persona", tools=[_NamedTool("some_other_tool")])
    assert "search the web for current or unfamiliar facts" not in prompt


def test_system_prompt_tells_model_not_to_promise_tools_when_none():
    # No tools at all: the model must answer from its own knowledge and explicitly not
    # promise to search or look anything up (the bug that left replies never coming back).
    prompt = brain.build_system_prompt("persona", tools=[])
    assert "search the web for current or unfamiliar facts" not in prompt
    assert "your own knowledge" in prompt
    assert "Never say" in prompt


def test_extra_capability_lists_sorted_tool_names():
    # MCP tools are advertised generically, by name, alphabetically.
    phrase = brain._extra_capability([_NamedTool("zeta"), _NamedTool("alpha")])
    assert phrase == "use your connected tools (alpha, zeta)"


def test_extra_capability_is_none_without_extra_tools():
    assert brain._extra_capability([]) is None


def test_system_prompt_advertises_mcp_extra_tools():
    # With MCP tools bound (but no built-in legs), the model must be told it HAS tools —
    # not handed the "no external tools" guidance — and the tools are named.
    prompt = brain.build_system_prompt("persona", tools=[], extra_tools=[_NamedTool("get_time")])
    assert "your own knowledge" not in prompt
    assert "use your connected tools (get_time)" in prompt


def test_join_clause_grammar():
    # One/two/three capability phrases each render with natural conjunctions.
    assert brain._join_clause(["a"]) == "a"
    assert brain._join_clause(["a", "b"]) == "a and b"
    assert brain._join_clause(["a", "b", "c"]) == "a, b, and c"


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


# --- build_completer (driving the real graph with a fake model) --------------


def test_completer_returns_final_spoken_text():
    graph = _graph(FakeChatModel(responses=[AIMessage(content="Hello there.")]))
    completer = brain.build_completer("k", CascadeConfig(), graph=graph)
    reply = completer([{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])
    assert reply == "Hello there."


def test_completer_strips_system_message_before_invoking():
    # The cascade prepends its own system message each turn, but the graph already owns
    # the system prompt — so the completer must drop it before invoking, leaving only the
    # conversation. We capture what the graph received to prove the system line is gone.
    captured = {}

    class _CapturingGraph:
        def invoke(self, value):
            captured["messages"] = value["messages"]
            return {"messages": [AIMessage(content="ok")]}

    completer = brain.build_completer("k", CascadeConfig(), graph=_CapturingGraph())
    completer([{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}])
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user"]


# --- _run_graph / _log_flow (verbose tool-call flow) -------------------------


class _StreamingGraph:
    """A graph that streams scripted state snapshots (the shape the real graph yields).

    Records the kwargs it was streamed with so a test can prove ``_run_graph`` asked for
    incremental value snapshots, and exposes an ``invoke`` that must never run on the
    verbose path."""

    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.stream_kwargs = None
        self.invoked = False

    def stream(self, graph_input, config, *, stream_mode):
        del graph_input, config
        self.stream_kwargs = stream_mode
        yield from self.snapshots

    def invoke(self, graph_input):
        del graph_input
        self.invoked = True
        return {"messages": []}


def _search_call_message():
    return AIMessage(
        content="Let me search.",
        tool_calls=[{"name": "tavily_search", "args": {"query": "weather"}, "id": "c1"}],
    )


def test_run_graph_streams_and_logs_flow_when_verbose(monkeypatch, caplog, preserve_logging_state):
    # Verbose mode streams the loop and logs each step — the assistant's interim line, the
    # tool call (name + args), and the tool result — so a stalled spoken turn is debuggable.
    monkeypatch.setattr(brain.debuglog, "active", lambda: True)
    call = _search_call_message()
    snapshots = [
        {"messages": [call]},
        {
            "messages": [
                call,
                ToolMessage(content="rainy, 52F", name="tavily_search", tool_call_id="c1"),
            ]
        },
        {
            "messages": [
                call,
                ToolMessage(content="rainy, 52F", name="tavily_search", tool_call_id="c1"),
                AIMessage(content="It's rainy and 52 degrees in Portland."),
            ]
        },
    ]
    graph = _StreamingGraph(snapshots)
    completer = brain.build_completer("k", CascadeConfig(), graph=graph)
    with caplog.at_level(logging.INFO, logger="aai_cli.agent_cascade.brain"):
        reply = completer([{"role": "user", "content": "weather?"}])
    # The streamed final state still yields the spoken reply, and the graph was streamed
    # for incremental value snapshots (not invoked).
    assert reply == "It's rainy and 52 degrees in Portland."
    assert graph.stream_kwargs == "values"
    assert graph.invoked is False
    # The flow log carries the tool call (with its args), the tool result, and the interim
    # assistant line — each logged exactly once despite the growing snapshots.
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "llm: Let me search.",
        "tool call tavily_search args={'query': 'weather'}",
        "tool result tavily_search -> rainy, 52F",
        "llm: It's rainy and 52 degrees in Portland.",
    ]


def test_run_graph_invokes_when_not_verbose():
    # Default (non-verbose): the graph is invoked once, never streamed, and nothing is logged.
    graph = _StreamingGraph([{"messages": [AIMessage(content="hi")]}])
    completer = brain.build_completer("k", CascadeConfig(), graph=graph)
    assert completer([{"role": "user", "content": "hi"}]) == ""
    assert graph.invoked is True
    assert graph.stream_kwargs is None


def test_run_graph_invokes_when_graph_cannot_stream(monkeypatch):
    # Verbose but the (test) graph only implements invoke: fall back to invoke rather than
    # crashing on a missing .stream — the fakes and any non-streaming graph stay supported.
    monkeypatch.setattr(brain.debuglog, "active", lambda: True)

    class _InvokeOnly:
        def invoke(self, graph_input):
            del graph_input
            return {"messages": [AIMessage(content="from invoke")]}

    completer = brain.build_completer("k", CascadeConfig(), graph=_InvokeOnly())
    assert completer([{"role": "user", "content": "hi"}]) == "from invoke"


def test_run_graph_converts_graph_errors_to_cli_error():
    # A graph failure (gateway 4xx/5xx, a tool raising, a recursion limit) must become a
    # CLIError so the cascade surfaces it instead of the reply worker dying silently.
    class _Boom:
        def invoke(self, graph_input):
            del graph_input
            raise ValueError("bedrock said no")

    completer = brain.build_completer("k", CascadeConfig(), graph=_Boom())
    with pytest.raises(CLIError) as excinfo:
        completer([{"role": "user", "content": "hi"}])
    assert "couldn't complete the turn" in excinfo.value.message
    assert "bedrock said no" in excinfo.value.message  # the cause is preserved for diagnosis


def test_run_graph_passes_cli_error_through():
    # A CLIError from the graph is already user-facing -> propagate as-is, not re-wrapped.
    class _CliBoom:
        def invoke(self, graph_input):
            del graph_input
            raise CLIError("already clean", error_type="x")

    completer = brain.build_completer("k", CascadeConfig(), graph=_CliBoom())
    with pytest.raises(CLIError, match="already clean"):
        completer([{"role": "user", "content": "hi"}])


def test_log_flow_ignores_non_list_messages():
    # Defensive: a snapshot without a messages list logs nothing and reports no progress.
    assert brain._log_flow({"messages": None}, 3) == 3


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


# --- _reply_text / _content_text ---------------------------------------------


def test_reply_text_skips_empty_ai_messages_and_takes_last_text():
    # Scanning from the end, a trailing empty AIMessage (a tool-call request with no
    # spoken text) is skipped so the reply falls back to the prior AIMessage's text,
    # rather than coming back blank.
    result = {
        "messages": [
            AIMessage(content="The answer is 42."),
            AIMessage(content=""),
        ]
    }
    assert brain._reply_text(result) == "The answer is 42."


def test_reply_text_joins_list_content_blocks():
    result = {"messages": [AIMessage(content=[{"type": "text", "text": "Hello "}, "world"])]}
    assert brain._reply_text(result) == "Hello world"


def test_reply_text_skips_non_assistant_messages():

    # Scanning from the end, a trailing non-assistant message (e.g. a tool result) is
    # skipped — the spoken reply is the AIMessage before it.
    result = {
        "messages": [
            AIMessage(content="hello there"),
            ToolMessage(content="tool output", tool_call_id="c1"),
        ]
    }
    assert brain._reply_text(result) == "hello there"


def test_content_text_coerces_unexpected_content():
    # A content that is neither a string nor a list of blocks (defensive fallback).
    assert brain._content_text(123) == "123"


def test_reply_text_is_empty_without_an_assistant_message():
    assert brain._reply_text({"messages": []}) == ""
    assert brain._reply_text({}) == ""


# --- build_live_tools --------------------------------------------------------


def test_build_live_tools_is_just_web_search_when_keyed(monkeypatch):
    search = object()
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: search)
    # The live agent's sole built-in tool is Firecrawl web search — no URL fetch, no docs.
    assert brain.build_live_tools() == [search]


def test_build_live_tools_is_empty_without_firecrawl_key(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: None)
    # No FIRECRAWL_API_KEY -> no tool at all; the agent then runs tool-free.
    assert brain.build_live_tools() == []


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
    completer = brain.build_completer("k", cfg, graph=graph)
    assert completer([{"role": "user", "content": "hi"}]) == "hi from the agent"


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
