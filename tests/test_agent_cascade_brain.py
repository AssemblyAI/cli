"""Tests for the deepagents reply brain behind `assembly live`.

The brain's only network seam is the compiled graph, so `build_completer` is driven
against the *real* deepagents graph wired to a fake chat model (pytest-socket stays
armed) — no sockets. `build_live_tools` and `build_model`'s new knobs are unit-tested
directly.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.code_agent import model as model_mod


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


def test_system_prompt_appends_tool_guidance_for_present_tools():
    prompt = brain.build_system_prompt(
        "You are a pirate.",
        tools=[
            _NamedTool(brain.WEB_SEARCH_TOOL_NAME),
            _NamedTool("fetch_url"),
            _NamedTool("docs_search"),
        ],
    )
    # The persona is preserved, and the guidance advertises each capability that a present
    # tool backs (the plain cascade persona never mentions tools).
    assert prompt.startswith("You are a pirate.")
    assert "search the web" in prompt
    assert "fetch a specific URL" in prompt
    assert "AssemblyAI documentation" in prompt


def test_system_prompt_omits_web_search_when_no_search_tool():
    # With no TAVILY_API_KEY the search tool is absent — the guidance must NOT promise web
    # search, since announcing a missing tool makes the agent narrate "I'll search…" and
    # then stall with no answer. The capabilities it *does* have still appear.
    prompt = brain.build_system_prompt(
        "persona", tools=[_NamedTool("fetch_url"), _NamedTool("docs_search")]
    )
    assert "search the web for current or unfamiliar facts" not in prompt
    assert "fetch a specific URL" in prompt
    assert "AssemblyAI documentation" in prompt


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
    from langchain_core.messages import ToolMessage

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


def test_build_live_tools_includes_search_when_keyed(monkeypatch):
    search = object()
    monkeypatch.setattr("aai_cli.code_agent.fetch_tool.build_fetch_tool", lambda: "fetch")
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: search)
    monkeypatch.setattr("aai_cli.code_agent.docs_mcp.load_docs_tools", lambda: ["docs"])
    tools = brain.build_live_tools()
    # Fetch + the keyed search + the docs tools, in that order.
    assert tools == ["fetch", search, "docs"]


def test_build_live_tools_omits_search_when_unkeyed(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.fetch_tool.build_fetch_tool", lambda: "fetch")
    monkeypatch.setattr("aai_cli.code_agent.firecrawl_search.build_web_search_tool", lambda: None)
    monkeypatch.setattr("aai_cli.code_agent.docs_mcp.load_docs_tools", list)
    tools = brain.build_live_tools()
    # No TAVILY_API_KEY -> no search tool, just the fetch tool.
    assert tools == ["fetch"]


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
    builtin = [_NamedTool("fetch_url")]
    extra = [_NamedTool("get_time")]
    graph = brain.build_graph("k", CascadeConfig(), tools=builtin, mcp_tools=extra)
    # The model is bound to both tool sets, in built-in-then-MCP order.
    assert graph == "graph"
    assert captured["tools"] == builtin + extra
    # The prompt advertises the built-in fetch leg AND the MCP tool by name.
    assert "fetch a specific URL" in captured["system_prompt"]
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
