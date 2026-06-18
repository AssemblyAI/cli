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


def test_system_prompt_appends_tool_guidance():
    prompt = brain.build_system_prompt("You are a pirate.")
    # The persona is preserved, and the tool guidance is appended so the model knows it
    # can search the web (the plain cascade persona never mentions tools).
    assert prompt.startswith("You are a pirate.")
    assert "search the web" in prompt


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
    monkeypatch.setattr("aai_cli.code_agent.web_search.build_web_search_tool", lambda: search)
    monkeypatch.setattr("aai_cli.code_agent.docs_mcp.load_docs_tools", lambda: ["docs"])
    tools = brain.build_live_tools()
    # Fetch + the keyed search + the docs tools, in that order.
    assert tools == ["fetch", search, "docs"]


def test_build_live_tools_omits_search_when_unkeyed(monkeypatch):
    monkeypatch.setattr("aai_cli.code_agent.fetch_tool.build_fetch_tool", lambda: "fetch")
    monkeypatch.setattr("aai_cli.code_agent.web_search.build_web_search_tool", lambda: None)
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


# --- build_model new knobs ---------------------------------------------------


def test_build_model_threads_max_tokens_and_extra():
    model = model_mod.build_model("k", model="claude-x", max_tokens=222, extra={"top_k": 5})
    assert model.max_tokens == 222
    assert model.extra_body == {"top_k": 5}


def test_build_model_defaults_have_no_extra():
    model = model_mod.build_model("k", model="claude-x")
    assert model.max_tokens is None
    assert model.extra_body is None
