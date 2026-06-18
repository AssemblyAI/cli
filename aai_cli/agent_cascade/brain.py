"""Deepagents-powered reply brain for the live voice cascade.

`assembly live` answers each spoken turn with a deepagents graph instead of a single
LLM completion, so the agent can transparently reach for tools — web search, URL
fetch, the AssemblyAI docs — mid-conversation, mimicking a live multimodal assistant
(the "talk to Gemini Live" experience). The graph is built once per session
(:func:`build_graph`) and invoked statelessly per turn with the running history the
cascade already keeps (:func:`build_completer`); tools are read-only and auto-approved,
because a spoken turn can't pause for a keyboard confirmation, and the system prompt
keeps every reply short and speakable.

The graph is the only network seam: :func:`build_completer` accepts an injected graph,
so the per-turn orchestration is unit-tested against a fake with no sockets — the same
seam the rest of the cascade uses for its STT/LLM/TTS legs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.code_agent.agent import CompiledAgent

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from openai.types.chat import ChatCompletionMessageParam

# Appended to the user's persona so the model knows it has tools and must keep replies
# spoken. The cascade's plain-LLM persona (CascadeConfig.system_prompt) says nothing
# about tools, so without this the agent would never reach for web search.
_TOOL_GUIDANCE = (
    "You can use tools to help answer: search the web for current or unfamiliar facts, "
    "fetch a specific URL, and look up the AssemblyAI documentation. Reach for a tool "
    "when a question needs fresh or external information; answer directly and instantly "
    "when you already know. Your reply is read aloud, so keep it short and spoken — no "
    "markdown, lists, code, or raw URLs."
)


def build_system_prompt(persona: str) -> str:
    """The live agent's system prompt: the user's persona plus the tool guidance."""
    return f"{persona}\n\n{_TOOL_GUIDANCE}"


def build_live_tools() -> list[BaseTool]:
    """The live agent's read-only toolset: URL fetch, web search (if keyed), and docs.

    All three are reused from the coding agent's tool modules. Unlike there they are
    *not* approval-gated — a spoken turn can't wait for a keyboard confirmation, so the
    live agent only gets read-only tools and runs them automatically. Web search is
    present only when ``TAVILY_API_KEY`` is set; the docs MCP is best-effort (an empty
    list when the host is unreachable), so neither blocks a session.
    """
    from aai_cli.code_agent.docs_mcp import load_docs_tools
    from aai_cli.code_agent.fetch_tool import build_fetch_tool
    from aai_cli.code_agent.web_search import build_web_search_tool

    tools: list[BaseTool] = [build_fetch_tool()]
    search = build_web_search_tool()
    if search is not None:
        tools.append(search)
    tools.extend(load_docs_tools())
    return tools


def build_graph(
    api_key: str, config: CascadeConfig, *, tools: Sequence[BaseTool] | None = None
) -> CompiledAgent:
    """Compile the deepagents graph for one live session over the gateway model.

    Reuses the coding agent's gateway-bound ``ChatOpenAI`` (so the live agent can only
    ever reach AssemblyAI), threading the cascade's ``--max-tokens``/``--llm-config``
    through it. ``tools`` defaults to :func:`build_live_tools`; tests pass an explicit
    (possibly empty) list to skip the network-touching docs probe.
    """
    from deepagents import create_deep_agent

    from aai_cli.code_agent.model import build_model

    model = build_model(
        api_key, model=config.model, max_tokens=config.max_tokens, extra=config.llm_extra
    )
    resolved = build_live_tools() if tools is None else list(tools)
    return create_deep_agent(
        model=model, tools=resolved, system_prompt=build_system_prompt(config.system_prompt)
    )


def build_completer(
    api_key: str, config: CascadeConfig, *, graph: CompiledAgent | None = None
) -> Callable[[list[ChatCompletionMessageParam]], str]:
    """A ``complete_reply`` for the cascade engine backed by the deepagents graph.

    The cascade prepends its own ``system`` message to the history each turn; the graph
    already owns the system prompt, so we drop it before invoking. The graph runs the
    full tool loop and we return its final spoken text. ``graph`` is injected in tests
    so the per-turn wiring runs against a fake with no network.
    """
    resolved = build_graph(api_key, config) if graph is None else graph

    def complete_reply(messages: list[ChatCompletionMessageParam]) -> str:
        conversation = [message for message in messages if message.get("role") != "system"]
        return _reply_text(resolved.invoke({"messages": conversation}))

    return complete_reply


def _reply_text(result: dict[str, object]) -> str:
    """The agent's final spoken reply: the last assistant message that carries text.

    A tool-using turn ends in an ``AIMessage`` whose ``content`` is the spoken answer,
    but earlier ``AIMessage``\\s in the same turn (the tool-call requests) have empty
    text — so we scan from the end for the last one with non-empty content.
    """
    messages = result.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if type(message).__name__ != "AIMessage":
            continue
        text = _content_text(getattr(message, "content", "")).strip()
        if text:
            return text
    return ""


def _content_text(content: object) -> str:
    """Coerce a message's content (a string, or a list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)
