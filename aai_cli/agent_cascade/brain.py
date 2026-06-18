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

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.code_agent.agent import CompiledAgent
from aai_cli.code_agent.fetch_tool import FETCH_TOOL_NAME
from aai_cli.code_agent.web_search import WEB_SEARCH_TOOL_NAME
from aai_cli.core import debuglog

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from openai.types.chat import ChatCompletionMessageParam

# Verbose (`-v`) flow logging for the agent's tool loop. `invoke` runs the whole loop
# internally, so without this `-v` only shows the httpx request lines and never which
# tools the agent reached for or what they returned — exactly what you need to see when
# a spoken turn stalls mid-tool. Logged at INFO so plain `-v` surfaces it.
_FLOW_LOG = logging.getLogger("aai_cli.agent_cascade.brain")

# Tool outputs (a fetched page, a search payload) can be huge; cap what we log per result
# so a single tool call doesn't bury the rest of the flow in stderr. The exact cap is an
# arbitrary tuning knob — a +-1 shift is behaviorally equivalent, so no test can kill it.
_RESULT_LOG_CAP = 500  # pragma: no mutate

# Closes every guidance variant: the reply is spoken, so it must stay short and plain.
_SPOKEN_TAIL = (
    "Your reply is read aloud, so keep it short and spoken — no markdown, lists, code, or raw URLs."
)

# When the session has *no* tools wired (e.g. no web search and the docs host is
# unreachable), the model must answer from its own knowledge — and crucially must not
# promise an action it can't take. Without this, telling it "you can search the web" while
# no search tool is bound makes it narrate "I'll search for that…" and then stop, so the
# answer never comes (the tool it announced was never actually available to call).
_NO_TOOLS_GUIDANCE = (
    "You have no external tools available, so answer from your own knowledge. Never say "
    "you will search the web, look something up, or fetch a page — you can't do any of "
    "that, so don't promise it; if a question needs information you don't have, say so "
    f"briefly instead. {_SPOKEN_TAIL}"
)


def _join_clause(parts: list[str]) -> str:
    """Join capability phrases into a readable clause: ``a``, ``a and b``, ``a, b, and c``."""
    *initial, last = parts
    if not initial:
        return last
    # Oxford comma only once there are three-or-more items (two or more lead the last).
    joiner = ", and " if initial[1:] else " and "
    return f"{', '.join(initial)}{joiner}{last}"


def _tool_capabilities(tools: Sequence[BaseTool]) -> list[str]:
    """The spoken-capability phrases backed by an actually-present tool.

    Derived from the resolved tool names so the prompt never advertises a capability the
    agent can't perform: web search is present only with a ``TAVILY_API_KEY``, and the docs
    tools are best-effort (absent when the docs host is unreachable).
    """
    names = {tool.name for tool in tools}
    capabilities: list[str] = []
    if WEB_SEARCH_TOOL_NAME in names:
        capabilities.append("search the web for current or unfamiliar facts")
    if FETCH_TOOL_NAME in names:
        capabilities.append("fetch a specific URL")
    if names - {WEB_SEARCH_TOOL_NAME, FETCH_TOOL_NAME}:
        capabilities.append("look up the AssemblyAI documentation")
    return capabilities


def build_system_prompt(persona: str, *, tools: Sequence[BaseTool]) -> str:
    """The live agent's system prompt: the user's persona plus tool guidance.

    The guidance is tailored to ``tools`` so the model is only told about capabilities it
    actually has — advertising a missing tool (web search without a ``TAVILY_API_KEY``) made
    the agent announce an action it then couldn't take, leaving the turn hanging with no
    answer. With no tools at all the model is told to answer from its own knowledge.
    """
    capabilities = _tool_capabilities(tools)
    if not capabilities:
        return f"{persona}\n\n{_NO_TOOLS_GUIDANCE}"
    guidance = (
        f"You can use tools to help answer: {_join_clause(capabilities)}. Reach for a "
        "tool when a question needs fresh or external information; answer directly and "
        "instantly when you already know. Only offer to do what these tools allow — don't "
        f"say you'll search the web or look something up unless it's listed here. {_SPOKEN_TAIL}"
    )
    return f"{persona}\n\n{guidance}"


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
        model=model,
        tools=resolved,
        system_prompt=build_system_prompt(config.system_prompt, tools=resolved),
    )


def build_completer(
    api_key: str, config: CascadeConfig, *, graph: CompiledAgent | None = None
) -> Callable[[list[ChatCompletionMessageParam]], str]:
    """A ``complete_reply`` for the cascade engine backed by the deepagents graph.

    The cascade prepends its own ``system`` message to the history each turn; the graph
    already owns the system prompt, so we drop it before invoking. The graph runs the
    full tool loop and we return its final spoken text. Under ``-v`` the loop is streamed
    so each tool call/result is logged as it lands (see :func:`_run_graph`). ``graph`` is
    injected in tests so the per-turn wiring runs against a fake with no network.
    """
    resolved = build_graph(api_key, config) if graph is None else graph

    def complete_reply(messages: list[ChatCompletionMessageParam]) -> str:
        conversation = [message for message in messages if message.get("role") != "system"]
        return _reply_text(_run_graph(resolved, conversation))

    return complete_reply


def _run_graph(
    graph: CompiledAgent, conversation: list[ChatCompletionMessageParam]
) -> dict[str, object]:
    """Run one turn through the graph, returning its end state.

    Normally a single ``invoke`` (the whole tool loop runs internally). Under verbose
    mode, and when the graph can stream, drive it as incremental state snapshots instead
    so :func:`_log_flow` can surface each tool call/result on stderr as it happens — which
    is what makes a stalled spoken turn debuggable. The test fakes only implement
    ``invoke``, so they (and the non-verbose path) take the plain branch.
    """
    graph_input = {"messages": conversation}
    if debuglog.active() and hasattr(graph, "stream"):
        last: dict[str, object] = {}
        seen = 0
        for chunk in graph.stream(graph_input, None, stream_mode="values"):
            seen = _log_flow(chunk, seen)
            last = chunk
        return last
    return graph.invoke(graph_input)


def _log_flow(state: dict[str, object], seen: int) -> int:
    """Log the tool calls/results added to ``state`` since the first ``seen`` messages.

    Reuses the coding agent's message→event vocabulary so the flow log knows the same
    AIMessage/ToolMessage shapes the TUI does. Returns the new high-water message count
    so the next snapshot only logs what it added.
    """
    from aai_cli.code_agent.events import AssistantText, ToolCall, ToolResult, message_events

    messages = state.get("messages")
    if not isinstance(messages, list):
        return seen
    for message in messages[seen:]:
        for event in message_events(message, announce_calls=True):
            if isinstance(event, ToolCall):
                _FLOW_LOG.info("tool call %s args=%s", event.name, event.args)
            elif isinstance(event, ToolResult):
                _FLOW_LOG.info("tool result %s -> %s", event.name, _clip(event.content))
            elif isinstance(event, AssistantText):
                _FLOW_LOG.info("llm: %s", event.text)
    return len(messages)


def _clip(text: str) -> str:
    """Flatten a tool result onto one line and truncate it for the flow log.

    Tool output is untrusted external content (a fetched page, a search payload), so its
    whitespace — newlines especially — is collapsed before logging: a result can't then
    forge extra ``[aai_cli.…]`` log lines, and each result stays on one readable line. The
    length is capped so a multi-KB payload can't bury the rest of the flow. (Secrets are
    separately masked by the debuglog formatter across every record.)
    """
    flattened = " ".join(text.split())
    if len(flattened) <= _RESULT_LOG_CAP:
        return flattened
    return f"{flattened[:_RESULT_LOG_CAP]}… ({len(flattened)} chars)"


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
