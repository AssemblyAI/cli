"""Deepagents-powered reply brain for the live voice cascade.

`assembly live` answers each spoken turn with a deepagents graph instead of a single
LLM completion, so the agent can transparently reach for a tool — web search —
mid-conversation, mimicking a live multimodal assistant (the "talk to Gemini Live"
experience). The toolset is deliberately minimal: a low-latency spoken turn does best
with one obvious tool rather than a menu it has to choose among. The graph is built once per session
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

from aai_cli.agent_cascade import weather_tool
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.code_agent.agent import CompiledAgent
from aai_cli.code_agent.firecrawl_search import WEB_SEARCH_TOOL_NAME
from aai_cli.core import debuglog
from aai_cli.core.errors import CLIError

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

# Human, speakable labels for the tool affordance the live UI shows while a tool runs (so a
# spoken turn that pauses to use a tool says *why* it's working, not just spin silently).
_TOOL_LABELS = {
    WEB_SEARCH_TOOL_NAME: "Searching the web",
    weather_tool.WEATHER_TOOL_NAME: "Checking the weather",
}


def _tool_label(name: str) -> str:
    """A short present-tense label for a tool call, shown as the live UI's tool affordance."""
    return _TOOL_LABELS.get(name, f"Using {name}")


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
    """The spoken-capability phrases backed by present built-in tools.

    The live agent's built-in legs are the keyless Open-Meteo weather tool (always
    present) and Firecrawl web search (only when ``FIRECRAWL_API_KEY`` is set) — so the
    prompt advertises each only when the agent can really do it. Advertising a missing
    tool made it announce an action ("I'll search…") it then couldn't take.
    """
    names = {tool.name for tool in tools}
    capabilities: list[str] = []
    if WEB_SEARCH_TOOL_NAME in names:
        capabilities.append("search the web for current or unfamiliar facts")
    if weather_tool.WEATHER_TOOL_NAME in names:
        capabilities.append("tell someone the current weather and short forecast for a place")
    return capabilities


def _extra_capability(extra_tools: Sequence[BaseTool]) -> str | None:
    """The spoken-capability phrase for user-configured MCP tools, listing them by name.

    The deepagents graph already shows the model each tool's schema, so this only has to
    name the tools so the guidance doesn't claim "no external tools" when MCP tools are
    bound — and so the model knows to reach for them.
    """
    names = sorted(tool.name for tool in extra_tools)
    if not names:
        return None
    return f"use your connected tools ({', '.join(names)})"


def build_system_prompt(
    persona: str, *, tools: Sequence[BaseTool], extra_tools: Sequence[BaseTool] = ()
) -> str:
    """The live agent's system prompt: the user's persona plus tool guidance.

    The guidance is tailored to the bound tools so the model is only told about
    capabilities it actually has — advertising a missing tool (web search without a
    ``FIRECRAWL_API_KEY``) made the agent announce an action it then couldn't take, leaving
    the turn hanging with no answer. ``tools`` are the built-in legs (web search, URL
    fetch, AssemblyAI docs); ``extra_tools`` are user-configured MCP tools, advertised
    generically by name. With no tools at all the model answers from its own knowledge.
    """
    capabilities = _tool_capabilities(tools)
    extra = _extra_capability(extra_tools)
    if extra is not None:
        capabilities.append(extra)
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
    """The live agent's built-in tools: the keyless weather tool, plus Firecrawl web
    search when ``FIRECRAWL_API_KEY`` is set.

    Deliberately minimal. A low-latency spoken turn does best with a few obvious tools
    rather than a large menu it must choose among. Open-Meteo needs no key, so the
    weather tool is always present (every session has at least one real capability);
    web search is reused (un-approval-gated) from the coding agent and added only when
    keyed. Extra tools remain strictly opt-in via ``--mcp-config``.
    """
    from aai_cli.agent_cascade.weather_tool import build_weather_tool
    from aai_cli.code_agent.firecrawl_search import build_web_search_tool

    tools: list[BaseTool] = [build_weather_tool()]
    search = build_web_search_tool()
    if search is not None:
        tools.append(search)
    return tools


def build_graph(
    api_key: str,
    config: CascadeConfig,
    *,
    tools: Sequence[BaseTool] | None = None,
    mcp_tools: Sequence[BaseTool] | None = None,
) -> CompiledAgent:
    """Compile the deepagents graph for one live session over the gateway model.

    Reuses the coding agent's gateway-bound ``ChatOpenAI`` (so the live agent can only
    ever reach AssemblyAI), threading the cascade's ``--max-tokens``/``--llm-config``
    through it. ``tools`` defaults to :func:`build_live_tools`; ``mcp_tools`` defaults to
    the tools of the servers in ``config.mcp_servers``. The two are kept apart so the
    system prompt advertises the built-in legs and the MCP tools differently, but the
    model is bound to both. Tests pass explicit (possibly empty) lists to skip the
    network-touching docs/MCP probes.
    """
    from deepagents import create_deep_agent

    from aai_cli.agent_cascade.mcp_tools import load_mcp_tools
    from aai_cli.code_agent.model import build_model

    model = build_model(
        api_key, model=config.model, max_tokens=config.max_tokens, extra=config.llm_extra
    )
    builtin = build_live_tools() if tools is None else list(tools)
    extra = load_mcp_tools(config.mcp_servers) if mcp_tools is None else list(mcp_tools)
    return create_deep_agent(
        model=model,
        tools=builtin + extra,
        system_prompt=build_system_prompt(config.system_prompt, tools=builtin, extra_tools=extra),
    )


def build_completer(
    api_key: str, config: CascadeConfig, *, graph: CompiledAgent | None = None
) -> Callable[..., str]:
    """A ``complete_reply`` for the cascade engine backed by the deepagents graph.

    The cascade prepends its own ``system`` message to the history each turn; the graph
    already owns the system prompt, so we drop it before invoking. The graph runs the full
    tool loop and we return its final spoken text. ``on_tool`` (when given) is called with a
    short label as each tool call lands, so the front-end can show a "Searching the web…"
    affordance instead of sitting silent while the agent works; the loop is also streamed —
    rather than ``invoke``-d — whenever a sink is wired or under ``-v`` (see :func:`_run_graph`).
    ``graph`` is injected in tests so the per-turn wiring runs against a fake with no network.
    """
    resolved = build_graph(api_key, config) if graph is None else graph

    def complete_reply(
        messages: list[ChatCompletionMessageParam],
        on_tool: Callable[[str], None] | None = None,
    ) -> str:
        conversation = [message for message in messages if message.get("role") != "system"]
        return _reply_text(_run_graph(resolved, conversation, on_tool))

    return complete_reply


def _run_graph(
    graph: CompiledAgent,
    conversation: list[ChatCompletionMessageParam],
    on_tool: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run one turn through the graph, returning its end state.

    Normally a single ``invoke`` (the whole tool loop runs internally). When a tool sink is
    wired (the live UI's affordance) or under verbose mode, and the graph can stream, drive
    it as incremental state snapshots instead so :func:`_log_flow` surfaces each tool call as
    it happens. The test fakes only implement ``invoke``, so they (and the plain path with no
    sink) take the invoke branch.
    """
    try:
        return _drive_graph(graph, {"messages": conversation}, on_tool)
    except CLIError:
        raise
    except Exception as exc:
        # The graph can fail anywhere in the tool loop — a gateway 4xx/5xx, a tool raising,
        # a langgraph recursion limit. Convert it to a CLIError so the cascade records and
        # *surfaces* it (the engine shows it in the transcript) instead of the reply worker
        # dying silently and the user getting no answer with no clue why.
        raise CLIError(
            f"the agent couldn't complete the turn: {exc}", error_type="agent_brain_error"
        ) from exc


def _drive_graph(
    graph: CompiledAgent,
    graph_input: dict[str, object],
    on_tool: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Invoke the graph, or stream it (when a tool sink is wired or under ``-v``) so
    :func:`_log_flow` can surface each tool call as it lands."""
    if (on_tool is not None or debuglog.active()) and hasattr(graph, "stream"):
        last: dict[str, object] = {}
        seen = 0
        for chunk in graph.stream(graph_input, None, stream_mode="values"):
            seen = _log_flow(chunk, seen, on_tool)
            last = chunk
        return last
    return graph.invoke(graph_input)


def _log_flow(
    state: dict[str, object], seen: int, on_tool: Callable[[str], None] | None = None
) -> int:
    """Surface the tool calls/results added to ``state`` since the first ``seen`` messages.

    Feeds ``on_tool`` a speakable label as each tool call lands (the live UI's affordance) and,
    under ``-v``, logs the call/result/interim line to stderr. Reuses the coding agent's
    message→event vocabulary so it reads the same AIMessage/ToolMessage shapes the TUI does.
    Returns the new high-water message count so the next snapshot only re-surfaces what it added.
    """
    from aai_cli.code_agent.events import message_events

    messages = state.get("messages")
    if not isinstance(messages, list):
        return seen
    verbose = debuglog.active()
    for message in messages[seen:]:
        for event in message_events(message, announce_calls=True):
            _surface_event(event, on_tool, verbose=verbose)
    return len(messages)


def _surface_event(event: object, on_tool: Callable[[str], None] | None, *, verbose: bool) -> None:
    """Surface one flow event: feed a tool call's label to ``on_tool``, and (under ``-v``)
    log the call/result/interim line to stderr."""
    from aai_cli.code_agent.events import AssistantText, ToolCall, ToolResult

    if isinstance(event, ToolCall) and on_tool is not None:
        on_tool(_tool_label(event.name))
    if not verbose:
        return
    if isinstance(event, ToolCall):
        _FLOW_LOG.info("tool call %s args=%s", event.name, event.args)
    elif isinstance(event, ToolResult):
        _FLOW_LOG.info("tool result %s -> %s", event.name, _clip(event.content))
    elif isinstance(event, AssistantText):
        _FLOW_LOG.info("llm: %s", event.text)


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
