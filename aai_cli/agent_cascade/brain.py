"""Deepagents-powered reply brain for the live voice cascade.

`assembly live` answers each spoken turn with a deepagents graph instead of a single
LLM completion, so the agent can transparently reach for a tool — web search —
mid-conversation, mimicking a live multimodal assistant (the "talk to Gemini Live"
experience). The toolset is deliberately minimal: a low-latency spoken turn does best
with one obvious tool rather than a menu it has to choose among. The graph is built once per session
(:func:`build_graph`) and driven turn-by-turn with the running history the
cascade already keeps (:func:`build_streamer`); tools are read-only and auto-approved,
because a spoken turn can't pause for a keyboard confirmation, and the system prompt
keeps every reply short and speakable.

The graph is the only network seam: :func:`build_streamer` accepts an injected graph,
so the per-turn streaming reply leg is unit-tested against a fake with no sockets — the
same seam the rest of the cascade uses for its STT/LLM/TTS legs.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aai_cli.agent_cascade import datetime_tool, weather_tool, webpage_tool
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.firecrawl_search import WEB_SEARCH_TOOL_NAME
from aai_cli.agent_cascade.prompt import build_system_prompt
from aai_cli.core import debuglog
from aai_cli.core.errors import CLIError

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import BaseTool
    from openai.types.chat import ChatCompletionMessageParam


class CompiledAgent(Protocol):
    """The slice of the compiled langgraph graph the live reply leg drives.

    A structural type so we needn't name langgraph's deeply-generic
    ``CompiledStateGraph`` (and don't drag its type params through our code).
    """

    def invoke(
        self, input: object, config: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Run one step of the graph, returning the updated state (incl. messages)."""


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
    webpage_tool.READ_URL_TOOL_NAME: "Reading the page",
    datetime_tool.DATETIME_TOOL_NAME: "Checking the time",
    # The --files filesystem tools (deepagents' built-in names).
    "read_file": "Reading a file",
    "write_file": "Writing a file",
    "edit_file": "Editing a file",
    "execute": "Running code",
    "task": "Working on a subtask",
    "ls": "Listing files",
    "glob": "Finding files",
    "grep": "Searching files",
}


def _tool_label(name: str) -> str:
    """A short present-tense label for a tool call, shown as the live UI's tool affordance."""
    return _TOOL_LABELS.get(name, f"Using {name}")


# Spoken filler the agent says aloud when it pauses for a tool, so a hands-free turn fills the
# silent tool round-trip with *why* it paused instead of dead air (the audible counterpart to the
# visual `_TOOL_LABELS` affordance). Each tool gets a few short, speakable variants the engine
# rotates across turns; unknown/MCP tools fall back to `_GENERIC_FILLERS`. Spoken-style only — no
# markdown, no trailing detail — since they're synthesized straight to TTS ahead of the answer.
_GENERIC_FILLERS: tuple[str, ...] = ("One sec.", "Let me check.")

_TOOL_FILLERS: dict[str, tuple[str, ...]] = {
    WEB_SEARCH_TOOL_NAME: (
        "Let me look that up.",
        "Searching now.",
        "One moment, checking the web.",
    ),
    weather_tool.WEATHER_TOOL_NAME: ("Let me check the weather.", "Checking the forecast now."),
    webpage_tool.READ_URL_TOOL_NAME: ("Let me pull up that page.", "Reading it now."),
    datetime_tool.DATETIME_TOOL_NAME: ("Let me check the time.", "One moment."),
}


def _tool_fillers(name: str) -> tuple[str, ...]:
    """The spoken filler variants for a tool call, falling back to the generic tuple.

    Mirrors :func:`_tool_label`: a known tool gets its own phrases, an unknown/MCP tool the
    generic fallback. The tuple (not a single pre-chosen phrase) rides on :class:`ToolNotice`
    so the engine owns rotation state and two notices for the same tool don't repeat.
    """
    return _TOOL_FILLERS.get(name, _GENERIC_FILLERS)


@dataclass(frozen=True)
class SpeechDelta:
    """A top-level assistant-text token delta to be spoken (one piece of the reply)."""

    text: str


@dataclass(frozen=True)
class ToolNotice:
    """A speakable affordance emitted when the agent starts a tool call mid-turn.

    ``label`` is the visual affordance ("Searching the web"); ``fillers`` are the spoken
    variants the engine may say aloud for the *first* tool call of a turn (it owns the
    rotation), so a hands-free turn isn't dead air during the tool round-trip.
    """

    label: str
    fillers: tuple[str, ...]


@dataclass(frozen=True)
class ApprovalPause:
    """Brackets a human write-approval wait (``--files``).

    Emitted ``active=True`` just before the streamer blocks on the user's y/n decision and
    ``active=False`` once it's answered, so the engine can suspend its reply-timeout deadline
    for exactly the human-think interval (a slow keypress must not cut off the write).
    """

    active: bool


@runtime_checkable
class _GatedGraph(Protocol):
    """The graph surface the --files write-approval loop drives beyond ``invoke``.

    ``CompiledAgent`` deliberately declares only ``invoke`` (mirroring the code agent), so the
    gated path narrows to this protocol for the ``stream``/``get_state`` it additionally needs.
    """

    def stream(
        self, graph_input: object, config: Mapping[str, object] | None, *, stream_mode: str
    ) -> Iterator[tuple[object, object]]:
        """Yield ``(message_chunk, metadata)`` pairs for one streamed segment."""

    def get_state(self, config: Mapping[str, object] | None) -> object:
        """The checkpointed state snapshot (its ``.interrupts`` carry any pending write)."""


# Decide whether a gated write may run (front-end supplied). Mirrors the code agent's Approver.
Approver = Callable[[str, dict[str, object]], bool]

# Message handed back to the model when the user declines a write (matches the code agent's copy).
_DECLINED = "User declined to run this tool."


def build_live_tools() -> list[BaseTool]:
    """The live agent's built-in tools: the keyless weather, read-a-URL, and date/time
    tools, plus Firecrawl web search when ``FIRECRAWL_API_KEY`` is set.

    Deliberately minimal. A low-latency spoken turn does best with a few obvious tools
    rather than a large menu it must choose among. Open-Meteo, the URL reader, and the
    system clock need no key, so the weather, read-url, and datetime tools are always
    present (every session has real capabilities); web search is reused (un-approval-gated)
    from the coding agent and added only when keyed. Extra tools remain strictly opt-in via
    ``--mcp-config``.
    """
    from aai_cli.agent_cascade.datetime_tool import build_datetime_tool
    from aai_cli.agent_cascade.firecrawl_search import build_web_search_tool
    from aai_cli.agent_cascade.weather_tool import build_weather_tool
    from aai_cli.agent_cascade.webpage_tool import build_read_url_tool

    tools: list[BaseTool] = [build_weather_tool(), build_read_url_tool(), build_datetime_tool()]
    search = build_web_search_tool()
    if search is not None:
        tools.append(search)
    return tools


# The mutating tools gated behind human approval when --files is on (reads — incl. grep — stay
# ungated). execute joins the gate because the backend is now sandbox-capable: it runs real
# commands in cwd, OS-confined, but every run is still approved.
_WRITE_TOOLS = ("write_file", "edit_file", "execute")


def _build_fs_backend() -> object:
    """A sandbox-capable deepagents backend rooted at the launch directory.

    ``virtual_mode=True`` maps the model's ``/``-rooted paths under cwd and blocks traversal
    escapes (same containment as before for file ops). Being a ``SandboxBackendProtocol`` backend
    is what makes deepagents bind a *functional* ``execute`` — and :class:`SandboxedShellBackend`
    runs it OS-sandboxed in cwd (no network, no escape) rather than on the host shell."""
    from aai_cli.agent_cascade.sandbox import SandboxedShellBackend

    return SandboxedShellBackend(root_dir=str(Path.cwd()), virtual_mode=True)


def _graph_kwargs(
    config: CascadeConfig, *, backend_factory: Callable[[], object] = _build_fs_backend
) -> dict[str, object]:
    """Extra ``create_deep_agent`` kwargs that turn on real-cwd files + write-gating.

    Empty when ``--files`` is off, so the graph is built exactly as before. When on: a real-cwd
    backend, ``interrupt_on`` pausing only the mutating tools for human approval, and an
    in-memory checkpointer (interrupt/resume needs one). ``backend_factory`` is the test seam. No
    ``subagents`` key: deepagents auto-adds a general-purpose subagent that inherits this
    ``interrupt_on`` (so a delegated write surfaces at the same parent gate — see ``subagents.py``).
    """
    if not config.files:
        return {}
    from langgraph.checkpoint.memory import InMemorySaver

    return {
        "backend": backend_factory(),
        "interrupt_on": dict.fromkeys(_WRITE_TOOLS, True),
        "checkpointer": InMemorySaver(),
        "memory": ["./.deepagents/AGENTS.md"],
    }


def _build_middleware(config: CascadeConfig) -> list[AgentMiddleware]:
    """The live brain's extra agent middleware: a per-turn tool-call budget.

    ``ToolCallLimitMiddleware(run_limit=…, exit_behavior="continue")`` caps tool calls *per
    spoken turn* and, once the budget is hit, blocks further tool calls so the model is forced to
    answer with what it has gathered — a graceful stop rather than looping until langgraph's
    recursion backstop raises. deepagents inserts this into its own middleware stack (additive,
    so the core file/subagent/summarization middleware is untouched).
    """
    from langchain.agents.middleware import ToolCallLimitMiddleware

    return [ToolCallLimitMiddleware(run_limit=config.tool_call_limit, exit_behavior="continue")]


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
    from aai_cli.agent_cascade.model import build_model
    from aai_cli.agent_cascade.subagents import register_gp_subagent_profile

    register_gp_subagent_profile()
    model = build_model(
        api_key, model=config.model, max_tokens=config.max_tokens, extra=config.llm_extra
    )
    builtin = build_live_tools() if tools is None else list(tools)
    extra = load_mcp_tools(config.mcp_servers) if mcp_tools is None else list(mcp_tools)
    return create_deep_agent(
        model=model,
        tools=builtin + extra,
        system_prompt=build_system_prompt(
            config.system_prompt, tools=builtin, extra_tools=extra, files=config.files
        ),
        middleware=_build_middleware(config),
        **_graph_kwargs(config),
    )


def build_streamer(
    api_key: str,
    config: CascadeConfig,
    *,
    graph: CompiledAgent | None = None,
    approver: Approver | None = None,
) -> Callable[..., Iterator[SpeechDelta | ToolNotice | ApprovalPause]]:
    """A streaming reply leg for the cascade engine, backed by the deepagents graph.

    The cascade prepends its own ``system`` message each turn; the graph owns the system
    prompt, so it is dropped before streaming. The graph is driven with
    ``stream_mode="messages"`` and each top-level assistant token delta is yielded as a
    :class:`SpeechDelta`, each started tool call as a :class:`ToolNotice` (the live UI's
    affordance). Under ``-v`` the flow is logged. ``graph`` is injected in tests so the
    per-turn wiring runs against a fake with no network.

    With ``--files`` on (``config.files``) the graph gates ``write_file``/``edit_file``: a
    pending write pauses the stream, ``approver`` decides, and the turn resumes (see
    :func:`_stream_gated`). Each turn uses a fresh ``thread_id`` so the checkpointer never
    accumulates the cascade's full-history-per-turn input across turns.
    """
    resolved = build_graph(api_key, config) if graph is None else graph
    turn_ids = itertools.count()

    def stream_reply(
        messages: list[ChatCompletionMessageParam],
    ) -> Iterator[SpeechDelta | ToolNotice | ApprovalPause]:
        conversation = [message for message in messages if message.get("role") != "system"]
        run_config = (
            {"configurable": {"thread_id": f"live-{next(turn_ids)}"}} if config.files else None
        )
        return _stream_graph(
            resolved, conversation, approver=approver, config=run_config, gated=config.files
        )

    return stream_reply


def _stream_graph(
    graph: CompiledAgent,
    conversation: list[ChatCompletionMessageParam],
    *,
    approver: Approver | None = None,
    config: dict[str, object] | None = None,
    gated: bool = False,
) -> Iterator[SpeechDelta | ToolNotice | ApprovalPause]:
    """Stream one turn through the graph token-by-token, yielding speech/tool events.

    Wraps any graph failure as a CLIError (a clean ``CLIError`` passes through) so the
    cascade surfaces it instead of the reply worker dying silently. Under ``-v`` the
    accumulated assistant text, each tool call, and each tool result are logged to
    ``_FLOW_LOG``. When ``gated`` (``--files``), writes pause for ``approver`` (see
    :func:`_stream_gated`); otherwise it is a single uninterrupted stream pass.
    """
    verbose = debuglog.active()
    pending: list[str] = []  # assistant deltas accumulated for one verbose "llm:" line

    def flush_log() -> None:
        if verbose and pending:
            _FLOW_LOG.info("llm: %s", "".join(pending))
        pending.clear()

    if not hasattr(graph, "stream"):
        raise CLIError(
            "the agent couldn't complete the turn: the agent graph cannot stream",
            error_type="agent_brain_error",
        )
    try:
        # The gated path needs stream + get_state (the graph is built with a checkpointer, so it
        # always satisfies _GatedGraph); the isinstance both narrows for mypy and falls back to a
        # plain stream for the impossible non-gated-graph case.
        if gated and isinstance(graph, _GatedGraph):
            yield from _stream_gated(
                graph,
                conversation,
                approver,
                config,
                verbose=verbose,
                pending=pending,
                flush_log=flush_log,
            )
        else:
            for chunk, _m in graph.stream(
                {"messages": conversation}, config, stream_mode="messages"
            ):
                yield from _events_from_chunk(
                    chunk, verbose=verbose, pending=pending, flush_log=flush_log
                )
            flush_log()
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(
            f"the agent couldn't complete the turn: {exc}", error_type="agent_brain_error"
        ) from exc


def _stream_gated(
    graph: _GatedGraph,
    conversation: list[ChatCompletionMessageParam],
    approver: Approver | None,
    config: dict[str, object] | None,
    *,
    verbose: bool,
    pending: list[str],
    flush_log: Callable[[], None],
) -> Iterator[SpeechDelta | ToolNotice | ApprovalPause]:
    """Stream a write-gated turn: each pause on a write asks ``approver`` and resumes.

    The graph pauses (before executing a gated write) by ending the ``messages`` stream with
    a pending interrupt on the checkpointed state. We surface its action requests, bracket the
    human decision with :class:`ApprovalPause` events, and resume with the approve/reject
    ``Command`` — looping until the turn finishes without pausing.
    """
    from langgraph.types import Command

    graph_input: object = {"messages": conversation}
    while True:
        for chunk, _m in graph.stream(graph_input, config, stream_mode="messages"):
            yield from _events_from_chunk(
                chunk, verbose=verbose, pending=pending, flush_log=flush_log
            )
        flush_log()
        requests = _pending_writes(graph, config)
        if not requests:
            return
        decisions: list[dict[str, object]] = []
        for request in requests:
            yield ApprovalPause(active=True)
            decisions.append(_decide(request, approver))
            yield ApprovalPause(active=False)
        graph_input = Command(resume={"decisions": decisions})


def _pending_writes(
    graph: _GatedGraph, config: dict[str, object] | None
) -> list[dict[str, object]]:
    """The action requests of a paused gated write (empty when the turn isn't paused).

    deepagents surfaces an approval pause as ``interrupts`` on the checkpointed state, each
    interrupt's ``.value`` carrying the ``action_requests`` (the gated tool calls).
    """
    state = graph.get_state(config)
    requests: list[dict[str, object]] = []
    for interrupt in getattr(state, "interrupts", ()) or ():
        value = getattr(interrupt, "value", None)
        actions = value.get("action_requests") if isinstance(value, dict) else None
        if isinstance(actions, list):
            requests.extend(action for action in actions if isinstance(action, dict))
    return requests


def _decide(action: dict[str, object], approver: Approver | None) -> dict[str, object]:
    """Ask the approver about one pending write and shape the resume decision (reject if none)."""
    name = str(action.get("name", ""))
    args = action.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if approver is not None and approver(name, args):
        return {"type": "approve"}
    return {"type": "reject", "message": _DECLINED}


def _events_from_chunk(
    chunk: object, *, verbose: bool, pending: list[str], flush_log: Callable[[], None]
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
            yield ToolNotice(_tool_label(name), _tool_fillers(name))
    text = _content_text(getattr(chunk, "content", ""))
    if text:
        pending.append(text)
        yield SpeechDelta(text)


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


def _content_text(content: object) -> str:
    """Coerce a message's content (a string, or a list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)
