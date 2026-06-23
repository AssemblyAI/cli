"""Deepagents-powered reply brain for the live voice cascade.

`assembly live` answers each spoken turn with a deepagents graph instead of a single
LLM completion, so the agent can transparently reach for a tool — web search —
mid-conversation, mimicking a live multimodal assistant (the "talk to Gemini Live"
experience). The toolset is deliberately minimal: a low-latency spoken turn does best
with one obvious tool rather than a menu it has to choose among. The graph is built once
per session (:func:`build_graph`); tools are read-only and auto-approved, because a spoken
turn can't pause for a keyboard confirmation, and the system prompt keeps every reply short
and speakable. Context-window management is deepagents' job (its built-in
``SummarizationMiddleware``), so the engine feeds the full untrimmed history each turn.

This module owns graph *assembly* (tools, backend, middleware, the compiled graph) plus the
shared stream-event types (:class:`SpeechDelta`/:class:`ToolNotice`/:class:`ApprovalPause`)
and tool affordance vocabulary. Driving the graph turn-by-turn lives beside it in
:mod:`aai_cli.agent_cascade.streamer` (``build_streamer``) — the natural build-vs-drive seam,
split out to keep each file within the length gate; that streaming leg is what the cascade's
STT/LLM/TTS injection seam exercises against a fake graph with no sockets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aai_cli.agent_cascade import datetime_tool, weather_tool, webpage_tool
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.firecrawl_search import WEB_SEARCH_TOOL_NAME
from aai_cli.agent_cascade.prompt import build_system_prompt
from aai_cli.agent_cascade.write_gate import write_interrupt_on

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import BaseTool


class CompiledAgent(Protocol):
    """The slice of the compiled langgraph graph the live reply leg drives.

    A structural type so we needn't name langgraph's deeply-generic
    ``CompiledStateGraph`` (and don't drag its type params through our code).
    """

    def invoke(
        self, input: object, config: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Run one step of the graph, returning the updated state (incl. messages)."""


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
# rotates across turns; unknown/MCP tools fall back to `_GENERIC_FILLERS` (spoken-style, no markdown).
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
# The streaming leg that consults it lives in :mod:`aai_cli.agent_cascade.streamer`.
Approver = Callable[[str, dict[str, object]], bool]


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
    backend, a path-scoped ``interrupt_on`` (writes outside the ``--auto-write`` subtrees pause
    for approval; ``execute`` always does — see :func:`write_gate.write_interrupt_on`), and an
    in-memory checkpointer (interrupt/resume needs one). ``backend_factory`` is the test seam. No
    ``subagents`` key: deepagents auto-adds a general-purpose subagent that inherits this
    ``interrupt_on`` (see ``subagents.py``).
    """
    if not config.files:
        return {}
    from langgraph.checkpoint.memory import InMemorySaver

    return {
        "backend": backend_factory(),
        "interrupt_on": write_interrupt_on(config.auto_write_paths),
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
            config.system_prompt,
            tools=builtin,
            extra_tools=extra,
            files=config.files,
            project_context=config.project_context,
        ),
        middleware=_build_middleware(config),
        **_graph_kwargs(config),
    )
