"""The per-turn streaming reply leg for the live voice cascade.

:func:`build_streamer` drives the deepagents graph (built by :mod:`brain`) one turn at a time
with ``stream_mode="messages"``, translating each streamed chunk into the speech/tool/plan events
the cascade engine consumes — :class:`~aai_cli.agent_cascade.brain.SpeechDelta` tokens to speak,
:class:`~aai_cli.agent_cascade.brain.ToolNotice` affordances, :class:`~aai_cli.agent_cascade.plan.TodoUpdate`
plans, and (under ``--files``) :class:`~aai_cli.agent_cascade.brain.ApprovalPause` brackets around
a human write approval. Split from :mod:`brain` (which assembles the graph) along the module's
natural build-vs-drive seam to keep each file within the length gate; the dependency is
one-directional (``streamer`` imports ``brain``, never the reverse).

The graph is the only network seam: ``build_streamer`` accepts an injected graph, so this leg is
unit-tested against a fake with no sockets (see ``tests/test_agent_cascade_streamer.py``).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from aai_cli.agent_cascade import plan
from aai_cli.agent_cascade.brain import (
    ApprovalPause,
    Approver,
    CompiledAgent,
    SpeechDelta,
    ToolNotice,
    _GatedGraph,
    _tool_fillers,
    _tool_label,
    build_graph,
)
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.core import debuglog
from aai_cli.core.errors import CLIError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Verbose (`-v`) flow logging for the agent's tool loop. `invoke` runs the whole loop internally,
# so without this `-v` only shows the httpx request lines and never which tools the agent reached
# for or what they returned — exactly what you need when a spoken turn stalls mid-tool. Logged at
# INFO so plain `-v` surfaces it.
_FLOW_LOG = logging.getLogger("aai_cli.agent_cascade.streamer")

# Tool outputs (a fetched page, a search payload) can be huge; cap what we log per result so a
# single tool call doesn't bury the rest of the flow in stderr. The exact cap is an arbitrary
# tuning knob — a +-1 shift is behaviorally equivalent, so no test can kill it.
_RESULT_LOG_CAP = 500  # pragma: no mutate

# Message handed back to the model when the user declines a write (matches the code agent's copy).
_DECLINED = "User declined to run this tool."


def build_streamer(
    api_key: str,
    config: CascadeConfig,
    *,
    graph: CompiledAgent | None = None,
    approver: Approver | None = None,
) -> Callable[..., Iterator[SpeechDelta | ToolNotice | plan.TodoUpdate | ApprovalPause]]:
    """A streaming reply leg for the cascade engine, backed by the deepagents graph.

    The cascade prepends its own ``system`` message each turn; the graph owns the system
    prompt, so it is dropped before streaming. The graph is driven with
    ``stream_mode="messages"`` and each top-level assistant token delta is yielded as a
    :class:`SpeechDelta`, each started tool call as a :class:`ToolNotice` (the live UI's
    affordance), and each ``write_todos`` plan as a :class:`plan.TodoUpdate`. Under ``-v`` the
    flow is logged. ``graph`` is injected in tests so the per-turn wiring runs against a fake with
    no network.

    With ``--files`` on (``config.files``) the graph gates ``write_file``/``edit_file``: a
    pending write pauses the stream, ``approver`` decides, and the turn resumes (see
    :func:`_stream_gated`). Each turn uses a fresh ``thread_id`` so the checkpointer never
    accumulates the cascade's full-history-per-turn input across turns.
    """
    resolved = build_graph(api_key, config) if graph is None else graph
    turn_ids = itertools.count()

    def stream_reply(
        messages: list[ChatCompletionMessageParam],
    ) -> Iterator[SpeechDelta | ToolNotice | plan.TodoUpdate | ApprovalPause]:
        conversation = [message for message in messages if message.get("role") != "system"]
        run_config: dict[str, object] | None = (
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
) -> Iterator[SpeechDelta | ToolNotice | plan.TodoUpdate | ApprovalPause]:
    """Stream one turn through the graph token-by-token, yielding speech/tool/plan events.

    Wraps any graph failure as a CLIError (a clean ``CLIError`` passes through) so the
    cascade surfaces it instead of the reply worker dying silently. Under ``-v`` the
    accumulated assistant text, each tool call, and each tool result are logged to
    ``_FLOW_LOG``. When ``gated`` (``--files``), writes pause for ``approver`` (see
    :func:`_stream_gated`); otherwise it is a single uninterrupted stream pass.
    """
    verbose = debuglog.active()
    pending: list[str] = []  # assistant deltas accumulated for one verbose "llm:" line
    todos = plan.TodoCollector()  # accumulates a write_todos call's args across the turn's chunks

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
                todos=todos,
            )
        else:
            for chunk, _m in graph.stream(
                {"messages": conversation}, config, stream_mode="messages"
            ):
                yield from _events_from_chunk(
                    chunk, verbose=verbose, pending=pending, flush_log=flush_log, todos=todos
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
    todos: plan.TodoCollector,
) -> Iterator[SpeechDelta | ToolNotice | plan.TodoUpdate | ApprovalPause]:
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
                chunk, verbose=verbose, pending=pending, flush_log=flush_log, todos=todos
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
    chunk: object,
    *,
    verbose: bool,
    pending: list[str],
    flush_log: Callable[[], None],
    todos: plan.TodoCollector,
) -> Iterator[SpeechDelta | ToolNotice | plan.TodoUpdate]:
    """Translate one streamed message chunk into speech/tool/plan events (and verbose logs).

    ``write_todos`` is special-cased: ``todos`` accumulates its args and surfaces them as a
    :class:`plan.TodoUpdate` when the tool result lands — never a generic :class:`ToolNotice`.
    """
    if type(chunk).__name__ == "ToolMessage":
        yield from _tool_result_events(chunk, verbose=verbose, flush_log=flush_log, todos=todos)
        return
    todos.note_chunk(chunk)
    yield from _tool_call_notices(chunk, verbose=verbose, flush_log=flush_log)
    text = _content_text(_content(chunk))
    if text:
        pending.append(text)
        yield SpeechDelta(text)


def _tool_result_events(
    chunk: object, *, verbose: bool, flush_log: Callable[[], None], todos: plan.TodoCollector
) -> Iterator[plan.TodoUpdate]:
    """Handle a ToolMessage: log its result and, for ``write_todos``, surface the reassembled plan."""
    flush_log()
    name = getattr(chunk, "name", "")
    if verbose:
        _FLOW_LOG.info("tool result %s -> %s", name, _clip(_content_text(_content(chunk))))
    if name == plan.WRITE_TODOS_TOOL_NAME and (update := todos.take()) is not None:
        yield update  # the plan, reassembled from the call's args


def _tool_call_notices(
    chunk: object, *, verbose: bool, flush_log: Callable[[], None]
) -> Iterator[ToolNotice]:
    """Emit a :class:`ToolNotice` for each started tool call (``write_todos`` surfaces as a plan)."""
    for call in getattr(chunk, "tool_call_chunks", None) or []:
        name = call.get("name")
        # The plan tool surfaces as a TodoUpdate (in _tool_result_events), not an affordance.
        if name and name != plan.WRITE_TODOS_TOOL_NAME:
            flush_log()
            if verbose:
                _FLOW_LOG.info("tool call %s", name)
            yield ToolNotice(_tool_label(name), _tool_fillers(name))


def _content(chunk: object) -> object:
    """A chunk's raw ``content`` attribute (empty string when absent)."""
    return getattr(chunk, "content", "")


def _clip(text: str) -> str:
    """Flatten a tool result onto one line and truncate it for the flow log.

    Tool output is untrusted, so whitespace (newlines especially) is collapsed first — a result
    can't then forge extra ``[aai_cli.…]`` log lines — and the length is capped so a multi-KB
    payload can't bury the flow. (Secrets are masked separately by the debuglog formatter.)
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
