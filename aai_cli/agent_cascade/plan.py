"""The `assembly live` plan subsystem: the agent's ``write_todos`` task list.

deepagents auto-installs langchain's ``TodoListMiddleware``, so the live agent can lay out a
multi-step plan with ``write_todos`` (and revise it as it works). This module owns the plan's
data shape (:class:`TodoItem` / :class:`TodoUpdate`) and the :class:`TodoCollector` that
reassembles a ``write_todos`` call's args out of the streamed message chunks — kept apart from
``brain`` (which builds and drives the graph) so neither module crowds the 500-line gate. The
collector is fed each chunk by ``streamer._events_from_chunk`` and surfaces a :class:`TodoUpdate`
when the call's tool result lands; ``engine``/the renderers then show the plan as a checklist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# The planning/task-list tool deepagents auto-installs (langchain's TodoListMiddleware). Its
# call is surfaced as a TodoUpdate (the visible plan), not a generic tool affordance — so the
# name is recognized rather than carried in brain's _TOOL_LABELS.
WRITE_TODOS_TOOL_NAME = "write_todos"


@dataclass(frozen=True)
class TodoItem:
    """One task in the agent's plan: its text and lifecycle status.

    ``status`` is one of ``pending`` / ``in_progress`` / ``completed`` (langchain's todo
    schema); it is kept as a plain string so an unknown future status renders rather than raises.
    """

    content: str
    status: str


@dataclass(frozen=True)
class TodoUpdate:
    """The agent's current task list, emitted when it (re)writes its plan via ``write_todos``.

    Carries the *whole* list each time (``write_todos`` replaces the list outright), so the live
    UI renders the plan in place rather than appending. Purely visual — never spoken — it's the
    on-screen counterpart to a hands-free "first I'll…, then I'll…".
    """

    todos: tuple[TodoItem, ...]


def _parse_todos(args_json: str) -> TodoUpdate | None:
    """Parse a ``write_todos`` args blob (``{"todos": [{content, status}, …]}``) into a TodoUpdate.

    Returns ``None`` for anything that doesn't parse to a non-empty todo list, so a malformed or
    still-partial fragment is dropped rather than surfaced as an empty plan.
    """
    try:
        payload = json.loads(args_json)
    except (ValueError, TypeError):
        return None
    return _todos_from_list(payload.get("todos") if isinstance(payload, dict) else None)


def _todos_from_list(raw: object) -> TodoUpdate | None:
    """Shape a raw ``todos`` list (of ``{content, status}`` dicts) into a TodoUpdate, or ``None``.

    Shared by the streamed-JSON path (:func:`_parse_todos`) and the non-streaming model path
    (a chunk's already-parsed ``.tool_calls`` args), so both yield the same event.
    """
    if not isinstance(raw, list):
        return None
    items = tuple(
        TodoItem(content=str(item.get("content", "")), status=str(item.get("status", "")))
        for item in raw
        if isinstance(item, dict)
    )
    return TodoUpdate(items) if items else None


class TodoCollector:
    """Accumulates a ``write_todos`` call's streamed args, parsing them into a :class:`TodoUpdate`.

    A streaming model emits the plan as JSON arg fragments across many AIMessage chunks; a
    non-streaming model (the test fakes) emits one chunk with already-parsed ``.tool_calls``
    args. Either way the call completes when its ``write_todos`` ToolMessage lands, at which point
    :meth:`take` shapes whatever was gathered into the event. At most one ``write_todos`` call is
    open at a time (langchain's planning middleware forbids parallel calls), so a single buffer
    suffices; a fresh call resets it (``write_todos`` replaces the whole list).
    """

    def __init__(self) -> None:
        self._index: int | None = None
        self._buffer = ""  # accumulated JSON arg fragments (streaming path)
        self._args: dict[str, object] | None = None  # complete args dict (non-streaming path)

    def note_chunk(self, chunk: object) -> None:
        """Record a chunk's contribution to the in-flight ``write_todos`` call (if any)."""
        self._note_complete_args(chunk)
        self._note_arg_fragments(chunk)

    def _note_complete_args(self, chunk: object) -> None:
        """Capture a non-streaming model's complete ``write_todos`` args dict (``.tool_calls``)."""
        for call in getattr(chunk, "tool_calls", None) or []:
            if call.get("name") == WRITE_TODOS_TOOL_NAME and isinstance(call.get("args"), dict):
                self._args = call["args"]

    def _note_arg_fragments(self, chunk: object) -> None:
        """Accumulate a streaming model's incremental ``write_todos`` JSON arg fragments."""
        for call in getattr(chunk, "tool_call_chunks", None) or []:
            index = call.get("index")
            if call.get("name") == WRITE_TODOS_TOOL_NAME:
                self._index, self._buffer = index, call.get("args") or ""
            elif self._index is not None and index == self._index:
                self._buffer += call.get("args") or ""

    def take(self) -> TodoUpdate | None:
        """Shape the gathered plan into a :class:`TodoUpdate` (``None`` if empty) and reset.

        The streamed JSON buffer is preferred — it's the complete, raw args at ToolMessage time —
        and the chunk's ``.tool_calls`` dict is only the fallback for a non-streaming model, since
        a streaming chunk's auto-derived ``.tool_calls`` can hold a *partial* parse of the args.
        """
        args, buffer = self._args, self._buffer
        self._args, self._index, self._buffer = None, None, ""
        return _parse_todos(buffer) or (
            _todos_from_list(args.get("todos")) if args is not None else None
        )
