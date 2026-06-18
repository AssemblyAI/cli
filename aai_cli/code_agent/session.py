"""Drive the compiled agent turn-by-turn, framework-agnostically.

`CodeSession.send` runs one user turn: it invokes the graph, resolves any
human-in-the-loop approval interrupts (asking the injected ``approver``), and emits
display events to the injected ``sink``. Both the Rich renderer and the Textual TUI
sit behind those two callables, so the orchestration here is unit-tested with a fake
chat model and plain functions — no terminal, no framework.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aai_cli.code_agent.agent import CompiledAgent
from aai_cli.code_agent.events import (
    ErrorText,
    Event,
    interrupt_request,
    message_events,
    new_messages,
)

# Given a pending tool's name and arguments, decide whether to run it.
Approver = Callable[[str, dict[str, object]], bool]
# Receives each display event as the turn unfolds.
EventSink = Callable[[Event], None]

# Lines that end the interactive loop.
QUIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})

_DECLINED = "User declined to run this tool."


@runtime_checkable
class _SupportsStream(Protocol):
    """An agent that can stream its run as incremental state snapshots.

    The real compiled graph supports this; the unit-test fakes that only implement
    ``invoke`` don't, so :meth:`CodeSession._run` falls back to a single emit for them.
    """

    def stream(
        self, graph_input: object, config: Mapping[str, object] | None, *, stream_mode: str
    ) -> Iterator[dict[str, object]]:
        """Yield the running state (incl. the growing ``messages``) after each super-step."""


@dataclass
class CodeSession:
    """One coding conversation: a compiled agent plus the I/O seams that render it."""

    agent: CompiledAgent
    sink: EventSink
    approver: Approver
    thread_id: str = "code"
    auto_approve: bool = False
    _seen: int = field(default=0, init=False)
    _cancel: threading.Event = field(
        default_factory=threading.Event,
        init=False,  # pragma: no mutate
    )

    def _config(self) -> dict[str, object]:
        return {"configurable": {"thread_id": self.thread_id}}

    def request_cancel(self) -> None:
        """Ask the running turn to stop its agent loop at the next step boundary.

        Set from another thread (the TUI's Ctrl-C / Escape); the streaming loop in
        :meth:`_run` and the approval loop both check it, so a long tool sequence stops
        without having to kill the worker thread mid-step.
        """
        self._cancel.set()

    def send(self, text: str) -> None:
        """Run one user turn, resolving approvals and emitting events as each step lands.

        Events stream out incrementally (responsive UI) and :meth:`request_cancel` can stop
        the loop early. A failure inside the graph (a gateway 5xx, a tool blowing up) is
        surfaced as an ``ErrorText`` event rather than propagating — a single bad turn must
        not crash the TUI worker or the REPL; the user can just try again.
        """
        self._cancel.clear()
        config = self._config()
        try:
            result = self._run({"messages": [{"role": "user", "content": text}]}, config)
            self._resolve_interrupts(result, config)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.sink(ErrorText(f"{type(exc).__name__}: {exc}"))
            return

    def _run(self, graph_input: object, config: dict[str, object]) -> dict[str, object]:
        """Drive one graph segment, emitting events as each step completes; return the end state.

        Streaming (``stream_mode="values"``) renders intermediate tool calls/results live and
        lets :meth:`request_cancel` break the loop between steps. A double that only implements
        ``invoke`` (the TUI/REPL test fakes) emits once at the end instead.
        """
        if isinstance(self.agent, _SupportsStream):
            last: dict[str, object] = {}
            for chunk in self.agent.stream(graph_input, config, stream_mode="values"):
                if self._cancel.is_set():
                    break
                self._emit_new(chunk)
                last = chunk
            return last
        result = self.agent.invoke(graph_input, config)
        self._emit_new(result)
        return result

    def _resolve_interrupts(
        self, result: dict[str, object], config: dict[str, object]
    ) -> dict[str, object]:
        """Loop approving/rejecting gated tool calls until the turn no longer pauses."""
        from langgraph.types import Command

        while True:
            if self._cancel.is_set():
                return result
            request = interrupt_request(result)
            if request is None:
                return result
            actions = request.get("action_requests")
            actions = actions if isinstance(actions, list) else []
            decisions = [self._decide(action) for action in actions]
            result = self._run(Command(resume={"decisions": decisions}), config)

    def _decide(self, action: dict[str, object]) -> dict[str, object]:
        """Ask the approver about one pending tool call and shape the resume decision."""
        name = str(action.get("name", ""))
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if self.approver(name, args):
            return {"type": "approve"}
        return {"type": "reject", "message": _DECLINED}

    def _emit_new(self, result: dict[str, object]) -> None:
        """Emit display events for every message added during the turn."""
        for message in new_messages(result, self._seen):
            for event in message_events(message, announce_calls=self.auto_approve):
                self.sink(event)
        messages = result.get("messages")
        if isinstance(messages, list):
            self._seen = len(messages)


def run_repl(
    session: CodeSession, *, read_line: Callable[[], str | None], initial: str | None = None
) -> None:
    """Run the read-eval loop: send ``initial`` (if any), then each line ``read_line`` yields.

    Stops on EOF (``read_line`` returns ``None``) or a quit command. Blank lines are
    skipped. The reader is injected so tests script a conversation without a TTY.
    """
    if initial:
        session.send(initial)
    while True:
        line = read_line()
        if line is None:
            return
        line = line.strip()
        if not line:
            continue
        if line in QUIT_COMMANDS:
            return
        session.send(line)
