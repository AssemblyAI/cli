from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text

from aai_cli.agent import events
from aai_cli.ui.render import BaseRenderer

if TYPE_CHECKING:
    from aai_cli.agent_cascade.plan import TodoItem

# Single-cell status marks for the plan, pipe-safe and aligned (the TUI styles its own).
_TODO_MARKS = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}


def _mark(status: str) -> str:
    """The checklist marker for a todo status, falling back to the pending box for an unknown one."""
    return _TODO_MARKS.get(status, "[ ]")


def _labeled(label: str, body: str, *, style: str = "aai.label") -> Text:
    """A transcript line tinted entirely in `style` — both the `label` prefix and the body.

    Coloring the whole turn (not just the prefix) is what makes a "you:" line and an
    "agent:" line read as different colors top to bottom; the two styles resolve to
    contrasting hues (see ``theme.aai.you`` / ``theme.aai.agent``).
    """
    return Text(f"{label}{body}", style=style)


class AgentRenderer(BaseRenderer):
    """Renders Voice Agent events in one of three modes.

    - JSON: NDJSON events to stdout. - text: plain ``you:``/``agent:`` transcript
    lines to stdout with status on stderr (so ``assembly agent -o text | assembly llm "…"``
    pipes the conversation). - human (default): live Rich transcript.

    Audio payloads are never written; only text/state events are surfaced.
    """

    def __init__(self, *, mic_input: bool = True, **kwargs: Any) -> None:
        # text_mode/err/json_mode/out/console are handled by BaseRenderer.
        super().__init__(**kwargs)
        # File-driven runs have no mic, so they skip the "start talking" prompt.
        self.mic_input = mic_input

    def _emit_event(self, event: events.Event) -> None:
        """Serialize one typed agent event to the NDJSON stream."""
        self._emit(event.model_dump())

    # --- lifecycle ---------------------------------------------------------
    def connected(self) -> None:
        if self.json_mode:
            self._emit_event(events.SessionReady())
        elif not self.mic_input:
            return
        elif self.text_mode:
            self._status("Connected — start talking. (Ctrl-C to stop)")
        else:
            self._line(Text("Connected — start talking. (Ctrl-C to stop)", style="aai.muted"))

    def notice(self, text: str) -> None:
        """Print a human-facing notice: suppressed in JSON, to stderr otherwise.

        Stderr in *every* non-JSON mode (not just ``-o text``): the default human
        mode is also piped sometimes (``assembly agent | head``), and a notice on stdout
        would be consumed as transcript data there.
        """
        if self.json_mode:
            return
        self._status(text.rstrip("\n"))

    # --- user --------------------------------------------------------------
    def user_partial(self, text: str) -> None:
        if self.json_mode:
            self._emit_event(events.UserDelta(text=text))
        elif not self.text_mode:  # partials are noise for piped text
            self._update_line(_labeled("you: ", text, style="aai.you"))

    def user_final(self, text: str) -> None:
        if self.json_mode:
            self._emit_event(events.UserFinal(text=text))
        elif self.text_mode:
            self._write(f"you: {text}\n")
        else:
            self._finalize_line(_labeled("you: ", text, style="aai.you"))

    def tool_call(self, label: str) -> None:
        """Surface that the agent is using a tool (e.g. "Searching the web") while it thinks.

        JSON emits a ``tool.use`` event; piped text keeps it off stdout (transcript-only) by
        routing to stderr; human mode shows a muted inline line.
        """
        if self.json_mode:
            self._emit_event(events.ToolUse(label=label))
        elif self.text_mode:
            self._status(f"{label}…")
        else:
            self._line(_labeled("", f"{label}…", style="aai.muted"))

    def todos_updated(self, todos: tuple[TodoItem, ...]) -> None:
        """Surface the agent's plan (its ``write_todos`` list), replacing any prior plan.

        JSON emits a ``plan`` event; piped text routes a compact one-line summary to stderr
        (transcript-only stdout); human mode shows a muted multi-line checklist.
        """
        if self.json_mode:
            items = tuple(events.TodoItem(content=t.content, status=t.status) for t in todos)
            self._emit_event(events.PlanUpdate(todos=items))
        elif self.text_mode:
            self._status("Plan: " + "; ".join(f"{_mark(t.status)} {t.content}" for t in todos))
        else:
            self._line(_labeled("", "Plan:", style="aai.muted"))
            for todo in todos:
                self._line(
                    _labeled("  ", f"{_mark(todo.status)} {todo.content}", style="aai.muted")
                )

    # --- agent -------------------------------------------------------------
    def reply_started(self) -> None:
        if self.json_mode:
            self._emit_event(events.ReplyStarted())

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit_event(events.AgentTranscript(text=text, interrupted=interrupted))
        elif self.text_mode:
            self._write(f"agent: {text}\n")
        else:
            # commits any open "you: …" partial first
            self._line(_labeled("agent: ", text, style="aai.agent"))

    def reply_done(self, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit_event(events.ReplyDone(interrupted=interrupted))
