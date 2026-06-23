"""Mounted transcript widgets for the coding-agent TUI.

The transcript is a ``VerticalScroll`` of these widgets rather than an append-only ``RichLog``,
which buys two things deepagents-code has: the assistant reply updates *in place* as it streams
(no separate live region), and a tool's output is a collapsible row — a clipped preview that
expands to the full output on Ctrl+O or a click.

Dynamic content (model/tool/user strings) is wrapped in ``rich.text.Text`` so it's shown
literally — Text doesn't parse console markup, so a stray ``[`` can't raise or inject styling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static

from aai_cli.agent_cascade import banner
from aai_cli.agent_cascade.summarize import summarize_call, summarize_result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aai_cli.agent_cascade.plan import TodoItem

_DIM = "#8a8f98"  # muted gray for tool lines / notes
_ERROR = "#f04438"
_DONE = "#22c55e"  # green for a completed task (the voice bar's "speaking" hue)


class Note(Static):
    """A dim one-line transcript aside (``cancelling…``, ``copied…``, ``voice off…``)."""

    def __init__(self, text: str) -> None:
        super().__init__(Text(text, style=_DIM))


class ToolAffordance(Static):
    """A dim live tool-call line: the friendly label plus its identifying arg.

    ``Searching the web · ai house Seattle…``. Distinct from :class:`ToolCallLine` (the coding
    agent's ``→ name(args)`` form) — this is the voice TUI's progress affordance, spaced by
    ``LiveAgentApp``: ``tight`` adds the ``-tight`` class so a consecutive call drops the top
    margin the first call of a turn keeps.
    """

    def __init__(self, text: str, *, tight: bool) -> None:
        super().__init__(Text(text, style=_DIM), classes="-tight" if tight else None)


def _user_markup(text: str) -> Text:
    """The styled `» …` prompt echo, built in one place for the constructor and set_text."""
    return Text(f"» {text}", style="bold #38bdf8")


class UserMessage(Static):
    """The echoed user prompt, with a top margin so each turn is visually separated."""

    DEFAULT_CSS = "UserMessage { margin-top: 1; }"

    def __init__(self, text: str) -> None:
        super().__init__(_user_markup(text))

    def set_text(self, text: str) -> None:
        """Replace the shown prompt text — grows an interim voice transcript in place."""
        self.update(_user_markup(text))


class AssistantMessage(Static):
    """The assistant's reply: streams plain text token-by-token, then renders as Markdown."""

    def __init__(self) -> None:
        super().__init__()
        self._tokens: list[str] = []  # accumulate tokens, not str +=, to avoid quadratic growth

    @property
    def text(self) -> str:
        """The reply text streamed so far (used to finalize a cancelled generation)."""
        return "".join(self._tokens)

    def stream(self, delta: str) -> None:
        """Append a streamed token and repaint as plain text (cheap; no per-token markdown)."""
        self._tokens.append(delta)
        self.update(Text(self.text))

    def finalize(self, text: str) -> None:
        """Replace the streamed text with the authoritative reply, rendered as Markdown."""
        self._tokens = [text]
        self.update(Markdown(text))


class ToolCallLine(Static):
    """A compact tool-call line, e.g. ``→ write_file(app.py)``."""

    def __init__(self, name: str, args: Mapping[str, object]) -> None:
        super().__init__(Text(f"→ {summarize_call(name, args)}", style=_DIM))


# Per-status glyph + style for a plan line: a green ✓ when done, a brand-accented ▸ for the
# task in progress, a dim ○ for one still pending (an unknown status falls back to pending).
_TODO_STYLES: dict[str, tuple[str, str]] = {
    "completed": ("✓", _DONE),
    "in_progress": ("▸", banner.BRAND_HEX),
    "pending": ("○", _DIM),
}


def _todos_markup(todos: Sequence[TodoItem]) -> Text:
    """Render the plan as a ``Plan`` heading above one styled line per task.

    Completed tasks are struck through and dimmed; the in-progress task is brand-accented; the
    rest are dim — so the panel reads as a live checklist of the spoken plan at a glance.
    """
    text = Text("Plan", style=f"bold {_DIM}")
    for todo in todos:
        glyph, color = _TODO_STYLES.get(todo.status, _TODO_STYLES["pending"])
        style = f"{color} strike" if todo.status == "completed" else color
        text.append(f"\n{glyph} ", style=color)
        text.append(todo.content, style=style)
    return text


class TodoList(Static):
    """The agent's task plan (its ``write_todos`` list), updated in place as the plan evolves.

    ``write_todos`` replaces the whole list on each call, so one widget shows the current plan
    (revised in place) rather than a fresh copy per revision — the visible counterpart to a
    hands-free "first I'll…, then I'll…". Spaced off the turn above by a top margin.
    """

    DEFAULT_CSS = "TodoList { margin-top: 1; }"

    def __init__(self, todos: Sequence[TodoItem]) -> None:
        super().__init__()
        self.set_todos(todos)

    def set_todos(self, todos: Sequence[TodoItem]) -> None:
        """Repaint the panel with the current plan (replacing whatever it showed before)."""
        self.update(_todos_markup(todos))


class ErrorMessage(Static):
    """A failed turn, shown instead of crashing the UI."""

    def __init__(self, text: str) -> None:
        super().__init__(Text(f"✗ {text}", style=_ERROR))


class ToolOutput(Static):
    """A tool's output: a clipped preview that expands to the full content (Ctrl+O / click)."""

    def __init__(self, name: str, content: str) -> None:
        super().__init__()
        self._name = name
        self._full = content.strip()
        self._preview = summarize_result(content)
        self._expandable = self._preview != self._full  # nothing to expand when it fits already
        self._expanded = False

    def on_mount(self) -> None:
        self._repaint()

    def on_click(self) -> None:
        self.toggle()

    def toggle(self) -> None:
        """Flip between the clipped preview and the full output (no-op when it all fits)."""
        if not self._expandable:
            return
        self._expanded = not self._expanded
        self._repaint()

    def _repaint(self) -> None:
        body = self._full if self._expanded else self._preview
        line = Text(f"  {self._name}: ", style=_DIM)
        line.append(body, style=_DIM)
        if self._expandable:
            hint = " (Ctrl+O to collapse)" if self._expanded else " (Ctrl+O to expand)"
            line.append(hint, style=f"{_DIM} italic")
        self.update(line)
