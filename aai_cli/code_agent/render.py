"""Rich rendering for the coding agent's non-TUI / headless path.

The Textual TUI (`tui.py`) is the primary front-end; this renders the same
:mod:`~aai_cli.code_agent.events` to the plain Rich consoles for piped/headless runs
and as the simple fallback. Errors/notices go to stderr so stdout stays clean.
"""

from __future__ import annotations

from collections.abc import Callable

from aai_cli.code_agent.events import AssistantText, ErrorText, Event, ToolCall, ToolResult
from aai_cli.code_agent.session import Approver
from aai_cli.ui import output

# Tool output can be long; clip it for the inline transcript.
_RESULT_PREVIEW = 2000


def _format_args(args: dict[str, object]) -> str:
    """A compact one-line view of a tool call's arguments."""
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


class RichRenderer:
    """An :data:`~aai_cli.code_agent.session.EventSink` that prints to the Rich console."""

    def __call__(self, event: Event) -> None:
        if isinstance(event, AssistantText):
            output.console.print(event.text)
        elif isinstance(event, ToolCall):
            output.console.print(
                f"[aai.muted]→ {event.name}({_format_args(event.args)})[/aai.muted]"
            )
        elif isinstance(event, ToolResult):
            preview = event.content.strip()[:_RESULT_PREVIEW]
            output.console.print(f"[aai.muted]  {event.name}: {preview}[/aai.muted]")
        elif isinstance(event, ErrorText):
            output.error_console.print(output.fail(event.text))

    def notice(self, text: str) -> None:
        """A dim advisory on stderr (so it never pollutes piped stdout)."""
        output.error_console.print(output.hint(text))


def make_approver(confirm: Callable[[str, dict[str, object]], bool]) -> Approver:
    """Build the approval callback from a yes/no ``confirm`` prompt.

    Kept thin so the interactive confirm (stdin read) is injected by the command and
    a test passes a scripted decision.
    """

    def approver(name: str, args: dict[str, object]) -> bool:
        return confirm(name, args)

    return approver
