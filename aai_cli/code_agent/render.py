"""Rich rendering for the coding agent's non-TUI / headless path.

The Textual TUI (`tui.py`) is the primary front-end; this renders the same
:mod:`~aai_cli.code_agent.events` to the plain Rich consoles for piped/headless runs
and as the simple fallback. Errors/notices go to stderr so stdout stays clean.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.markup import escape

from aai_cli.code_agent.events import AssistantText, ErrorText, Event, ToolCall, ToolResult
from aai_cli.code_agent.summarize import summarize_call, summarize_result
from aai_cli.ui import output


class RichRenderer:
    """An :data:`~aai_cli.code_agent.session.EventSink` that prints to the Rich console."""

    def __call__(self, event: Event) -> None:
        # escape() dynamic content so a model/tool string with "[" can't inject Rich
        # markup or raise MarkupError (matches the inline-escape convention in output.py).
        if isinstance(event, AssistantText):
            # Render as Markdown so fenced code blocks are syntax-highlighted (and lists/
            # headings format) instead of showing raw ``` markers — Markdown parses its own
            # syntax, not console markup, so no escape()/injection concern.
            output.console.print(Markdown(event.text))
        elif isinstance(event, ToolCall):
            output.console.print(
                f"[aai.muted]→ {escape(summarize_call(event.name, event.args))}[/aai.muted]"
            )
        elif isinstance(event, ToolResult):
            preview = escape(summarize_result(event.content))
            output.console.print(f"[aai.muted]  {escape(event.name)}: {preview}[/aai.muted]")
        elif isinstance(event, ErrorText):
            output.error_console.print(output.fail(escape(event.text)))

    def notice(self, text: str) -> None:
        """A dim advisory on stderr (so it never pollutes piped stdout)."""
        output.error_console.print(output.hint(text))
