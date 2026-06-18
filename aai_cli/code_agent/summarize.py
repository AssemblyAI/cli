"""Compact one-line summaries of tool activity, shared by both front-ends.

A coding agent's tool args and output are routinely whole files or long command output.
Dumping them verbatim into the transcript buries the conversation — and, because args go
through ``repr``, renders literal ``\\n`` escapes. Both the Textual TUI (`tui.py`) and the
Rich fallback (`render.py`) route tool calls/results through these helpers so the
transcript stays scannable, mirroring how deepagents-code's collapsible tool rows show
just the identifying arg (a filename / command) and a short output preview with a
"+N more lines" tail rather than the full payload.
"""

from __future__ import annotations

from collections.abc import Mapping

# Output preview budget (deepagents-code previews tool output at 4 lines / 300 chars behind
# an expand toggle; our append-only log has no expander, so we clip and tag the remainder).
_PREVIEW_LINES = 4
_PREVIEW_CHARS = 300
# Per-arg and arg-count caps so one giant value (a file's contents) can't flood the line.
_MAX_ARG_VALUE = 60
_MAX_ARGS = 3
# Per-value cap for the *expanded* approval view: values shown whole (newlines kept) but bounded
# so a multi-megabyte file can't make the modal unbounded.
_EXPANDED_VALUE = 1000
# Args that identify a call on their own — show only this and elide bulky siblings (content).
_IDENTITY_ARGS = ("file_path", "path", "filename", "command", "url", "query", "pattern")


def _one_line(value: object, *, limit: int) -> str:
    """Collapse ``value`` to a single clipped line (newlines → spaces, ellipsis if long)."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_args(args: Mapping[str, object]) -> str:
    """The compact arg view shared by the transcript line and the approval prompt.

    Prefers a single identifying arg (a path/command/URL) so a ``write_file`` reads as
    ``app.py`` instead of inlining the file being written; otherwise shows up to a few
    short ``key=value`` args, each clipped, with a trailing ``…`` when more were elided.
    """
    for key in _IDENTITY_ARGS:
        if key in args:
            return _one_line(args[key], limit=_MAX_ARG_VALUE)
    shown = list(args.items())[:_MAX_ARGS]
    body = ", ".join(f"{key}={_one_line(value, limit=_MAX_ARG_VALUE)}" for key, value in shown)
    if len(args) > _MAX_ARGS:
        body = f"{body}, …" if body else "…"
    return body


def summarize_call(name: str, args: Mapping[str, object]) -> str:
    """A compact ``name(key arg)`` view of a tool call for the transcript."""
    return f"{name}({describe_args(args)})"


def full_args(args: Mapping[str, object]) -> str:
    """The full ``key=value`` arg view shown when the approval prompt is expanded (``e``).

    Values are shown whole (newlines preserved) but each is capped at ``_EXPANDED_VALUE`` so a
    huge file can't make the modal unbounded; :func:`describe_args` is the collapsed view.
    """
    lines = []
    for key, value in args.items():
        text = str(value)
        if len(text) > _EXPANDED_VALUE:
            text = (
                f"{text[:_EXPANDED_VALUE].rstrip()} … (+{len(text) - _EXPANDED_VALUE} more chars)"
            )
        lines.append(f"{key}={text}")
    return "\n".join(lines)


def summarize_result(content: str) -> str:
    """A short preview of tool output: the first few lines, clipped, with a hidden-count tail.

    Returns at most ``_PREVIEW_LINES`` lines and ``_PREVIEW_CHARS`` characters; when the
    output was longer, appends ``… (+N more lines)`` (or ``… (+N more chars)`` when a single
    long line was clipped) so the elision is visible rather than silent.
    """
    text = content.strip()
    if not text:
        return ""
    lines = text.splitlines()
    preview_lines = lines[:_PREVIEW_LINES]
    preview = "\n".join(preview_lines)
    hidden_lines = len(lines) - len(preview_lines)
    if len(preview) > _PREVIEW_CHARS:
        kept = preview[:_PREVIEW_CHARS].rstrip()
        hidden_chars = len(preview) - len(kept)
        tail = f"+{hidden_lines} more lines" if hidden_lines else f"+{hidden_chars} more chars"
        return f"{kept} … ({tail})"
    if hidden_lines > 0:
        return f"{preview} … (+{hidden_lines} more lines)"
    return preview
