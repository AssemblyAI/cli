from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from rich.markup import escape

from aai_cli import theme
from aai_cli.errors import UsageError

if TYPE_CHECKING:
    from aai_cli.errors import CLIError

T = TypeVar("T")

console = theme.make_console()
# Errors go to stderr so they never pollute piped stdout (e.g. `aai transcribe x -o text > out`).
error_console = theme.make_console(stderr=True)

_AGENT_ENV_VARS = ("CI", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


def _stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def _is_agentic() -> bool:
    if not _stdout_is_tty():
        return True
    return any(os.environ.get(var) for var in _AGENT_ENV_VARS)


def resolve_json(*, explicit: bool) -> bool:
    """JSON output when asked for, or when not attached to an interactive human."""
    return explicit or _is_agentic()


def validate_output_field(field: str | None, allowed: tuple[str, ...]) -> None:
    """Reject an unknown ``-o/--output`` value with a consistent, listing error."""
    if field is not None and field not in allowed:
        raise UsageError(f"Unknown --output {field!r}. Choose one of: {', '.join(allowed)}.")


def stream_output_modes(field: str | None, json_mode: bool) -> tuple[bool, bool]:
    """Fold a streaming command's ``-o/--output`` into ``(text_mode, json_mode)``.

    Shared by `stream` and `agent`, whose renderers take the same two flags: `text`
    emits plain finalized lines, `json` forces NDJSON, and an unset field falls back
    to the auto-detected `json_mode` (JSON when piped/agentic, human otherwise).
    """
    validate_output_field(field, ("text", "json"))
    text_mode = field == "text"
    return text_mode, (field == "json") or (json_mode and not text_mode)


def emit(data: T, human_renderer: Callable[[T], object], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(data, default=str))
    else:
        console.print(human_renderer(data))


def emit_ndjson(obj: object) -> None:
    """Write one newline-delimited JSON record to stdout, flushed for live pipelines."""
    print(json.dumps(obj, default=str), flush=True)


def emit_error(err: CLIError, *, json_mode: bool) -> None:
    # Always to stderr, so stdout stays clean for `aai … | next-tool` pipelines.
    if json_mode:
        print(json.dumps(err.to_dict(), default=str), file=sys.stderr)
    else:
        error_console.print(f"[aai.error]Error:[/aai.error] {escape(err.message)}")
        suggestion = getattr(err, "suggestion", None)
        if suggestion:
            error_console.print(f"[aai.muted]Suggestion:[/aai.muted] {escape(suggestion)}")


def print_code(code: str, *, language: str = "python") -> None:
    """Print generated source: syntax-highlighted for an interactive human, raw text
    otherwise. Piping/redirecting (or an agent) yields plain text with no ANSI, so
    `aai … --show-code > script.py` stays byte-clean and runnable.
    """
    if _is_agentic():
        print(code)
        return
    from rich.syntax import Syntax  # lazily import Pygments-backed highlighter

    console.print(Syntax(code, language, theme="ansi_dark", background_color="default"))
