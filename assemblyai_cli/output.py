from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from rich.markup import escape

from assemblyai_cli import theme

if TYPE_CHECKING:
    from assemblyai_cli.errors import CLIError

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


def emit(data: T, human_renderer: Callable[[T], object], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(data, default=str))
    else:
        console.print(human_renderer(data))


def emit_error(err: CLIError, *, json_mode: bool) -> None:
    # Always to stderr, so stdout stays clean for `aai … | next-tool` pipelines.
    if json_mode:
        print(json.dumps(err.to_dict(), default=str), file=sys.stderr)
    else:
        error_console.print(f"[aai.error]Error:[/aai.error] {escape(err.message)}")


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
