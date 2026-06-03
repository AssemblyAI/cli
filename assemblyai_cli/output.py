from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:
    from assemblyai_cli.errors import CLIError

T = TypeVar("T")

console = Console()

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
    if json_mode:
        print(json.dumps(err.to_dict(), default=str))
    else:
        console.print(f"[red]Error:[/red] {escape(err.message)}")
