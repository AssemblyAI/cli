from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from rich import box
from rich.markup import escape
from rich.table import Table

from aai_cli import theme
from aai_cli.errors import UsageError

if TYPE_CHECKING:
    from aai_cli.errors import CLIError

console = theme.make_console()
# Errors go to stderr so they never pollute piped stdout (e.g. `aai transcribe x -o text > out`).
error_console = theme.make_console(stderr=True)

_AGENT_ENV_VARS = ("CI", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
_MIN_MASKABLE_SECRET_LENGTH = 8


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


def stream_output_modes(field: str | None, *, json_mode: bool) -> tuple[bool, bool]:
    """Fold a streaming command's ``-o/--output`` into ``(text_mode, json_mode)``.

    Shared by `stream` and `agent`, whose renderers take the same two flags: `text`
    emits plain finalized lines, `json` forces NDJSON, and an unset field falls back
    to the auto-detected `json_mode` (JSON when piped/agentic, human otherwise).
    """
    validate_output_field(field, ("text", "json"))
    text_mode = field == "text"
    return text_mode, (field == "json") or (json_mode and not text_mode)


def mask_secret(value: str) -> str:
    """Render a secret (API key, token) for display: first 3 + last 4 chars, else ``***``."""
    return f"{value[:3]}…{value[-4:]}" if len(value) >= _MIN_MASKABLE_SECRET_LENGTH else "***"


def success(text: str) -> str:
    """A success line — green ``✓`` + message — as a Rich-markup string.

    Helpers here return markup for a human renderer to print; they do NOT escape
    interpolated values, so callers escape any dynamic text (matching the inline
    ``escape(...)`` convention used throughout the command layer).
    """
    return f"[aai.success]{theme.SYMBOL_SUCCESS}[/aai.success] {text}"


def fail(text: str) -> str:
    """A failure line: red ``✗`` + message (for inline status, not the error path)."""
    return f"[aai.error]{theme.SYMBOL_ERROR}[/aai.error] {text}"


def warn(text: str) -> str:
    """A warning line: yellow ``!`` + message."""
    return f"[aai.warn]{theme.SYMBOL_WARN}[/aai.warn] {text}"


def hint(text: str) -> str:
    """A dim next-step hint, prefixed with the hint glyph to point at what's next."""
    return f"[aai.muted]{theme.SYMBOL_HINT} {text}[/aai.muted]"


def heading(text: str) -> str:
    """A section heading in the brand accent — the one voice for multi-line output."""
    return f"[aai.heading]{text}[/aai.heading]"


def data_table(*columns: str) -> Table:
    """A list table with the one consistent, minimal look used CLI-wide.

    Headers render in the brand heading style with a single rule beneath them and
    no surrounding box — the quiet, scannable style the Vercel/Supabase CLIs use.
    Defined once here so every listing command (`transcripts list`, `keys list`,
    `sessions list`, `usage`, `limits`, `audit`) shares the same table, rather than
    each re-deriving Rich's heavy default box.
    """
    return Table(*columns, box=box.SIMPLE_HEAD, header_style="aai.heading", pad_edge=False)


def detail_table() -> Table:
    """A borderless label/value grid for single-record views (`whoami`, `sessions get`).

    The label column is muted so the values read as the content and the pair scans
    as a definition list, not a boxed table. Centralizes what was two divergent
    one-off tables into one look.
    """
    table = Table.grid(padding=(0, 3))
    table.add_column(style="aai.muted")
    table.add_column()
    return table


def emit[T](data: T, human_renderer: Callable[[T], object], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(data, default=str))
    else:
        console.print(human_renderer(data))


def emit_ndjson(obj: object) -> None:
    """Write one newline-delimited JSON record to stdout, flushed for live pipelines."""
    print(json.dumps(obj, default=str), flush=True)


def emit_text(text: str) -> None:
    """Write one raw text value to stdout for pipe-friendly single-field output."""
    print(text)


def emit_error(err: CLIError, *, json_mode: bool) -> None:
    # Always to stderr, so stdout stays clean for `aai … | next-tool` pipelines.
    if json_mode:
        print(json.dumps(err.to_dict(), default=str), file=sys.stderr)
    else:
        error_console.print(f"[aai.error]Error:[/aai.error] {escape(err.message)}")
        if err.suggestion:
            error_console.print(f"[aai.muted]Suggestion:[/aai.muted] {escape(err.suggestion)}")


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
