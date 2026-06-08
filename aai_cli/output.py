from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from aai_cli import choices, theme

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
    """True when there's no interactive human at stdout: piped/redirected, or a CI/agent
    env var is set. Used to suppress *interactivity* (the spinner) — never to change the
    output *shape*; `resolve_json` keeps text the default regardless (see its docstring).
    """
    if not _stdout_is_tty():
        return True
    return any(os.environ.get(var) for var in _AGENT_ENV_VARS)


def resolve_json(*, explicit: bool) -> bool:
    """JSON output only when explicitly requested with ``--json`` (or ``-o json``).

    Human-readable text is the default for every command, in every context — a
    terminal, a pipe, CI, or an agent. We deliberately do NOT switch the output
    *shape* to JSON just because stdout is piped or a ``CI``/``CLAUDECODE`` env var
    is set: that surprised plain-text pipelines like ``aai transcribe x | grep word``
    by handing them a JSON blob instead of the transcript. Being off a TTY still
    drops color and interactivity (Rich handles that automatically); it just no
    longer changes the structure. This matches gh/docker/kubectl, which keep their
    human/tabular output until you opt in to ``--json``.
    """
    return explicit


def stream_output_modes(field: choices.TextOrJson | None, *, json_mode: bool) -> tuple[bool, bool]:
    """Fold a streaming command's ``-o/--output`` into ``(text_mode, json_mode)``.

    Shared by `stream` and `agent`. ``-o text`` emits plain finalized lines (handy for
    ``aai stream -o text | aai llm -f``); ``-o json`` or ``--json`` forces NDJSON; an
    unset field renders the live human panel. With output now human-by-default
    (`resolve_json` only flips on an explicit `--json`), `json_mode` here is simply
    whether `--json` was passed — we never auto-switch to NDJSON just because piped.
    Typer validates `field` against the enum, so no value check is needed here.
    """
    text_mode = field is choices.TextOrJson.text
    return text_mode, (field is choices.TextOrJson.json) or (json_mode and not text_mode)


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


def muted(text: str) -> Text:
    """A dim secondary line — empty-state messages and "Hidden: …" footnotes.

    Returns a Rich ``Text`` (not markup) so it composes into a `stack` alongside a
    table, the way listing commands (`usage`, `audit`) tack a quiet note onto a view.
    """
    return Text(text, style="aai.muted")


def stack(*items: RenderableType | None) -> RenderableType:
    """Stack renderables top-to-bottom, dropping any ``None``.

    A lone surviving item is returned bare so a single table or line isn't wrapped in
    a redundant `Group`. Lets a command assemble "summary + table + optional footnote"
    without re-deriving the same `Group`/None branching each time.
    """
    present = [item for item in items if item is not None]
    return present[0] if len(present) == 1 else Group(*present)


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


@contextlib.contextmanager
def status(message: str, *, json_mode: bool) -> Generator[None]:
    """Show an ephemeral spinner on stderr during a long human-facing wait.

    A no-op in JSON or non-interactive mode (piped / agent-run), so stdout stays
    clean for pipelines and machine output is never decorated. Rendered on the
    stderr console so even an interactive `aai transcribe x -o text` keeps stdout
    pristine.
    """
    if json_mode or _is_agentic():
        yield
        return
    with error_console.status(message):
        yield


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
    if not _stdout_is_tty():
        print(code)
        return
    from rich.syntax import Syntax  # lazily import Pygments-backed highlighter

    console.print(Syntax(code, language, theme="ansi_dark", background_color="default"))
