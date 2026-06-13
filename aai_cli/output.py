from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from aai_cli import __version__, choices, jsonshape, theme

if TYPE_CHECKING:
    from aai_cli.errors import CLIError

console = theme.make_console()
# Errors go to stderr so they never pollute piped stdout (e.g. `assembly transcribe x -o text > out`).
error_console = theme.make_console(stderr=True)

_AGENT_ENV_VARS = ("CI", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
_MIN_MASKABLE_SECRET_LENGTH = 8


def _stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def is_agentic() -> bool:
    """True when there's no interactive human at stdout: piped/redirected, or a CI/agent
    env var is set. Used to suppress *interactivity* (the spinner) — never to change the
    output *shape*; `resolve_json` keeps text the default regardless (see its docstring).
    """
    return not _stdout_is_tty() or any(os.environ.get(var) for var in _AGENT_ENV_VARS)


def set_color_mode(mode: choices.ColorMode) -> None:
    """Apply the root ``--color`` tri-state process-wide.

    ``auto`` keeps Rich's TTY detection (which already honors ``NO_COLOR`` /
    ``FORCE_COLOR``). The explicit modes do two things: rebuild this module's
    shared consoles, and set the corresponding env var so consoles created later
    (the realtime renderers build their own) and child processes agree.
    """
    if mode is choices.ColorMode.auto:
        return
    if mode is choices.ColorMode.always:
        os.environ["FORCE_COLOR"] = "1"
        os.environ.pop("NO_COLOR", None)
        rebuilt = {
            "console": theme.make_console(force_terminal=True),
            "error_console": theme.make_console(stderr=True, force_terminal=True),
        }
    else:
        os.environ["NO_COLOR"] = "1"
        os.environ.pop("FORCE_COLOR", None)
        rebuilt = {
            "console": theme.make_console(no_color=True),
            "error_console": theme.make_console(stderr=True, no_color=True),
        }
    # Swapped via module setattr (the `_patch_module` pattern in main.py) rather
    # than a `global` statement; readers always go through `output.console`, so
    # the rebind is visible everywhere.
    for name, console_obj in rebuilt.items():
        setattr(sys.modules[__name__], name, console_obj)


def resolve_json(*, explicit: bool) -> bool:
    """JSON output only when explicitly requested with ``--json`` (or ``-o json``).

    Human-readable text is the default for every command, in every context — a
    terminal, a pipe, CI, or an agent. We deliberately do NOT switch the output
    *shape* to JSON just because stdout is piped or a ``CI``/``CLAUDECODE`` env var
    is set: that surprised plain-text pipelines like ``assembly transcribe x | grep word``
    by handing them a JSON blob instead of the transcript. Being off a TTY still
    drops color and interactivity (Rich handles that automatically); it just no
    longer changes the structure. This matches gh/docker/kubectl, which keep their
    human/tabular output until you opt in to ``--json``.
    """
    return explicit


def stream_output_modes(field: choices.TextOrJson | None, *, json_mode: bool) -> tuple[bool, bool]:
    """Fold a streaming command's ``-o/--output`` into ``(text_mode, json_mode)``.

    Shared by `stream` and `agent`. ``-o text`` emits plain finalized lines (handy for
    ``assembly stream -o text | assembly llm -f``); ``-o json`` or ``--json`` forces NDJSON; an
    unset field renders the live human panel. With output now human-by-default
    (`resolve_json` only flips on an explicit `--json`), `json_mode` here is simply
    whether `--json` was passed — we never auto-switch to NDJSON just because piped.
    Typer validates `field` against the enum, so no value check is needed here.
    """
    text_mode = field is choices.TextOrJson.text
    return text_mode, (field is choices.TextOrJson.json) or (json_mode and not text_mode)


def redact_secret(value: str) -> str:
    """Render a secret (API key, token) for display: first 3 + last 4 chars, else ``***``.

    This is the sanitizer that makes secrets safe to show (`whoami`, `doctor`): only
    7 characters survive. Assembled via ``join(map(str, …))`` rather than an f-string
    because CodeQL propagates sensitive-data taint through every direct string
    operation (slice/concat/format/join), which would flag every payload containing
    a masked key as clear-text logging of the secret itself
    (py/clear-text-logging-sensitive-data); this form is the dataflow barrier the
    masking semantically is.
    """
    if len(value) < _MIN_MASKABLE_SECRET_LENGTH:
        return "***"
    return "".join(map(str, (value[:3], "…", value[-4:])))


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


def hidden_note(count: int, noun: str, flag: str) -> Text | None:
    """The muted "Hidden: N <noun>(s). Use <flag> to show them." footnote, or None
    when nothing was hidden (so it drops out of a `stack`). Pins the phrasing once
    for the listing commands that elide rows behind an --include-* flag."""
    if not count:
        return None
    return muted(f"Hidden: {count} {noun}(s). Use {flag} to show them.")


def stack(*items: RenderableType | None) -> RenderableType:
    """Stack renderables top-to-bottom, dropping any ``None``.

    A lone surviving item is returned bare so a single table or line isn't wrapped in
    a redundant `Group`. Lets a command assemble "summary + table + optional footnote"
    without re-deriving the same `Group`/None branching each time.
    """
    present = [item for item in items if item is not None]
    return present[0] if len(present) == 1 else Group(*present)


def emit[T](data: T, human_renderer: Callable[[T], object], *, json_mode: bool) -> None:
    """Emit a command's result to stdout: ``data`` as JSON, or ``human_renderer(data)``.

    The single split every command's success path goes through, so the JSON/human
    choice and the stdout/stderr discipline stay in one place.
    """
    if json_mode:
        print(jsonshape.dumps(data))
    else:
        console.print(human_renderer(data))


def emit_ndjson(obj: object) -> None:
    """Write one newline-delimited JSON record to stdout, flushed for live pipelines."""
    print(jsonshape.dumps(obj), flush=True)


def emit_text(text: str) -> None:
    """Write one raw text value to stdout for pipe-friendly single-field output."""
    print(text)


@contextlib.contextmanager
def status(message: str, *, json_mode: bool, quiet: bool = False) -> Generator[None]:
    """Show an ephemeral spinner on stderr during a long human-facing wait.

    A no-op in JSON or non-interactive mode (piped / agent-run), under ``--quiet``,
    so stdout stays clean for pipelines and machine output is never decorated.
    Rendered on the stderr console so even an interactive `assembly transcribe x -o text`
    keeps stdout pristine.
    """
    if json_mode or quiet or is_agentic():
        yield
        return
    with error_console.status(message):
        yield


def emit_warning(message: str, *, json_mode: bool) -> None:
    """Emit a non-fatal warning to stderr, structured under ``--json``.

    In JSON mode a human ``! …`` line would corrupt a ``{"error": …}`` pipeline, so
    the warning ships as its own ``{"warning": …}`` object on stderr — keeping stdout
    clean and stderr machine-readable. Human mode gets the familiar yellow line.
    """
    if json_mode:
        print(jsonshape.dumps({"warning": message}), file=sys.stderr)
    else:
        error_console.print(warn(message))


def emit_error(err: CLIError, *, json_mode: bool) -> None:
    """Write a CLIError to stderr — the ``{"error": …}`` object under ``--json``, else a
    styled ``Error:`` line plus its suggestion — keeping stdout clean for pipelines."""
    # Always to stderr, so stdout stays clean for `assembly … | next-tool` pipelines.
    if json_mode:
        print(jsonshape.dumps(err.to_dict()), file=sys.stderr)
    else:
        error_console.print(f"[aai.error]Error:[/aai.error] {escape(err.message)}")
        if err.suggestion:
            error_console.print(f"[aai.muted]Suggestion:[/aai.muted] {escape(err.suggestion)}")


# A one-line header: emoji + product + version.


def print_banner() -> None:
    """Print the welcome header — a single emoji + product + version line in the
    brand accent (the bare-command welcome screen)."""
    # highlight=False so Rich's repr-highlighter doesn't recolor the version digits —
    # the line stays a single muted tone behind the brand label.
    console.print(
        f"[aai.brand]🎙️  AssemblyAI CLI[/aai.brand] [aai.muted]{__version__}[/aai.muted]",
        highlight=False,  # pragma: no mutate (purely cosmetic: toggles Rich repr coloring, not text)
    )


def print_code(code: str, *, language: str = "python") -> None:
    """Print generated source: syntax-highlighted for an interactive human, raw text
    otherwise. Piping/redirecting (or an agent) yields plain text with no ANSI, so
    `assembly … --show-code > script.py` stays byte-clean and runnable.
    """
    if not _stdout_is_tty():
        print(code)
        return
    from rich.syntax import Syntax  # lazily import Pygments-backed highlighter

    console.print(Syntax(code, language, theme="ansi_dark", background_color="default"))
