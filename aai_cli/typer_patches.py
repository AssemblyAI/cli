"""Every patch the CLI applies to Typer's vendored Click and Rich rendering.

Typer's defaults break four contracts this CLI keeps: a rainbow help palette
(retinted to the brand family), flag-name columns clipped to "--end-of-turn-c…"
on narrow terminals (pinned via ``_NoClipTable``), `assembly --help | head` exiting 1
on the closed pipe (``theme.PipeSafeConsole``), and unknown-flag errors that
leak a tuple repr or suggest the wrong placement (the error-formatter patch).

Isolated here — not inline in ``main.py`` — so a Typer/Click/Rich upgrade that
breaks a patch is fixed in one file. Written against Typer >= 0.13 (the
vendored-click era); each patch notes the upstream behavior it overrides.
``main.py`` calls :func:`apply` once at import time, before any help renders.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.style import StyleType
from rich.table import Table
from typer import completion, rich_utils
from typer._click.exceptions import ClickException, NoSuchOption
from typer._click.exceptions import UsageError as ClickUsageError

from aai_cli import argscan, output, theme
from aai_cli.errors import UsageError

if TYPE_CHECKING:
    # Typer (>=0.13) vendors its own click; these patches receive its context
    # type, not the upstream click.Context. Imported for typing only.
    from typer._click.core import Context as ClickContext


# Help tables put flag/command names in the leading columns and wrapping prose
# (metavar, help text) in the trailing two. Rich's width collapse only spares no_wrap
# columns, so on a narrow terminal it happily clips a flag name to "--end-of-turn-c…" —
# unlearnable from the help screen itself. Pin every column except the last two so the
# prose columns absorb the squeeze instead.
class _NoClipTable(Table):
    def add_row(
        self,
        *renderables: RenderableType | None,
        style: StyleType | None = None,
        end_section: bool = False,
    ) -> None:
        super().add_row(*renderables, style=style, end_section=end_section)
        for column in self.columns[:-2]:
            column.no_wrap = True


def _patch_module(module: ModuleType, **attrs: object) -> None:
    """Replace module attributes that are imports (not definitions) in their module —
    strict mypy's no-implicit-reexport rejects plain attribute assignment for those."""
    for name, value in attrs.items():
        setattr(module, name, value)


# The original Click error renderer, captured at import time — before apply() swaps
# it — so the patched formatter can delegate the human-text path to the real one.
_format_click_error = rich_utils.rich_format_error

# Flags users habitually pass at the wrong level: `--json` belongs on the subcommand
# (`assembly transcribe --json`), while the root callback's flags belong before it
# (`assembly --sandbox transcribe`). A bare "No such option" — or worse, a similarity
# guess like "(Possible options: --version)" — is unlearnable, so the Click error
# formatter appends the correct placement instead.


def _root_only_flags(ctx: ClickContext) -> frozenset[str]:
    """Every flag the root callback declares (--quiet, --sandbox, --env, …), read off
    the declarations themselves so a new global flag gets the placement hint without
    a hand-maintained parallel list."""
    return frozenset(opt for param in ctx.find_root().command.params for opt in param.opts)


def _misplaced_flag_hint(err: NoSuchOption) -> str | None:
    """A placement hint when a known flag landed at the wrong level, else None."""
    ctx = err.ctx
    if ctx is None:
        return None
    if ctx.parent is None:
        if err.option_name in argscan.JSON_FLAGS:
            return "Pass --json after the subcommand: assembly <command> --json"
        return None
    if err.option_name in _root_only_flags(ctx):
        command = ctx.command_path.removeprefix("assembly ")
        return (
            "This is a global flag; pass it before the subcommand: "
            f"assembly {err.option_name} {command} …"
        )
    return None


def _rewrite_version_command_error(err: ClickException) -> None:
    # There is no `version` subcommand (the reflex is `assembly --version`), and the
    # closest-match engine would suggest an unrelated command ("Did you mean
    # 'sessions'?"). Point at the real spelling instead.
    if err.message.startswith("No such command 'version'"):
        err.message = "No such command 'version'. Did you mean 'assembly --version'?"


def _click_error_requests_json(err: ClickException) -> bool:
    """Whether the invocation that failed to parse had opted into JSON output.

    A parse error fires before any command's own ``--json`` is read, so sniff the raw
    token list the root group stashed on the context (see ``_OrderedGroup.parse_args``
    in main.py). A ClickException raised without a context falls back to the process
    argv.
    """
    ctx = err.ctx if isinstance(err, ClickUsageError) else None
    if ctx is not None and argscan.RAW_ARGS_META_KEY in ctx.meta:
        raw_args: list[str] = ctx.meta[argscan.RAW_ARGS_META_KEY]
    else:
        raw_args = sys.argv[1:]
    return argscan.requests_json(raw_args)


def _format_click_error_fixed(self: ClickException) -> None:
    # Typer's vendored Click renders flag suggestions as a stringified 1-tuple:
    # "No such option: --jsno ('(Possible options: --json)',)". Fold the suggestion
    # into the message ourselves so the user sees "(Possible options: --json)" — or,
    # for a known flag passed at the wrong level, the placement hint instead of a
    # misleading similarity guess.
    if isinstance(self, NoSuchOption):
        hint = _misplaced_flag_hint(self)
        if hint is not None:
            self.message = f"{self.message}. {hint}"
        elif self.possibilities:
            self.message = (
                f"{self.message} (Possible options: {', '.join(sorted(self.possibilities))})"
            )
        self.possibilities = None
    _rewrite_version_command_error(self)
    if _click_error_requests_json(self):
        # An invocation that opted into JSON gets the uniform {"error": …} envelope for
        # parse errors too, mirroring the root-callback failure path; the exit code (2)
        # is Click's and unchanged. NoArgsIsHelpError never reaches this branch: its
        # message is the help screen and a bare invocation carries no JSON flag.
        output.emit_error(UsageError(self.format_message()), json_mode=True)
        return
    _format_click_error(self)


def _trim_completion_help() -> None:
    # Typer's built-in `--show-completion` help is long enough to wrap several lines in
    # the options panel. Trim it so it fits on fewer rows. The OptionInfo objects live on
    # the completion placeholder's parameter defaults; reach the (underscore-prefixed)
    # placeholder through the module dict so it isn't flagged as private-attribute use.
    completion_placeholder = vars(completion)["_install_completion_placeholder_function"]
    for opt in completion_placeholder.__defaults__ or ():
        if isinstance(opt.help, str) and opt.help.startswith("Show completion"):
            opt.help = "Show completion for the current shell."


def apply() -> None:
    """Apply every patch. Idempotent; must run before the app renders any help."""
    # Typer's default help palette is a rainbow: option flags/command names in "bold
    # cyan", the short switch (e.g. -p) in "bold green", and the type metavar (e.g.
    # TEXT) in "bold yellow". Retint the whole panel into the Cobolt brand family so
    # help reads as one monochrome hierarchy: flags and command names in the bold
    # primary accent, their short aliases matching, and the type metavar in the
    # lighter secondary Cobolt so it recedes.
    rich_utils.STYLE_OPTION = f"bold {theme.BRAND}"
    rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {theme.BRAND}"
    rich_utils.STYLE_SWITCH = f"bold {theme.BRAND}"
    rich_utils.STYLE_METAVAR = theme.ACCENT
    # The usage line ("Usage: assembly [OPTIONS] COMMAND [ARGS]...") defaults to yellow.
    # Keep the program name in the bold brand accent so it matches command names
    # elsewhere, but drop the "Usage:" label and arg spec to muted warm gray — it's
    # boilerplate that should recede.
    rich_utils.STYLE_USAGE = theme.MUTED
    rich_utils.STYLE_USAGE_COMMAND = f"bold {theme.BRAND}"
    # Typer's own help/error consoles must also honor the closed-pipe contract: with
    # Rich's default Console, `assembly --help | head -2` exits 1 via Console.on_broken_pipe.
    _patch_module(rich_utils, Table=_NoClipTable, Console=theme.PipeSafeConsole)
    rich_utils.rich_format_error = _format_click_error_fixed
    _trim_completion_help()
