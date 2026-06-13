from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import typer
from rich.console import RenderableType
from rich.style import StyleType
from rich.table import Table
from typer import completion, rich_utils
from typer._click.exceptions import ClickException, NoSuchOption
from typer._click.exceptions import UsageError as ClickUsageError
from typer._click.utils import PacifyFlushWrapper
from typer.core import TyperGroup

if TYPE_CHECKING:
    # Typer (>=0.13) vendors its own click; TyperGroup.list_commands receives this
    # context type, not the upstream click.Context. Imported for typing only.
    from typer._click.core import Context as ClickContext

from aai_cli import __version__, argscan, debuglog, environments, help_panels, output, stdio, theme
from aai_cli.commands import (
    account,
    agent,
    audit,
    caption,
    clip,
    deploy,
    dev,
    dictate,
    doctor,
    dub,
    evaluate,
    init,
    keys,
    llm,
    login,
    onboard,
    sessions,
    setup,
    share,
    speak,
    stream,
    telemetry,
    transcribe,
    transcripts,
    webhooks,
)
from aai_cli.context import AppState
from aai_cli.errors import CLIError, NotAuthenticated, UsageError
from aai_cli.help_text import examples_epilog
from aai_cli.onboard import wizard
from aai_cli.onboard.sections import WizardContext

# The order commands appear under `assembly --help`. Commands are grouped into named
# Rich panels (see `help_panels.py`); panels render in the order their first
# command appears here, so keep each panel's commands contiguous and ordered
# most-common-first. Names not listed fall to the end, sorted alphabetically.
_COMMAND_ORDER = (
    # Quick Start — zero-to-running onboarding
    "onboard",
    # Build an App — scaffold a new project
    "init",
    "dev",
    "share",
    "deploy",
    # Run AssemblyAI — use AssemblyAI directly from the terminal
    "transcribe",
    "stream",
    "dictate",
    "agent",
    "speak",
    "llm",
    "clip",
    "dub",
    "caption",
    "eval",
    "webhooks",
    # Setup & Tools — get set up & maintain
    "doctor",
    "setup",
    "telemetry",
    # History — browse past work
    "transcripts",
    "sessions",
    # Account — auth, then billing, then keys
    "login",
    "logout",
    "whoami",
    "balance",
    "usage",
    "limits",
    "keys",
    "audit",
)


class _OrderedGroup(TyperGroup):
    """Lists commands in `_COMMAND_ORDER` rather than registration order.

    Typer renders all direct commands before sub-typer groups, so registration
    order alone can't control the panel layout; sorting here drives help output.
    """

    def list_commands(self, ctx: ClickContext) -> list[str]:
        rank = {name: i for i, name in enumerate(_COMMAND_ORDER)}
        return sorted(
            super().list_commands(ctx), key=lambda name: (rank.get(name, len(rank)), name)
        )

    def parse_args(self, ctx: ClickContext, args: list[str]) -> list[str]:
        # Stash the full token list before anything is parsed, so the root callback can
        # tell whether the (not-yet-parsed) subcommand opted into JSON — see
        # `argscan.requests_json`. Recorded here because Click clears the pending
        # args off the context before the group callback runs.
        ctx.meta[_RAW_ARGS_META_KEY] = list(args)
        return super().parse_args(ctx, args)


# Typer's default help palette is a rainbow: option flags/command names in "bold cyan",
# the short switch (e.g. -p) in "bold green", and the type metavar (e.g. TEXT) in "bold
# yellow". Retint the whole panel into the Cobolt brand family so help reads as one
# monochrome hierarchy: flags and command names in the bold primary accent, their short
# aliases matching, and the type metavar in the lighter secondary Cobolt so it recedes.
# Set before the app renders any help.
rich_utils.STYLE_OPTION = f"bold {theme.BRAND}"
rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {theme.BRAND}"
rich_utils.STYLE_SWITCH = f"bold {theme.BRAND}"
rich_utils.STYLE_METAVAR = theme.ACCENT
# The usage line ("Usage: assembly [OPTIONS] COMMAND [ARGS]...") defaults to yellow. Keep the
# program name in the bold brand accent so it matches command names elsewhere, but drop
# the "Usage:" label and arg spec to muted warm gray — it's boilerplate that should recede.
rich_utils.STYLE_USAGE = theme.MUTED
rich_utils.STYLE_USAGE_COMMAND = f"bold {theme.BRAND}"


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


# Typer's own help/error consoles must also honor the closed-pipe contract: with
# Rich's default Console, `assembly --help | head -2` exits 1 via Console.on_broken_pipe.
_patch_module(rich_utils, Table=_NoClipTable, Console=theme.PipeSafeConsole)

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
    token list the root group stashed on the context (see ``_OrderedGroup.parse_args``).
    A ClickException raised without a context falls back to the process argv.
    """
    ctx = err.ctx if isinstance(err, ClickUsageError) else None
    if ctx is not None and _RAW_ARGS_META_KEY in ctx.meta:
        raw_args: list[str] = ctx.meta[_RAW_ARGS_META_KEY]
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


rich_utils.rich_format_error = _format_click_error_fixed

# Typer's built-in `--show-completion` help is long enough to wrap several lines in
# the options panel. Trim it so it fits on fewer rows. The OptionInfo objects live on
# the completion placeholder's parameter defaults; reach the (underscore-prefixed)
# placeholder through the module dict so it isn't flagged as private-attribute use.
_completion_placeholder = vars(completion)["_install_completion_placeholder_function"]
for _opt in _completion_placeholder.__defaults__ or ():
    if isinstance(_opt.help, str) and _opt.help.startswith("Show completion"):
        _opt.help = "Show completion for the current shell."


app = typer.Typer(
    name="assembly",
    # No top-level `help=`: the bare-`assembly` welcome banner plus the command table
    # below already introduce the tool, so a description here would be redundant.
    # `assembly --install-completion` / `--show-completion` for bash/zsh/fish/PowerShell,
    # the discoverability affordance gh/kubectl/docker users reach for.
    add_completion=True,
    cls=_OrderedGroup,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit when `assembly --version`/`-V` is passed, before any command
    runs. Mirrors the reflex (`tool --version`) every other CLI answers. There is
    deliberately no `version` subcommand; the unknown-command error points here instead."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _profile_has_key(state: AppState) -> bool:
    try:
        state.resolve_api_key()
    except NotAuthenticated:
        return False
    return True


# The root callback runs before the subcommand parses its own ``--json``, so a failure
# raised there (e.g. a bad ``--env``) would otherwise always render human text — leaving a
# ``… --json`` pipeline without the uniform ``{"error": …}`` shape it relies on. The group
# stashes the raw token list in ``ctx.meta`` (see ``_OrderedGroup.parse_args``) before the
# callback runs, so sniffing it with ``argscan.requests_json`` lets every failure class
# honor the request.
_RAW_ARGS_META_KEY = "aai_raw_args"

_LOG = logging.getLogger("aai_cli")


def _sandbox_conflict_warning(sandbox: bool, env: str | None) -> str | None:
    """A warning when ``--sandbox`` and a contradictory ``--env`` are both passed.

    Credentials are environment-bound, so the conflict must not be resolved silently:
    ``--env`` wins, and the warning names the loser so the user can drop a flag.
    """
    if sandbox and env is not None and env != environments.SANDBOX_ENV:
        return f"--sandbox ignored: --env {env} takes precedence."
    return None


def _offer_or_help(ctx: typer.Context, state: AppState) -> None:
    """No subcommand given: offer guided setup to a credential-less, interactive user;
    otherwise print help. Never prompts in a non-interactive session, and never on
    `--help` (Click handles that eagerly before the callback)."""
    if not state.quiet:
        output.print_banner()
    if stdio.interactive_stdio() and not _profile_has_key(state):
        if not state.quiet:
            output.console.print()  # blank line so the prompt isn't flush against the banner
        if typer.confirm("Welcome to AssemblyAI. Run guided setup now?", default=True):
            wiz_ctx = WizardContext(state=state, profile=state.resolve_profile(), json_mode=False)
            raise typer.Exit(code=wizard.run_onboarding(onboard.build_prompter(), wiz_ctx))
    typer.echo(ctx.get_help())
    raise typer.Exit()


@app.callback(
    invoke_without_command=True,
    epilog=examples_epilog(
        [
            ("Guided setup (start here)", "assembly onboard"),
            ("Transcribe a file", "assembly transcribe call.mp3"),
            ("Stream live audio in real time", "assembly stream"),
            ("Talk to a voice agent", "assembly agent"),
            (
                "Summarize while transcribing",
                'assembly transcribe call.mp3 --llm "summarize action items"',
            ),
        ]
    ),
)
def main(
    ctx: typer.Context,
    profile: str | None = typer.Option(None, "--profile", "-p", help="Named credential profile."),
    env: str | None = typer.Option(
        None, "--env", help=f"Backend environment ({', '.join(environments.ENVIRONMENTS)})."
    ),
    sandbox: bool = typer.Option(
        False, "--sandbox", help=f"Shortcut for --env {environments.SANDBOX_ENV}."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress non-essential messages (warnings, hints)."
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Log diagnostics to stderr (-v: requests, -vv: wire-level detail).",
    ),
    # Underscore name: the eager callback does the work, so the parameter is intentionally
    # unused in the body (avoids ARG001 without a `del`).
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the CLI version and exit.",
        callback=_version_callback,
        # Eager so --version short-circuits before subcommand/arg parsing — the idiomatic
        # default. The plain `assembly --version` path behaves identically with or without this,
        # so there's no cheap test that distinguishes it.
        is_eager=True,  # pragma: no mutate
    ),
) -> None:
    # The command's own --json flag isn't parsed yet, so sniff the pending command line:
    # a root-callback failure (e.g. bad --env) still emits the JSON error shape when the
    # invocation opted into JSON, and renders human text on stderr otherwise.
    # Enabled before anything else runs so even environment/profile resolution
    # failures can be diagnosed with -v.
    debuglog.enable(verbose)
    raw_args: list[str] = ctx.meta.get(_RAW_ARGS_META_KEY, [])
    json_mode = output.resolve_json(explicit=argscan.requests_json(raw_args))
    conflict_warning = _sandbox_conflict_warning(sandbox, env)
    if sandbox and env is None:
        env = environments.SANDBOX_ENV
    state = AppState(profile=profile, env=env, quiet=quiet)
    ctx.obj = state
    try:
        environments.set_active(state.resolve_environment())
    except CLIError as err:
        output.emit_error(err, json_mode=json_mode)
        raise typer.Exit(code=err.exit_code) from None
    active_env = environments.active()
    _LOG.debug("environment: %s (%s)", active_env.name, active_env.api_base)
    for warning in (conflict_warning, state.env_override_warning()):
        if warning and not quiet:
            # Surfaced in JSON mode too (as {"warning": …}), so a `--json` pipeline gets
            # a machine-readable hint instead of an unexplained downstream auth failure.
            output.emit_warning(warning, json_mode=json_mode)
    if ctx.invoked_subcommand is None:
        _offer_or_help(ctx, state)


# Help-panel grouping: named sub-typers carry their panel on `add_typer`; merged
# (nameless) sub-typers don't propagate it, so those commands set `rich_help_panel`
# on their own `@app.command()` (see each command module). Final ordering within a
# panel is controlled by `_COMMAND_ORDER` via `_OrderedGroup`, not registration order.
app.add_typer(transcribe.app)
app.add_typer(stream.app)
app.add_typer(dictate.app)
app.add_typer(transcripts.app, name="transcripts", rich_help_panel=help_panels.HISTORY)
app.add_typer(sessions.app, name="sessions", rich_help_panel=help_panels.HISTORY)
app.add_typer(audit.app)  # audit
app.add_typer(agent.app)
app.add_typer(speak.app)
app.add_typer(llm.app)
app.add_typer(clip.app)
app.add_typer(dub.app)
app.add_typer(caption.app)
app.add_typer(evaluate.app)  # eval
app.add_typer(account.app)  # balance, usage, limits
app.add_typer(login.app)  # login, logout, whoami
app.add_typer(doctor.app)
app.add_typer(init.app)
app.add_typer(dev.app)
app.add_typer(share.app)
app.add_typer(deploy.app)
app.add_typer(onboard.app)
app.add_typer(setup.app, name="setup", rich_help_panel=help_panels.SETUP)
app.add_typer(telemetry.app, name="telemetry", rich_help_panel=help_panels.SETUP)
app.add_typer(keys.app, name="keys", rich_help_panel=help_panels.ACCOUNT)
app.add_typer(webhooks.app, name="webhooks", rich_help_panel=help_panels.TRANSCRIPTION)


@app.command(
    name="_update-check",
    hidden=True,
    epilog=examples_epilog(
        [("Internal plumbing, spawned by the CLI itself", "assembly _update-check")]
    ),
)
def update_check_command() -> None:
    """Internal: refresh the cached latest version (spawned detached). Hidden."""
    from aai_cli import update_check

    update_check.fetch_and_cache()


def run() -> None:
    """Console-script entry point: run the app, exiting cleanly on a closed pipe.

    A downstream consumer (e.g. `assembly … | head`) can close the pipe before we finish
    writing. Without this, the write — or Python's flush at shutdown — raises
    BrokenPipeError and prints an ugly "Exception ignored" traceback. We treat a
    closed pipe as success: silence stdout and exit 0. Streaming commands also catch
    it earlier; this is the catch-all for the one-shot `output.emit`/`print` paths.
    """
    try:
        app(prog_name="assembly")
    except BrokenPipeError:
        stdio.silence_stdout()
        sys.exit(0)
    except SystemExit as exc:
        # Typer's vendored Click handles EPIPE before our handler can see it: it wraps
        # the streams in PacifyFlushWrapper and exits 1. That wrapper only ever appears
        # on the closed-pipe path, so rewrite that exit to the documented success code.
        if exc.code == 1 and isinstance(sys.stdout, PacifyFlushWrapper):
            stdio.silence_stdout()
            sys.exit(0)
        raise
