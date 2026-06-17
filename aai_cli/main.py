from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING

import typer
from typer._click.utils import PacifyFlushWrapper
from typer.core import TyperGroup, TyperOption

if TYPE_CHECKING:
    # Typer (>=0.13) vendors its own click; TyperGroup.list_commands receives this
    # context type, not the upstream click.Context. Imported for typing only.
    from typer._click.core import Command as ClickCommand
    from typer._click.core import Context as ClickContext
    from typer._click.formatting import HelpFormatter as ClickHelpFormatter

from aai_cli import __version__, command_registry
from aai_cli.app.context import AppState
from aai_cli.commands import onboard
from aai_cli.core import access, argscan, choices, debuglog, environments, stdio
from aai_cli.core.environments import Environment
from aai_cli.core.errors import CLIError, NotAuthenticated
from aai_cli.onboard import wizard
from aai_cli.onboard.sections import WizardContext
from aai_cli.ui import output, typer_patches
from aai_cli.ui.help_text import examples_epilog

# Every module under aai_cli/commands/ declares its own panel, rank, and command
# names (`SPEC`, see aai_cli/command_registry.py); discovery imports and orders them
# all, so registering a new command edits no shared list in this file.
_REGISTERED_COMMAND_MODULES = command_registry.discover()

# The order commands appear under `assembly --help`, derived from each module's SPEC:
# panels render in help_panels.PANEL_ORDER and stay contiguous by construction.
# Names not listed (the hidden _update-check) fall to the end, sorted alphabetically.
_COMMAND_ORDER = command_registry.command_order(_REGISTERED_COMMAND_MODULES)

# Rank lookup derived once from the static order; list_commands runs per `--help`
# and per completion, so there's no reason to rebuild this on every call.
_COMMAND_RANK = {name: i for i, name in enumerate(_COMMAND_ORDER)}


# Root flags and the marker the sandbox-only command docstrings open with: the single
# place the "what is a sandbox option" surface is defined, so help-filtering and the
# command docstrings stay the lone declarations (no parallel command list to maintain).
_SANDBOX_ROOT_FLAGS = frozenset({"sandbox", "env"})
_SANDBOX_HELP_MARKER = "[sandbox]"


def _is_sandbox_command(command: ClickCommand) -> bool:
    """Whether a command is sandbox-only, detected by the ``[sandbox]`` help prefix.

    The docstrings escape the bracket for Rich (``\\[sandbox]``), so strip a leading
    backslash before matching.
    """
    text = (command.help or command.short_help or "").lstrip()
    return text.lstrip("\\").startswith(_SANDBOX_HELP_MARKER)


class _OrderedGroup(TyperGroup):
    """Lists commands in `_COMMAND_ORDER` rather than registration order.

    Typer renders all direct commands before sub-typer groups, so registration
    order alone can't control the panel layout; sorting here drives help output.
    """

    def list_commands(self, ctx: ClickContext) -> list[str]:
        return sorted(
            super().list_commands(ctx),
            key=lambda name: (_COMMAND_RANK.get(name, len(_COMMAND_RANK)), name),
        )

    def parse_args(self, ctx: ClickContext, args: list[str]) -> list[str]:
        # Stash the full token list before anything is parsed, so the root callback can
        # tell whether the (not-yet-parsed) subcommand opted into JSON — see
        # `argscan.requests_json`. Recorded here because Click clears the pending
        # args off the context before the group callback runs.
        ctx.meta[argscan.RAW_ARGS_META_KEY] = list(args)
        return super().parse_args(ctx, args)

    def _sandbox_surface(self, ctx: ClickContext) -> list[TyperOption | ClickCommand]:
        """The sandbox root flags and ``[sandbox]`` commands — the surface to hide."""
        flags: list[TyperOption | ClickCommand] = [
            param
            for param in self.get_params(ctx)
            if isinstance(param, TyperOption) and param.name in _SANDBOX_ROOT_FLAGS
        ]
        commands = [
            command
            for name in self.list_commands(ctx)
            if (command := self.get_command(ctx, name)) is not None and _is_sandbox_command(command)
        ]
        return [*flags, *commands]

    @contextlib.contextmanager
    def _sandbox_surface_hidden(self, ctx: ClickContext) -> Generator[None]:
        """Mark the sandbox flags/commands ``hidden`` for one render, then restore.

        Restored in ``finally``: the parameter/command objects are process-global
        (one Typer tree per process), so a leaked ``hidden=True`` would wrongly hide
        the sandbox surface from a later in-process render or from shell completion.
        """
        targets = self._sandbox_surface(ctx)
        saved = [(target, target.hidden) for target in targets]
        for target in targets:
            target.hidden = True
        try:
            yield
        finally:
            for target, was_hidden in saved:
                target.hidden = was_hidden

    def format_help(self, ctx: ClickContext, formatter: ClickHelpFormatter) -> None:
        """Render `assembly --help`, hiding the sandbox surface from external accounts.

        The sandbox runs on internal infrastructure, so its flags and commands are
        noise (and a dead end) for an external account — show them only to an
        AssemblyAI login. Internal users get the full surface unchanged.
        """
        if access.profile_is_internal():
            super().format_help(ctx, formatter)
            return
        with self._sandbox_surface_hidden(ctx):
            super().format_help(ctx, formatter)


# Brand-retint Typer's help palette, pin help-table columns against clipping, make
# Typer's consoles pipe-safe, fix Click's error formatting, and trim the completion
# help — every Typer/Click/Rich override lives in typer_patches so a dependency
# upgrade that breaks one is fixed in one file. Must run before any help renders.
typer_patches.apply()

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


_LOG = logging.getLogger("aai_cli")


def _sandbox_conflict_warning(sandbox: bool, env: str | None) -> str | None:
    """A warning when ``--sandbox`` and a contradictory ``--env`` are both passed.

    Credentials are environment-bound, so the conflict must not be resolved silently:
    ``--env`` wins, and the warning names the loser so the user can drop a flag.
    """
    if sandbox and env is not None and env != environments.SANDBOX_ENV:
        return f"--sandbox ignored: --env {env} takes precedence."
    return None


def _enforce_internal_env(
    ctx: typer.Context, state: AppState, active_env: Environment, *, json_mode: bool
) -> None:
    """Reject an internal-only environment for a profile that isn't an AssemblyAI account.

    The sandbox runs on internal infrastructure an external account can neither reach
    nor authenticate against, so selecting it (via --sandbox / --env / AAI_ENV) fails
    here with a clean error instead of a confusing downstream auth failure. ``login``
    is exempt: a first-time employee must be able to target the sandbox to sign in
    there, which is what records the email this gate then reads.
    """
    if active_env.name == environments.DEFAULT_ENV:
        return
    if ctx.invoked_subcommand == "login":
        return
    if access.profile_is_internal(state.resolve_profile()):
        return
    err = CLIError(
        f"The {active_env.name} environment is restricted to AssemblyAI accounts.",
        error_type="restricted_environment",
        exit_code=2,
        suggestion=(
            "Drop --sandbox/--env (and unset AAI_ENV) to use production, or run "
            "'assembly login' with an AssemblyAI account."
        ),
    )
    output.emit_error(err, json_mode=json_mode)
    raise typer.Exit(code=err.exit_code)


def _offer_or_help(ctx: typer.Context, state: AppState) -> None:
    """No subcommand given: offer guided setup to a credential-less, interactive user;
    otherwise print help. Never prompts in a non-interactive session, never on
    `--help` (Click handles that eagerly before the callback), and never when the
    stored config is unparseable — a deferred ``invalid_config`` error means
    ``resolve_api_key``/``resolve_profile`` would re-raise (escaping the callback as
    a traceback), and the wizard would only write atop a broken file."""
    if not state.quiet:
        output.print_banner()
    if (
        state.deferred_config_error is None
        and stdio.interactive_stdio()
        and not _profile_has_key(state)
    ):
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
    profile: str | None = typer.Option(None, "--profile", "-p", help="Named credential profile"),
    env: str | None = typer.Option(
        None, "--env", help=f"Backend environment ({', '.join(environments.ENVIRONMENTS)})"
    ),
    sandbox: bool = typer.Option(
        False, "--sandbox", help=f"Shortcut for --env {environments.SANDBOX_ENV}"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress non-essential messages (warnings, hints)"
    ),
    color: choices.ColorMode = typer.Option(
        choices.ColorMode.auto,
        "--color",
        help="Color output: auto (TTY detection), always, or never. NO_COLOR is also honored.",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Log diagnostics to stderr (-v: requests, -vv: wire-level detail)",
    ),
    # Underscore name: the eager callback does the work, so the parameter is intentionally
    # unused in the body (avoids ARG001 without a `del`).
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the CLI version and exit",
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
    output.set_color_mode(color)
    raw_args: list[str] = ctx.meta.get(argscan.RAW_ARGS_META_KEY, [])
    json_mode = output.resolve_json(explicit=argscan.requests_json(raw_args))
    conflict_warning = _sandbox_conflict_warning(sandbox, env)
    if sandbox and env is None:
        env = environments.SANDBOX_ENV
    state = AppState(profile=profile, env=env, quiet=quiet)
    ctx.obj = state
    try:
        environments.set_active(state.resolve_environment())
    except CLIError as err:
        if err.error_type != "invalid_config":
            output.emit_error(err, json_mode=json_mode)
            raise typer.Exit(code=err.exit_code) from None
        # A corrupt config.toml can't tell us the profile's stored env. Defer the error
        # (run_command re-raises it for real commands) and fall back to the explicit
        # --env or the default, so `assembly config path` can still locate the file.
        # An explicit bad --env still wins — surface it now rather than the deferred one.
        state.deferred_config_error = err
        try:
            environments.set_active(environments.resolve(state.env, None))
        except CLIError as env_err:
            output.emit_error(env_err, json_mode=json_mode)
            raise typer.Exit(code=env_err.exit_code) from None
    active_env = environments.active()
    _LOG.debug("environment: %s (%s)", active_env.name, active_env.api_base)
    _enforce_internal_env(ctx, state, active_env, json_mode=json_mode)
    for warning in (conflict_warning, state.env_override_warning()):
        if warning and not quiet:
            # Surfaced in JSON mode too (as {"warning": …}), so a `--json` pipeline gets
            # a machine-readable hint instead of an unexplained downstream auth failure.
            output.emit_warning(warning, json_mode=json_mode)
    if ctx.invoked_subcommand is None:
        _offer_or_help(ctx, state)


# Help-panel grouping: named sub-typers (SPEC.group_name set) carry their panel on
# `add_typer`; merged (nameless) sub-typers don't propagate it, so those commands set
# `rich_help_panel` on their own `@app.command()` (see each command module). Final
# ordering within a panel comes from each module's SPEC via `_OrderedGroup`, not
# registration order.
for _registered in _REGISTERED_COMMAND_MODULES:
    if _registered.spec.group_name is None:
        app.add_typer(_registered.app)
    else:
        app.add_typer(
            _registered.app,
            name=_registered.spec.group_name,
            rich_help_panel=_registered.spec.panel,
        )


@app.command(
    name="_update-check",
    hidden=True,
    epilog=examples_epilog(
        [("Internal plumbing, spawned by the CLI itself", "assembly _update-check")]
    ),
)
def update_check_command() -> None:
    """Internal: refresh the cached latest version (spawned detached). Hidden."""
    from aai_cli.ui import update_check

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
