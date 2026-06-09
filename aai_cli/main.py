from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import typer
from typer import completion, rich_utils
from typer.core import TyperGroup

if TYPE_CHECKING:
    # Typer (>=0.13) vendors its own click; TyperGroup.list_commands receives this
    # context type, not the upstream click.Context. Imported for typing only.
    from typer._click.core import Context as ClickContext

from aai_cli import __version__, config, environments, help_panels, output, stdio, theme
from aai_cli.commands import (
    account,
    agent,
    audit,
    doctor,
    init,
    keys,
    llm,
    login,
    onboard,
    sessions,
    setup,
    stream,
    transcribe,
    transcripts,
)
from aai_cli.context import AppState, env_override_warning, resolve_environment
from aai_cli.errors import CLIError, NotAuthenticated
from aai_cli.help_text import examples_epilog
from aai_cli.onboard import wizard
from aai_cli.onboard.sections import WizardContext

# The order commands appear under `aai --help`. Commands are grouped into named
# Rich panels (see `help_panels.py`); panels render in the order their first
# command appears here, so keep each panel's commands contiguous and ordered
# most-common-first. Names not listed fall to the end, sorted alphabetically.
_COMMAND_ORDER = (
    # Quick Start — zero-to-running onboarding
    "onboard",
    # Build an App — scaffold a new project
    "init",
    # Run AssemblyAI — use AssemblyAI directly from the terminal
    "transcribe",
    "stream",
    "agent",
    "llm",
    # Setup & Tools — get set up & maintain
    "doctor",
    "setup",
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
        # `_command_line_requests_json`. Recorded here because Click clears the pending
        # args off the context before the group callback runs.
        ctx.meta[_RAW_ARGS_META_KEY] = list(args)
        return super().parse_args(ctx, args)


# Typer renders option flags and command names in "bold cyan" by default; retint
# both to the brand accent (the logo blue) so the help screen matches the rest of
# the CLI. Set before the app renders any help.
rich_utils.STYLE_OPTION = f"bold {theme.BRAND}"
rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {theme.BRAND}"

# Typer's built-in `--show-completion` help is long enough to wrap several lines in
# the options panel. Trim it so it fits on fewer rows. The OptionInfo objects live on
# the completion placeholder's parameter defaults; reach the (underscore-prefixed)
# placeholder through the module dict so it isn't flagged as private-attribute use.
_completion_placeholder = vars(completion)["_install_completion_placeholder_function"]
for _opt in _completion_placeholder.__defaults__ or ():
    if isinstance(_opt.help, str) and _opt.help.startswith("Show completion"):
        _opt.help = "Show completion for the current shell."


app = typer.Typer(
    name="aai",
    # No top-level `help=`: the bare-`aai` welcome banner already carries the
    # "AssemblyAI from your terminal" tagline, so a description here would duplicate it.
    # `aai --install-completion` / `--show-completion` for bash/zsh/fish/PowerShell,
    # the discoverability affordance gh/kubectl/docker users reach for.
    add_completion=True,
    cls=_OrderedGroup,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit when `aai --version`/`-V` is passed, before any command
    runs. Mirrors the reflex (`tool --version`) every other CLI answers; the `version`
    subcommand stays for parity."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _profile_has_key(state: AppState) -> bool:
    try:
        config.resolve_api_key(profile=state.profile)
    except NotAuthenticated:
        return False
    return True


def _interactive_session() -> bool:
    """True only when both ends are a real TTY (so we never block a piped/CI run)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


_RAW_ARGS_META_KEY = "aai_raw_args"


def _command_line_requests_json(raw_args: list[str]) -> bool:
    """Whether the token list opts into JSON (``--json``, ``-o json``, ``--output json``,
    or their glued forms).

    The root callback runs before the subcommand parses its own ``--json``, so a failure
    raised here (e.g. a bad ``--env``) would otherwise always render human text — leaving a
    ``… --json`` pipeline without the uniform ``{"error": …}`` shape it relies on. The group
    stashes the raw token list in ``ctx.meta`` (see ``_OrderedGroup.parse_args``) before the
    callback runs, so sniffing it lets every failure class honor the request.
    """
    for index, token in enumerate(raw_args):
        if token in ("--json", "--output=json", "-ojson"):
            return True
        if token in ("-o", "--output") and raw_args[index + 1 : index + 2] == ["json"]:
            return True
    return False


def _offer_or_help(ctx: typer.Context, state: AppState) -> None:
    """No subcommand given: offer guided setup to a credential-less, interactive user;
    otherwise print help. Never prompts in a non-interactive session, and never on
    `--help` (Click handles that eagerly before the callback)."""
    if not state.quiet:
        output.print_banner()
    if _interactive_session() and not _profile_has_key(state):
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
            ("Guided setup (start here)", "aai onboard"),
            ("Transcribe a file", "aai transcribe call.mp3"),
            ("Stream live audio in real time", "aai stream"),
            ("Talk to a voice agent", "aai agent"),
            (
                "Summarize while transcribing",
                'aai transcribe call.mp3 --llm "summarize action items"',
            ),
        ]
    ),
)
def main(
    ctx: typer.Context,
    profile: str | None = typer.Option(None, "--profile", "-p", help="Named credential profile."),
    env: str | None = typer.Option(
        None, "--env", help="Backend environment (production, sandbox000)."
    ),
    sandbox: bool = typer.Option(False, "--sandbox", help="Shortcut for --env sandbox000."),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress non-essential messages (warnings, hints)."
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
        # default. The plain `aai --version` path behaves identically with or without this,
        # so there's no cheap test that distinguishes it.
        is_eager=True,  # pragma: no mutate
    ),
) -> None:
    if sandbox and env is None:
        env = "sandbox000"
    state = AppState(profile=profile, env=env, quiet=quiet)
    ctx.obj = state
    # The command's own --json flag isn't parsed yet, so sniff the pending command line:
    # a root-callback failure (e.g. bad --env) still emits the JSON error shape when the
    # invocation opted into JSON, and renders human text on stderr otherwise.
    raw_args: list[str] = ctx.meta.get(_RAW_ARGS_META_KEY, [])
    json_mode = output.resolve_json(explicit=_command_line_requests_json(raw_args))
    try:
        environments.set_active(resolve_environment(state))
    except CLIError as err:
        output.emit_error(err, json_mode=json_mode)
        raise typer.Exit(code=err.exit_code) from None
    warning = env_override_warning(state)
    if warning and not quiet:
        output.error_console.print(output.warn(warning))
    if ctx.invoked_subcommand is None:
        _offer_or_help(ctx, state)


# Help-panel grouping: named sub-typers carry their panel on `add_typer`; merged
# (nameless) sub-typers don't propagate it, so those commands set `rich_help_panel`
# on their own `@app.command()` (see each command module). Final ordering within a
# panel is controlled by `_COMMAND_ORDER` via `_OrderedGroup`, not registration order.
app.add_typer(transcribe.app)
app.add_typer(stream.app)
app.add_typer(transcripts.app, name="transcripts", rich_help_panel=help_panels.HISTORY)
app.add_typer(sessions.app, name="sessions", rich_help_panel=help_panels.HISTORY)
app.add_typer(audit.app)  # audit
app.add_typer(agent.app)
app.add_typer(llm.app)
app.add_typer(account.app)  # balance, usage, limits
app.add_typer(login.app)  # login, logout, whoami
app.add_typer(doctor.app)
app.add_typer(init.app)
app.add_typer(onboard.app)
app.add_typer(setup.app, name="setup", rich_help_panel=help_panels.SETUP)
app.add_typer(keys.app, name="keys", rich_help_panel=help_panels.ACCOUNT)


def run() -> None:
    """Console-script entry point: run the app, exiting cleanly on a closed pipe.

    A downstream consumer (e.g. `aai … | head`) can close the pipe before we finish
    writing. Without this, the write — or Python's flush at shutdown — raises
    BrokenPipeError and prints an ugly "Exception ignored" traceback. We treat a
    closed pipe as success: silence stdout and exit 0. Streaming commands also catch
    it earlier; this is the catch-all for the one-shot `output.emit`/`print` paths.
    """
    try:
        app(prog_name="aai")
    except BrokenPipeError:
        stdio.silence_stdout()
        sys.exit(0)
