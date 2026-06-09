from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import typer
from typer.core import TyperGroup

if TYPE_CHECKING:
    # Typer (>=0.13) vendors its own click; TyperGroup.list_commands receives this
    # context type, not the upstream click.Context. Imported for typing only.
    from typer._click.core import Context as ClickContext

from aai_cli import __version__, environments, help_panels, output, stdio
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
    samples,
    sessions,
    setup,
    stream,
    transcribe,
    transcripts,
)
from aai_cli.context import AppState, env_override_warning, resolve_environment
from aai_cli.errors import CLIError
from aai_cli.help_text import examples_epilog

# The order commands appear under `aai --help`. Commands are grouped into named
# Rich panels (see `help_panels.py`); panels render in the order their first
# command appears here, so keep each panel's commands contiguous and ordered
# most-common-first. Names not listed fall to the end, sorted alphabetically.
_COMMAND_ORDER = (
    # Quick Start — zero-to-running onboarding
    "init",
    # Setup & Tools — get set up & maintain; `version` last
    "samples",
    "doctor",
    "setup",
    "version",
    # Transcription & AI — the verbs you run
    "transcribe",
    "stream",
    "agent",
    "llm",
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
    order alone can't place `version` last; sorting here controls help output.
    """

    def list_commands(self, ctx: ClickContext) -> list[str]:
        rank = {name: i for i, name in enumerate(_COMMAND_ORDER)}
        return sorted(
            super().list_commands(ctx), key=lambda name: (rank.get(name, len(rank)), name)
        )


app = typer.Typer(
    name="aai",
    help="AssemblyAI from your terminal — transcribe, stream, and build voice AI.",
    no_args_is_help=True,
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


@app.callback(
    epilog=examples_epilog(
        [
            ("Sign in with your browser", "aai login"),
            ("Transcribe a file", "aai transcribe call.mp3"),
            ("Scaffold a starter app", "aai init"),
        ]
    )
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
    # The command's own --json flag isn't parsed yet, and output is human-by-default, so a
    # root-callback (e.g. bad --env) error renders as human text on stderr.
    json_mode = output.resolve_json(explicit=False)
    try:
        environments.set_active(resolve_environment(state))
    except CLIError as err:
        output.emit_error(err, json_mode=json_mode)
        raise typer.Exit(code=err.exit_code) from None
    warning = env_override_warning(state)
    if warning and not quiet:
        output.error_console.print(output.warn(warning))


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
app.add_typer(samples.app, name="samples", rich_help_panel=help_panels.SETUP)
app.add_typer(init.app)
app.add_typer(onboard.app)
app.add_typer(setup.app, name="setup", rich_help_panel=help_panels.SETUP)
app.add_typer(keys.app, name="keys", rich_help_panel=help_panels.ACCOUNT)


@app.command(rich_help_panel=help_panels.SETUP)
def version() -> None:
    """Show the CLI version."""
    typer.echo(__version__)


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
