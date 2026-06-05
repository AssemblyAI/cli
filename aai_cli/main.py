from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import typer
from typer.core import TyperGroup

if TYPE_CHECKING:
    # Typer (>=0.13) vendors its own click; TyperGroup.list_commands receives this
    # context type, not the upstream click.Context. Imported for typing only.
    from typer._click.core import Context as ClickContext

from aai_cli import __version__, environments, stdio
from aai_cli.commands import (
    account,
    agent,
    claude,
    doctor,
    init,
    keys,
    llm,
    login,
    samples,
    stream,
    transcribe,
    transcripts,
)
from aai_cli.context import AppState, env_override_warning, resolve_environment
from aai_cli.errors import CLIError

# The order commands appear under `aai --help`: core transcription first, then
# voice/LLM, then account, then tooling, with `version` last. Names not listed
# fall to the end, sorted alphabetically.
_COMMAND_ORDER = (
    "transcribe",
    "stream",
    "transcripts",
    "agent",
    "llm",
    "login",
    "logout",
    "whoami",
    "doctor",
    "samples",
    "init",
    "claude",
    "version",
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
    help="Command-line interface for AssemblyAI.",
    no_args_is_help=True,
    add_completion=False,
    cls=_OrderedGroup,
)


@app.callback()
def main(
    ctx: typer.Context,
    profile: str = typer.Option(None, "--profile", "-p", help="Named credential profile."),
    env: str = typer.Option(None, "--env", help="Backend environment (production, sandbox000)."),
    sandbox: bool = typer.Option(False, "--sandbox", help="Shortcut for --env sandbox000."),
) -> None:
    if sandbox and env is None:
        env = "sandbox000"
    state = AppState(profile=profile, env=env)
    ctx.obj = state
    try:
        environments.set_active(resolve_environment(state))
    except CLIError as err:
        typer.echo(err.message, err=True)
        raise typer.Exit(code=err.exit_code) from None
    warning = env_override_warning(state)
    if warning:
        typer.echo(warning, err=True)


# Commands are registered in the order they should appear in `aai --help`:
# core transcription first, then voice/LLM, then account, then tooling. `version`
# is defined last so it sorts to the bottom (registration order is preserved).
app.add_typer(transcribe.app)
app.add_typer(stream.app)
app.add_typer(transcripts.app, name="transcripts")
app.add_typer(agent.app)
app.add_typer(llm.app)
app.add_typer(account.app)  # balance, usage, limits
app.add_typer(login.app)  # login, logout, whoami
app.add_typer(doctor.app)
app.add_typer(samples.app, name="samples")
app.add_typer(init.app)
app.add_typer(claude.app, name="claude")
app.add_typer(keys.app, name="keys")


@app.command()
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
