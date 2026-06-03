from __future__ import annotations

import click
import typer
from typer.core import TyperGroup

from assemblyai_cli import __version__
from assemblyai_cli.commands import (
    agent,
    claude,
    doctor,
    llm,
    login,
    samples,
    stream,
    transcribe,
    transcripts,
)
from assemblyai_cli.context import AppState

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
    "claude",
    "version",
)


class _OrderedGroup(TyperGroup):
    """Lists commands in `_COMMAND_ORDER` rather than registration order.

    Typer renders all direct commands before sub-typer groups, so registration
    order alone can't place `version` last; sorting here controls help output.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
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
) -> None:
    ctx.obj = AppState(profile=profile)


# Commands are registered in the order they should appear in `aai --help`:
# core transcription first, then voice/LLM, then account, then tooling. `version`
# is defined last so it sorts to the bottom (registration order is preserved).
app.add_typer(transcribe.app)
app.add_typer(stream.app)
app.add_typer(transcripts.app, name="transcripts")
app.add_typer(agent.app)
app.add_typer(llm.app)
app.add_typer(login.app)  # login, logout, whoami
app.add_typer(doctor.app)
app.add_typer(samples.app, name="samples")
app.add_typer(claude.app, name="claude")


@app.command()
def version() -> None:
    """Show the CLI version."""
    typer.echo(__version__)
