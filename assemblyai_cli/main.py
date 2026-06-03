from __future__ import annotations

import typer

from assemblyai_cli import __version__
from assemblyai_cli.commands import (
    agent,
    claude,
    llm,
    login,
    samples,
    stream,
    transcribe,
    transcripts,
)
from assemblyai_cli.context import AppState

app = typer.Typer(
    name="aai",
    help="Command-line interface for AssemblyAI.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    profile: str = typer.Option(None, "--profile", "-p", help="Named credential profile."),
) -> None:
    ctx.obj = AppState(profile=profile)


@app.command()
def version() -> None:
    """Show the CLI version."""
    typer.echo(__version__)


app.add_typer(agent.app)
app.add_typer(llm.app)
app.add_typer(login.app)
app.add_typer(stream.app)
app.add_typer(transcribe.app)
app.add_typer(transcripts.app, name="transcripts")
app.add_typer(samples.app, name="samples")
app.add_typer(claude.app, name="claude")
