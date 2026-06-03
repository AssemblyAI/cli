from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from assemblyai_cli import client, config, output
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import APIError

app = typer.Typer()


@app.command()
def get(
    ctx: typer.Context,
    transcript_id: str = typer.Argument(..., help="Transcript id."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Fetch a past transcript by id and print its text."""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        transcript = client.get_transcript(api_key, transcript_id)
        if getattr(transcript.status, "value", transcript.status) == "error":
            raise APIError(
                getattr(transcript, "error", None) or "Transcript failed.",
                transcript_id=transcript_id,
            )
        output.emit(
            {
                "id": transcript.id,
                "status": getattr(transcript.status, "value", transcript.status),
                "text": transcript.text,
            },
            lambda d: escape(str(d["text"])),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(name="list")
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many transcripts to show."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent transcripts."""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        rows = client.list_transcripts(api_key, limit=limit)

        def render(data: list[dict[str, object]]) -> Table:
            table = Table("id", "status", "created")
            for row in data:
                table.add_row(
                    escape(str(row["id"])),
                    escape(str(row["status"])),
                    escape(str(row.get("created", ""))),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
