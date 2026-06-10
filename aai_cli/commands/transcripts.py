from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import choices, client, config, options, output, theme
from aai_cli.context import AppState, run_command
from aai_cli.errors import APIError
from aai_cli.help_text import examples_epilog

app = typer.Typer(help="Browse and fetch past transcripts.", no_args_is_help=True)


@app.command(
    epilog=examples_epilog(
        [
            ("Fetch a transcript's text by id", "aai transcripts get 5551234-abcd"),
            ("Speaker-labeled turns", "aai transcripts get 5551234-abcd -o utterances"),
            ("Save SRT subtitles", "aai transcripts get 5551234-abcd -o srt > captions.srt"),
            ("Get the raw JSON", "aai transcripts get 5551234-abcd --json"),
        ]
    )
)
def get(
    ctx: typer.Context,
    transcript_id: str = typer.Argument(..., help="Transcript id."),
    output_field: choices.TranscriptOutput | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field of the result.",
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Fetch a past transcript by id and print its text."""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        transcript = client.get_transcript(api_key, transcript_id)
        if client.status_str(transcript) == "error":
            raise APIError(
                getattr(transcript, "error", None) or "Transcript failed.",
                transcript_id=transcript_id,
            )
        if output_field is not None:
            # Raw single-field output for pipelines (overrides --json), matching `transcribe`.
            output.emit_text(client.select_transcript_field(transcript, output_field))
            return
        output.emit(
            client.transcript_summary(transcript),
            lambda d: escape(str(d["text"])),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List your recent transcripts", "aai transcripts list"),
            ("Show more at once", "aai transcripts list --limit 50"),
            ("Grab the latest transcript id", "aai transcripts list --json | jq -r '.[0].id'"),
            (
                "Summarize your latest transcript",
                'aai llm "summarize" --transcript-id '
                "$(aai transcripts list --json | jq -r '.[0].id')",
            ),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many transcripts to show.", min=1),
    json_out: bool = options.json_option(),
) -> None:
    """List recent transcripts."""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        rows = client.list_transcripts(api_key, limit=limit)

        def render(data: list[dict[str, object]]) -> Table:
            table = output.data_table("id", "status", "created")
            for row in data:
                table.add_row(
                    escape(str(row["id"])),
                    theme.status_text(str(row["status"])),
                    escape(str(row.get("created", ""))),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
