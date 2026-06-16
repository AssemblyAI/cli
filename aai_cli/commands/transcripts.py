from __future__ import annotations

import typer
from rich.markup import escape

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.core import choices, client, stdio, timeparse
from aai_cli.core.errors import APIError, UsageError
from aai_cli.ui import output, theme
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer(help="Browse and fetch past transcripts", no_args_is_help=True)

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.HISTORY,
    order=10,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("transcripts",),
    group_name="transcripts",
)


# `list` is registered before `get` so the subcommand help lists them in that
# order, matching `assembly sessions --help`.
@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List your recent transcripts", "assembly transcripts list"),
            ("Show more at once", "assembly transcripts list --limit 50"),
            ("Grab the latest transcript id", "assembly transcripts list --json | jq -r '.[0].id'"),
            (
                "Summarize your latest transcript",
                'assembly llm "summarize" --transcript-id '
                "$(assembly transcripts list --json | jq -r '.[0].id')",
            ),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many transcripts to show", min=1),
    json_out: bool = options.json_option(),
) -> None:
    """List recent transcripts"""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = state.resolve_api_key()
        rows = client.list_transcripts(api_key, limit=limit)

        def render(data: list[dict[str, object]]) -> object:
            if not data:
                return output.muted("No transcripts yet.")
            table = output.data_table("id", "status", "created (UTC)")
            for row in data:
                table.add_row(
                    escape(str(row["id"])),
                    theme.status_text(str(row["status"])),
                    escape(timeparse.format_utc_datetime(row.get("created"))),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


def _resolve_ids(transcript_id: str | None) -> tuple[list[str], bool]:
    """The transcript ids to fetch, and whether this is a stdin batch.

    A positional id stays the single-fetch path (output shape unchanged). With no
    id, ids are read from piped stdin so ``transcripts list --json | …`` composes;
    running interactively with no id is a usage error rather than a hang.
    """
    if transcript_id is not None:
        return [transcript_id], False
    piped = stdio.piped_stdin_text()
    if piped is None:
        raise UsageError(
            "Give a transcript id, or pipe transcript ids on stdin.",
            suggestion="e.g. assembly transcripts list --json | assembly transcripts get -o text",
        )
    ids = client.parse_transcript_ids(piped)
    if not ids:
        raise UsageError(
            "No transcript ids found on stdin.",
            suggestion="Pipe `assembly transcripts list --json`, or one id per line.",
        )
    return ids, True


def _emit_transcript(
    transcript: object,
    output_field: choices.TranscriptOutput | None,
    chars_per_caption: int | None,
    *,
    json_mode: bool,
    batch: bool,
) -> None:
    """Render one fetched transcript for the chosen output mode."""
    if output_field is not None:
        # Raw single-field output for pipelines (overrides --json), matching `transcribe`.
        output.emit_text(
            client.select_transcript_field(
                transcript, output_field, chars_per_caption=chars_per_caption
            )
        )
    elif json_mode and batch:
        # One NDJSON record per id so a downstream stage can map over the stream;
        # "type" discriminates NDJSON lines CLI-wide (matching `transcribe` batch).
        output.emit_ndjson({"type": "transcript", **client.transcript_json_payload(transcript)})
    elif json_mode:
        # The full SDK payload, identical to `assembly transcribe … --json`, so the
        # same `jq` works whether the transcript is fetched fresh or re-fetched.
        output.emit(client.transcript_json_payload(transcript), lambda d: d, json_mode=True)
    elif batch:
        output.emit_text(str(client.transcript_summary(transcript)["text"]))
    else:
        output.emit(
            client.transcript_summary(transcript),
            lambda d: escape(str(d["text"])),
            json_mode=False,
        )


@app.command(
    epilog=examples_epilog(
        [
            ("Fetch a transcript's text by id", "assembly transcripts get 5551234-abcd"),
            ("Speaker-labeled turns", "assembly transcripts get 5551234-abcd -o utterances"),
            ("Save SRT subtitles", "assembly transcripts get 5551234-abcd -o srt > captions.srt"),
            ("Save VTT subtitles", "assembly transcripts get 5551234-abcd -o vtt > captions.vtt"),
            ("Get the raw JSON", "assembly transcripts get 5551234-abcd --json"),
            (
                "Fetch many at once from a piped list",
                "assembly transcripts list --json | assembly transcripts get -o text",
            ),
        ]
    )
)
def get(
    ctx: typer.Context,
    transcript_id: str | None = typer.Argument(
        None, help="Transcript id; omit to read ids from stdin"
    ),
    output_field: choices.TranscriptOutput | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field of the result",
    ),
    chars_per_caption: int | None = options.chars_per_caption_option(),
    json_out: bool = options.json_option(),
) -> None:
    """Fetch a past transcript by id and print its text"""

    def body(state: AppState, json_mode: bool) -> None:
        # Cheap local validation first: a malformed id or flag conflict is a usage
        # error whether or not the user is signed in, so it must not trigger auth.
        client.validate_chars_per_caption(chars_per_caption, output_field)
        ids, batch = _resolve_ids(transcript_id)
        for tid in ids:
            client.validate_transcript_id(tid)
        api_key = state.resolve_api_key()
        for tid in ids:
            transcript = client.get_transcript(api_key, tid)
            if client.status_str(transcript) == "error":
                raise APIError(
                    getattr(transcript, "error", None) or "Transcript failed.",
                    transcript_id=tid,
                )
            _emit_transcript(
                transcript, output_field, chars_per_caption, json_mode=json_mode, batch=batch
            )

    run_command(ctx, body, json=json_out)
