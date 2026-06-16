from __future__ import annotations

import typer
from rich.markup import escape

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.app.transcribe.run import render_transform_steps
from aai_cli.core import choices, client, llm, stdio, timeparse
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
            ("Grab the latest transcript id", "assembly transcripts list -o id | head -1"),
            (
                "Summarize your latest transcript",
                'assembly llm "summarize" --transcript-id '
                "$(assembly transcripts list -o id | head -1)",
            ),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many transcripts to show", min=1),
    json_out: bool = options.json_option(),
    fields: str | None = options.fields_option(),
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

        output.emit(rows, render, json_mode=json_mode, fields=fields)

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


def _emit_transcript(transcript: object, *, json_mode: bool, batch: bool) -> None:
    """Render one fetched transcript (no -o field, no --llm) for the chosen output mode."""
    if json_mode and batch:
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


def _id_of(transcript: object) -> str:
    return str(getattr(transcript, "id", "") or "")


def _text_of(transcript: object) -> str:
    return str(getattr(transcript, "text", "") or "")


def _emit_transform(
    transcript: object,
    model: str,
    steps: list[dict[str, str]],
    *,
    json_mode: bool,
    batch: bool,
) -> None:
    """Emit a transcript's ``--llm`` chain result: NDJSON per id in batch, else like `transcribe`."""
    record = client.transcript_summary(transcript) | {"transform": {"model": model, "steps": steps}}
    if json_mode and batch:
        output.emit_ndjson({"type": "transcript", **record})
    else:
        output.emit(record, render_transform_steps, json_mode=json_mode)


def _deliver_transcript(
    transcript: object,
    api_key: str,
    *,
    output_field: choices.TranscriptOutput | None,
    chars_per_caption: int | None,
    chain: list[str],
    model: str,
    max_tokens: int,
    json_mode: bool,
    batch: bool,
    suppress: bool,
) -> str:
    """Emit one fetched transcript (unless ``suppress``ed for a pending reduce) and return
    its ``--llm-reduce`` contribution — the last ``--llm`` output, else the transcript text."""
    if output_field is not None:
        # -o wins over the chain, matching `transcribe` deliver_result precedence; a
        # pending human reduce suppresses the per-id field so only the aggregate prints.
        if not suppress:
            output.emit_text(
                client.select_transcript_field(
                    transcript, output_field, chars_per_caption=chars_per_caption
                )
            )
        return _text_of(transcript)
    if chain:
        steps = llm.run_chain_steps(
            api_key, chain, transcript_id=_id_of(transcript), model=model, max_tokens=max_tokens
        )
        if not suppress:
            _emit_transform(transcript, model, steps, json_mode=json_mode, batch=batch)
        return steps[-1]["output"] if steps else ""
    if not suppress:
        _emit_transcript(transcript, json_mode=json_mode, batch=batch)
    return _text_of(transcript)


def _run_reduce(
    api_key: str,
    contributions: list[tuple[str, str]],
    *,
    prompts: list[str],
    model: str,
    max_tokens: int,
    json_mode: bool,
) -> None:
    """Run the ``--llm-reduce`` chain once over every fetched transcript; print to stdout.

    Mirrors `transcribe`'s reduce: concatenate each id's contribution under a header,
    skip the billable call when there's nothing to reduce, and emit the same additive
    ``{"type": "reduce", …}`` NDJSON record under --json.
    """
    combined = "\n\n".join(f"### Transcript: {tid}\n{text}" for tid, text in contributions if text)
    if not combined:
        output.emit_warning(
            "Nothing to reduce: no transcript text across ids.", json_mode=json_mode
        )
        return
    result = llm.run_chain(
        api_key, prompts, transcript_text=combined, model=model, max_tokens=max_tokens
    )
    if json_mode:
        output.emit_ndjson({"type": "reduce", "model": model, "prompts": prompts, "output": result})
    else:
        output.emit_text(result)


@app.command(
    epilog=examples_epilog(
        [
            ("Fetch a transcript's text by id", "assembly transcripts get 5551234-abcd"),
            ("Speaker-labeled turns", "assembly transcripts get 5551234-abcd -o utterances"),
            ("Save SRT subtitles", "assembly transcripts get 5551234-abcd -o srt > captions.srt"),
            ("Get the raw JSON", "assembly transcripts get 5551234-abcd --json"),
            (
                "Fetch many at once from a piped list",
                "assembly transcripts list --json | assembly transcripts get -o text",
            ),
            (
                "Summarize each transcript in a piped list",
                "assembly transcripts list --json | "
                'assembly transcripts get --llm "Summarize this call"',
            ),
            (
                "Rank a piped list with one reduce prompt",
                "assembly transcripts list --json | "
                'assembly transcripts get --llm-reduce "Rank these worst-to-best"',
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
    llm_prompt: list[str] | None = typer.Option(
        None,
        "--llm",
        help="Transform each transcript through LLM Gateway. Repeatable: each prompt runs "
        "on the previous one's response (a chain), the first on the transcript.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    llm_reduce: list[str] | None = typer.Option(
        None,
        "--llm-reduce",
        help="Run one LLM-Gateway prompt over all fetched transcripts (a reduce). "
        "Repeatable: each runs on the previous one's output. For a single id it "
        "extends the --llm chain over that transcript.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Fetch a past transcript by id and print its text

    Omit the id to read transcript ids from stdin — one per line, or the JSON from
    `assembly transcripts list --json`. Add --llm to transform each transcript through
    LLM Gateway (a map), or --llm-reduce to run one prompt over them all (a reduce).
    """

    def body(state: AppState, json_mode: bool) -> None:
        # Cheap local validation first: a malformed id or flag conflict is a usage
        # error whether or not the user is signed in, so it must not trigger auth.
        client.validate_chars_per_caption(chars_per_caption, output_field)
        map_prompts = list(llm_prompt or [])
        reduce_prompts = list(llm_reduce or [])
        ids, batch = _resolve_ids(transcript_id)
        for tid in ids:
            client.validate_transcript_id(tid)
        # A single source has nothing to aggregate, so --llm-reduce just extends the
        # --llm chain over that transcript (mirrors `transcribe`); a stdin batch runs
        # the reduce separately over every fetched transcript.
        per_transcript_chain = map_prompts if batch else map_prompts + reduce_prompts
        do_reduce = batch and bool(reduce_prompts)
        api_key = state.resolve_api_key()
        contributions: list[tuple[str, str]] = []
        for tid in ids:
            transcript = client.get_transcript(api_key, tid)
            if client.status_str(transcript) == "error":
                raise APIError(
                    getattr(transcript, "error", None) or "Transcript failed.",
                    transcript_id=tid,
                )
            contribution = _deliver_transcript(
                transcript,
                api_key,
                output_field=output_field,
                chars_per_caption=chars_per_caption,
                chain=per_transcript_chain,
                model=model,
                max_tokens=max_tokens,
                json_mode=json_mode,
                batch=batch,
                suppress=do_reduce and not json_mode,
            )
            contributions.append((tid, contribution))
        if do_reduce:
            _run_reduce(
                api_key,
                contributions,
                prompts=reduce_prompts,
                model=model,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

    run_command(ctx, body, json=json_out)
