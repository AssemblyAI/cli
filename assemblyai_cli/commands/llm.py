from __future__ import annotations

import typer
from rich.markup import escape

from assemblyai_cli import config, output, stdio
from assemblyai_cli import llm as gateway
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError
from assemblyai_cli.follow import FollowRenderer

app = typer.Typer()


@app.command()
def llm(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, help="The prompt to send to the model."),
    # Note: text piped on stdin is injected into the prompt (e.g. `cat notes | aai llm "summarize"`).
    model: str = typer.Option(gateway.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    transcript_id: str = typer.Option(
        None, "--transcript-id", help="Inject this transcript's text into the prompt."
    ),
    system: str = typer.Option(None, "--system", help="Optional system prompt."),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Re-run the prompt over a growing transcript piped on stdin, refreshing "
        "the answer in place on every finalized turn (e.g. aai stream -o text | aai "
        'llm -f "summarize action items as I talk"). Ctrl-C to stop.',
    ),
    output_field: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field of the result: text (just the answer, pipe-friendly) or json.",
    ),
    max_tokens: int = typer.Option(
        gateway.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens to generate."
    ),
    list_models: bool = typer.Option(False, "--list-models", help="Print known models and exit."),
    json_out: bool = typer.Option(
        False, "--json", help="Output raw JSON (one object per turn in --follow mode)."
    ),
) -> None:
    """Send a prompt to AssemblyAI's LLM Gateway and print the response.

    With --transcript-id the transcript's text is injected server-side, so you
    can ask questions about a past transcription (e.g. aai llm "summarize" --transcript-id ID).
    """

    if list_models:
        typer.echo("\n".join(gateway.KNOWN_MODELS))
        raise typer.Exit(code=0)

    def follow_body(state: AppState, json_mode: bool) -> None:
        if not prompt:
            raise UsageError("Provide a prompt to run over the streamed transcript.")
        if output_field is not None:
            raise UsageError(
                "--output applies to one-shot mode; --follow renders a live panel "
                "(or NDJSON when piped)."
            )
        if transcript_id:
            raise UsageError(
                "--follow runs over live transcript text piped on stdin; it can't be "
                "combined with --transcript-id."
            )
        if not stdio.stdin_is_piped():
            raise UsageError(
                "--follow needs transcript text piped on stdin, e.g. "
                '`aai stream -o text | aai llm -f "summarize action items as I talk"`.'
            )
        api_key = config.resolve_api_key(profile=state.profile)

        def ask(transcript_text: str) -> str:
            messages = gateway.build_messages(
                prompt, system=system, transcript_text=transcript_text
            )
            response = gateway.complete(
                api_key, model=model, messages=messages, max_tokens=max_tokens
            )
            return gateway.content_of(response)

        with FollowRenderer(json_mode=json_mode) as render:
            transcript: list[str] = []
            try:
                for turn in stdio.iter_piped_stdin_lines():
                    transcript.append(turn)
                    render(ask("\n".join(transcript)), len(transcript))
            except KeyboardInterrupt:
                # Ctrl-C is the normal "stop watching" signal -> exit cleanly (code 0).
                pass

    def body(state: AppState, json_mode: bool) -> None:
        if not prompt:
            raise UsageError("Provide a prompt, or use --list-models.")
        output.validate_output_field(output_field, ("text", "json"))
        api_key = config.resolve_api_key(profile=state.profile)
        # Text piped on stdin becomes the content the prompt operates on, unless an
        # explicit --transcript-id is given (that injects server-side and takes priority).
        stdin_text = stdio.piped_stdin_text() if not transcript_id else None
        messages = gateway.build_messages(
            prompt, system=system, transcript_id=transcript_id, transcript_text=stdin_text
        )
        response = gateway.complete(
            api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            transcript_id=transcript_id,
        )
        content = gateway.content_of(response)
        if output_field == "text":
            # Just the answer, raw — so `… | aai llm -o text "…" | next` composes cleanly.
            print(content)
            return
        output.emit(
            {"model": model, "output": content, "usage": gateway.usage_of(response)},
            lambda d: escape(str(d["output"])),
            json_mode=json_mode or output_field == "json",
        )

    run_command(ctx, follow_body if follow else body, json=json_out)
