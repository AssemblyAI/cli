from __future__ import annotations

import typer
from rich.markup import escape

from aai_cli import choices, config, help_panels, options, output, stdio
from aai_cli import llm as gateway
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.follow import FollowRenderer
from aai_cli.help_text import examples_epilog

app = typer.Typer()

_FOLLOW_STDIN_MESSAGE = (
    "--follow needs transcript text piped on stdin, e.g. "
    '`assembly stream -o text | assembly llm -f "summarize action items as I talk"`.'
)


def _validate_follow_args(
    prompt: str | None, output_field: str | None, transcript_id: str | None
) -> str:
    """Reject flag combinations that don't apply to --follow's live-panel mode.

    Returns the validated (non-empty) prompt so the caller has a plain ``str``.
    """
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
        raise UsageError(_FOLLOW_STDIN_MESSAGE)
    return prompt


def _emit_model_list(_state: AppState, json_mode: bool) -> None:
    """--list-models body, routed through run_command so --json yields a
    machine-readable array instead of the human list; needs no auth."""
    output.emit(list(gateway.KNOWN_MODELS), "\n".join, json_mode=json_mode)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            (
                "Ask about a past transcript",
                'assembly llm "summarize the key decisions" --transcript-id 5551234-abcd',
            ),
            ("Pipe any text in", 'echo "meeting notes" | assembly llm "turn into action items"'),
            (
                "Pick a model and add a system prompt",
                'assembly llm "draft a follow-up email" --model claude-opus-4-7 --system "Be concise."',
            ),
            ("See available models", "assembly llm --list-models"),
        ]
    ),
)
def llm(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(None, help="The prompt to send to the model."),
    # Note: text piped on stdin is injected into the prompt (e.g. `cat notes | assembly llm "summarize"`).
    model: str = typer.Option(
        gateway.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model.",
        autocompletion=gateway.complete_model,
    ),
    transcript_id: str | None = typer.Option(
        None, "--transcript-id", help="Inject this transcript's text into the prompt."
    ),
    system: str | None = typer.Option(None, "--system", help="Optional system prompt."),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Re-run the prompt over a growing transcript piped on stdin, refreshing "
        "the answer in place on every finalized turn (e.g. assembly stream -o text | assembly "
        'llm -f "summarize action items as I talk"). Ctrl-C to stop.',
    ),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field of the result: text (just the answer, pipe-friendly) or json.",
    ),
    max_tokens: int = typer.Option(
        gateway.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens to generate."
    ),
    list_models: bool = typer.Option(False, "--list-models", help="Print known models and exit."),
    json_out: bool = options.json_option("Output raw JSON (one object per turn in --follow mode)."),
) -> None:
    """Send a prompt to AssemblyAI's LLM Gateway and print the response.

    With --transcript-id the transcript's text is injected server-side, so you
    can ask questions about a past transcription (e.g. assembly llm "summarize" --transcript-id ID).
    """

    if list_models:
        run_command(ctx, _emit_model_list, json=json_out)
        return

    def follow_body(state: AppState, json_mode: bool) -> None:
        prompt_text = _validate_follow_args(prompt, output_field, transcript_id)
        api_key = config.resolve_api_key(profile=state.profile)

        def ask(transcript_text: str) -> str:
            messages = gateway.build_messages(
                prompt_text, system=system, transcript_text=transcript_text
            )
            response = gateway.complete(
                api_key, model=model, messages=messages, max_tokens=max_tokens
            )
            return gateway.content_of(response)

        transcript: list[str] = []
        interrupted = False
        with FollowRenderer(json_mode=json_mode) as render:
            # Ctrl-C is the normal "stop watching" signal -> exit cleanly (code 0).
            try:
                for turn in stdio.iter_piped_stdin_lines():
                    transcript.append(turn)
                    render(ask("\n".join(transcript)), len(transcript))
            except KeyboardInterrupt:
                interrupted = True
        if not transcript and not interrupted:
            # An empty pipe (`assembly llm -f "…" </dev/null`) would otherwise exit 0
            # silently, having asked nothing.
            raise UsageError(_FOLLOW_STDIN_MESSAGE)

    def body(state: AppState, json_mode: bool) -> None:
        if not prompt:
            raise UsageError(
                "Provide a prompt.",
                suggestion="Or pass --list-models to see available models.",
            )
        prompt_text = prompt
        api_key = config.resolve_api_key(profile=state.profile)
        # Text piped on stdin becomes the content the prompt operates on, unless an
        # explicit --transcript-id is given (that injects server-side and takes priority).
        stdin_text = stdio.piped_stdin_text() if not transcript_id else None
        messages = gateway.build_messages(
            prompt_text, system=system, transcript_id=transcript_id, transcript_text=stdin_text
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
            # Just the answer, raw — so `… | assembly llm -o text "…" | next` composes cleanly.
            output.emit_text(content)
            return
        output.emit(
            {"model": model, "output": content, "usage": gateway.usage_of(response)},
            lambda d: escape(str(d["output"])),
            json_mode=json_mode or output_field == "json",
        )

    run_command(ctx, follow_body if follow else body, json=json_out)
