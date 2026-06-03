from __future__ import annotations

import typer
from rich.markup import escape

from assemblyai_cli import config, output
from assemblyai_cli import llm as gateway
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError

app = typer.Typer()


@app.command()
def llm(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, help="The instruction / prompt to send."),
    model: str = typer.Option(gateway.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    transcript_id: str = typer.Option(
        None, "--transcript-id", help="Inject this transcript's text into the prompt."
    ),
    system: str = typer.Option(None, "--system", help="Optional system prompt."),
    max_tokens: int = typer.Option(
        gateway.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens to generate."
    ),
    list_models: bool = typer.Option(False, "--list-models", help="Print known models and exit."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Send a prompt to AssemblyAI's LLM Gateway and print the response.

    With --transcript-id the transcript's text is injected server-side, so you
    can ask questions about a past transcription (e.g. aai llm "summarize" --transcript-id ID).
    """

    if list_models:
        typer.echo("\n".join(gateway.KNOWN_MODELS))
        raise typer.Exit(code=0)

    def body(state: AppState, json_mode: bool) -> None:
        if not prompt:
            raise UsageError("Provide a prompt, or use --list-models.")
        api_key = config.resolve_api_key(profile=state.profile)
        messages = gateway.build_messages(prompt, system=system, transcript_id=transcript_id)
        response = gateway.complete(
            api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            transcript_id=transcript_id,
        )
        output.emit(
            {
                "model": model,
                "output": gateway.content_of(response),
                "usage": gateway.usage_of(response),
            },
            lambda d: escape(str(d["output"])),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
