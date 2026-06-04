from __future__ import annotations

import json

import typer
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel

from assemblyai_cli import config, output, stdio
from assemblyai_cli import llm as gateway
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError

app = typer.Typer()


class _FollowRenderer:
    """Render a live transcript transform that refreshes on every turn.

    On a terminal, the latest answer is redrawn in place inside a Rich panel so a
    human watches one evolving summary. When piped or run by an agent (json_mode),
    each refresh is emitted as one NDJSON object so it stays machine-readable.
    """

    def __init__(self, *, json_mode: bool) -> None:
        self.json_mode = json_mode
        self._live: Live | None = None

    def __enter__(self) -> _FollowRenderer:
        if not self.json_mode:
            self._live = Live(console=output.console, auto_refresh=False, transient=False)
            self._live.start()
        return self

    def __call__(self, answer: str, turns: int) -> None:
        if self.json_mode:
            print(json.dumps({"turns": turns, "output": answer}, default=str), flush=True)
        elif self._live is not None:
            title = f"scribe · {turns} turn{'s' if turns != 1 else ''}"
            self._live.update(
                Panel(escape(answer or "…"), title=title, border_style="aai.brand"),
                refresh=True,
            )

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


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

    def follow_body(state: AppState, json_mode: bool) -> None:
        if not prompt:
            raise UsageError("Provide a prompt to run over the streamed transcript.")
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

        with _FollowRenderer(json_mode=json_mode) as render:
            transcript: list[str] = []
            for turn in stdio.iter_piped_stdin_lines():
                transcript.append(turn)
                render(ask("\n".join(transcript)), len(transcript))

    def body(state: AppState, json_mode: bool) -> None:
        if not prompt:
            raise UsageError("Provide a prompt, or use --list-models.")
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
        output.emit(
            {
                "model": model,
                "output": gateway.content_of(response),
                "usage": gateway.usage_of(response),
            },
            lambda d: escape(str(d["output"])),
            json_mode=json_mode,
        )

    run_command(ctx, follow_body if follow else body, json=json_out)
