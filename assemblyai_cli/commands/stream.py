from __future__ import annotations

import typer

from assemblyai_cli import client, config, llm
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError
from assemblyai_cli.microphone import MicrophoneSource
from assemblyai_cli.streaming.render import StreamRenderer
from assemblyai_cli.streaming.sources import TARGET_RATE, FileSource

app = typer.Typer()


@app.command()
def stream(
    ctx: typer.Context,
    source: str = typer.Argument(
        None, help="Audio file path or URL to stream. Omit to use the microphone."
    ),
    sample: bool = typer.Option(False, "--sample", help="Stream the hosted wildfires.mp3 sample."),
    sample_rate: int = typer.Option(
        TARGET_RATE, "--sample-rate", help="Microphone sample rate in Hz."
    ),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    prompt: str = typer.Option(
        None,
        "--prompt",
        help="After streaming, transform the full transcript through LLM Gateway.",
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model for --prompt."),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens for the --prompt transform."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
) -> None:
    """Transcribe live audio from the microphone, a file, or a URL in real time.

    Pass --prompt to transform the full transcript through LLM Gateway once the
    stream ends (e.g. --prompt "summarize the call").
    """

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        from_file = bool(source) or sample
        if from_file and (sample_rate != TARGET_RATE or device is not None):
            raise UsageError("--sample-rate and --device apply only to microphone input.")
        audio: FileSource | MicrophoneSource
        if from_file:
            audio = FileSource(client.resolve_audio_source(source, sample=sample))
            rate = audio.sample_rate
        else:
            audio = MicrophoneSource(sample_rate=sample_rate, device=device)
            rate = sample_rate
        renderer = StreamRenderer(json_mode=json_mode)
        # Collect finalized turns so we can transform the full transcript at the end.
        turns: list[str] = []

        def on_turn(event: object) -> None:
            renderer.turn(event)
            if prompt and getattr(event, "end_of_turn", False):
                text = getattr(event, "transcript", "") or ""
                if text:
                    turns.append(text)

        try:
            client.stream_audio(
                api_key,
                audio,
                sample_rate=rate,
                on_begin=renderer.begin,
                on_turn=on_turn,
                on_termination=renderer.termination,
            )
        except KeyboardInterrupt:
            # Ctrl-C is a normal "user stopped" signal -> exit 0 (still transform below).
            renderer.close()
            renderer.stopped()
        except BrokenPipeError:
            # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
            raise typer.Exit(code=0) from None
        finally:
            renderer.close()

        if prompt and turns:
            transformed = llm.transform_transcript(
                api_key,
                prompt=prompt,
                model=model,
                transcript_text=" ".join(turns),
                max_tokens=max_tokens,
            )
            renderer.llm(transformed)

    run_command(ctx, body, json=json_out)
