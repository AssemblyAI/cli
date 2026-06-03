from __future__ import annotations

import typer

from assemblyai_cli import client, config
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError
from assemblyai_cli.streaming.render import StreamRenderer
from assemblyai_cli.streaming.sources import TARGET_RATE, FileSource, MicSource

app = typer.Typer()


@app.command()
def stream(
    ctx: typer.Context,
    source: str = typer.Argument(None, help="Audio file to stream. Omit to use the microphone."),
    sample_rate: int = typer.Option(
        TARGET_RATE, "--sample-rate", help="Microphone sample rate in Hz."
    ),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
) -> None:
    """Transcribe live audio from the microphone or a file in real time."""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        if source and (sample_rate != TARGET_RATE or device is not None):
            raise UsageError("--sample-rate and --device apply only to microphone input.")
        audio: FileSource | MicSource
        if source:
            audio = FileSource(source)
            rate = audio.sample_rate
        else:
            audio = MicSource(sample_rate=sample_rate, device=device)
            rate = sample_rate
        renderer = StreamRenderer(json_mode=json_mode)
        try:
            client.stream_audio(
                api_key,
                audio,
                sample_rate=rate,
                on_begin=renderer.begin,
                on_turn=renderer.turn,
                on_termination=renderer.termination,
            )
        except KeyboardInterrupt:
            # Ctrl-C is a normal "user stopped" signal -> exit 0.
            renderer.close()
            if not json_mode:
                renderer.out.write("Stopped.\n")
                renderer.out.flush()
        except BrokenPipeError:
            # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
            raise typer.Exit(code=0) from None
        finally:
            renderer.close()

    run_command(ctx, body, json=json_out)
