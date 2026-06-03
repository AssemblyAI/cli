from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

from assemblyai_cli import client, config, llm, youtube
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
        None,
        help="Audio file path, URL, or YouTube URL to stream. Omit to use the microphone.",
    ),
    sample: bool = typer.Option(False, "--sample", help="Stream the hosted wildfires.mp3 sample."),
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        help="Force a microphone capture rate in Hz (default: device native).",
    ),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    prompt: str = typer.Option(
        None, "--prompt", help="Bias the speech model with this prompt (u3-rt-pro)."
    ),
    llm_gateway_prompt: str = typer.Option(
        None,
        "--llm-gateway-prompt",
        help="After streaming, transform the full transcript through LLM Gateway.",
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL, "--model", help="LLM Gateway model for --llm-gateway-prompt."
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens for the LLM Gateway transform."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
) -> None:
    """Transcribe live audio from the microphone, a file, a URL, or YouTube in real time.

    --prompt biases the speech model. --llm-gateway-prompt transforms the full
    transcript through LLM Gateway once the stream ends (e.g. "summarize the call").
    """

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        from_file = bool(source) or sample
        if from_file and (sample_rate is not None or device is not None):
            raise UsageError("--sample-rate and --device apply only to microphone input.")

        renderer = StreamRenderer(json_mode=json_mode)
        # Collect finalized turns so we can transform the full transcript at the end.
        turns: list[str] = []

        def on_turn(event: object) -> None:
            renderer.turn(event)
            if llm_gateway_prompt and getattr(event, "end_of_turn", False):
                text = getattr(event, "transcript", "") or ""
                if text:
                    turns.append(text)

        def run(audio: FileSource | MicrophoneSource, rate: int) -> None:
            try:
                client.stream_audio(
                    api_key,
                    audio,
                    params=StreamingParameters(
                        sample_rate=rate,
                        format_turns=True,
                        speech_model=SpeechModel.universal_streaming_multilingual,
                        prompt=prompt,
                    ),
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

            if llm_gateway_prompt and turns:
                transformed = llm.transform_transcript(
                    api_key,
                    prompt=llm_gateway_prompt,
                    model=model,
                    transcript_text=" ".join(turns),
                    max_tokens=max_tokens,
                )
                renderer.llm(transformed)

        if source and youtube.is_youtube_url(source):
            # Fetch the audio first, then stream the local file in real time.
            with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
                local = youtube.download_audio(source, Path(td))
                run(FileSource(str(local)), TARGET_RATE)
        elif from_file:
            file_audio = FileSource(client.resolve_audio_source(source, sample=sample))
            run(file_audio, file_audio.sample_rate)
        else:
            # Capture at the device's native rate (or --sample-rate override) and tell
            # the streaming API that rate, rather than forcing one the device may reject.
            # Announce "Listening…" only once the device is open and recording,
            # not when the session opens — so early speech isn't lost in the gap.
            mic = MicrophoneSource(
                device=device, capture_rate=sample_rate, on_open=renderer.listening
            )
            run(mic, mic.sample_rate)

    run_command(ctx, body, json=json_out)
