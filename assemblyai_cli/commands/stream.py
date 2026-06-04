from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from assemblyai.streaming.v3 import SpeechModel

from assemblyai_cli import client, code_gen, config, config_builder, llm, youtube
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError
from assemblyai_cli.microphone import MicrophoneSource
from assemblyai_cli.streaming.render import StreamRenderer
from assemblyai_cli.streaming.sources import TARGET_RATE, FileSource, StdinSource

app = typer.Typer()

DEFAULT_SPEECH_MODEL = SpeechModel.universal_streaming_multilingual.value


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
    # model & input
    speech_model: str = typer.Option(
        DEFAULT_SPEECH_MODEL, "--speech-model", help="Streaming speech model."
    ),
    encoding: str = typer.Option(None, "--encoding", help="pcm_s16le or pcm_mulaw."),
    language_detection: bool = typer.Option(
        None, "--language-detection", help="Auto-detect the spoken language."
    ),
    domain: str = typer.Option(None, "--domain", help="Domain preset (e.g. medical)."),
    # turn detection
    end_of_turn_confidence_threshold: float = typer.Option(
        None, "--end-of-turn-confidence-threshold", help="0-1 end-of-turn confidence."
    ),
    min_turn_silence: int = typer.Option(None, "--min-turn-silence", help="Min turn silence (ms)."),
    max_turn_silence: int = typer.Option(None, "--max-turn-silence", help="Max turn silence (ms)."),
    vad_threshold: float = typer.Option(None, "--vad-threshold", help="Voice-activity threshold."),
    format_turns: bool = typer.Option(
        None, "--format-turns/--no-format-turns", help="Punctuate/format finalized turns."
    ),
    include_partial_turns: bool = typer.Option(
        None, "--include-partial-turns", help="Emit partial turns."
    ),
    # features
    keyterms_prompt: list[str] = typer.Option(
        None, "--keyterms-prompt", help="Boost a key term (repeatable)."
    ),
    filter_profanity: bool = typer.Option(None, "--filter-profanity", help="Mask profanity."),
    speaker_labels: bool = typer.Option(None, "--speaker-labels", help="Label speakers."),
    max_speakers: int = typer.Option(None, "--max-speakers", help="Max speakers."),
    voice_focus: str = typer.Option(None, "--voice-focus", help="near_field or far_field."),
    voice_focus_threshold: float = typer.Option(
        None, "--voice-focus-threshold", help="Voice-focus threshold."
    ),
    redact_pii: bool = typer.Option(None, "--redact-pii", help="Redact PII from turns."),
    redact_pii_policy: str = typer.Option(
        None, "--redact-pii-policy", help="Comma-separated PII policies."
    ),
    redact_pii_sub: str = typer.Option(None, "--redact-pii-sub", help="hash or entity_name."),
    inactivity_timeout: int = typer.Option(
        None, "--inactivity-timeout", help="Auto-close after N seconds idle."
    ),
    webhook_url: str = typer.Option(None, "--webhook-url", help="Webhook URL."),
    webhook_auth_header: str = typer.Option(
        None, "--webhook-auth-header", help="Webhook auth header as NAME:VALUE."
    ),
    # escape hatch
    config_kv: list[str] = typer.Option(
        None, "--config", help="Set any StreamingParameters field as KEY=VALUE (repeatable)."
    ),
    config_file: str = typer.Option(None, "--config-file", help="JSON file of streaming fields."),
    # existing
    prompt: str = typer.Option(None, "--prompt", help="Bias the speech model (u3-pro)."),
    llm_gateway_prompt: str = typer.Option(
        None,
        "--llm-gateway-prompt",
        help="After streaming, transform the full transcript through LLM Gateway.",
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    max_tokens: int = typer.Option(llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not stream).",
    ),
) -> None:
    """Transcribe live audio in real time with the full StreamingParameters surface.

    --prompt biases the speech model. --llm-gateway-prompt transforms the full
    transcript through LLM Gateway once the stream ends (e.g. "summarize the call").
    """

    def body(state: AppState, json_mode: bool) -> None:
        def make_flags(rate: int) -> dict[str, object]:
            flags: dict[str, object] = {
                "sample_rate": rate,
                "speech_model": speech_model,
                "format_turns": format_turns if format_turns is not None else True,
                "encoding": encoding,
                "language_detection": language_detection,
                "domain": domain,
                "end_of_turn_confidence_threshold": end_of_turn_confidence_threshold,
                "min_turn_silence": min_turn_silence,
                "max_turn_silence": max_turn_silence,
                "vad_threshold": vad_threshold,
                "include_partial_turns": include_partial_turns,
                "keyterms_prompt": list(keyterms_prompt) if keyterms_prompt else None,
                "filter_profanity": filter_profanity,
                "speaker_labels": speaker_labels,
                "max_speakers": max_speakers,
                "voice_focus": voice_focus,
                "voice_focus_threshold": voice_focus_threshold,
                "redact_pii": redact_pii,
                "redact_pii_policies": config_builder.split_csv(redact_pii_policy),
                "redact_pii_sub": redact_pii_sub,
                "inactivity_timeout": inactivity_timeout,
                "webhook_url": webhook_url,
                "prompt": prompt,
            }
            header = config_builder.parse_auth_header(webhook_auth_header)
            if header is not None:
                flags["webhook_auth_header_name"] = header[0]
                flags["webhook_auth_header_value"] = header[1]
            return flags

        if show_code:
            # Print-only: emit the canonical microphone-streaming script (16 kHz) from
            # the flags and exit without opening audio or authenticating. Raw stdout so
            # `--show-code > script.py` yields a runnable file.
            merged = config_builder.merge_streaming_params(
                flags=make_flags(TARGET_RATE),
                overrides=list(config_kv or []),
                config_file=config_file,
            )
            print(code_gen.stream(merged))
            return

        api_key = config.resolve_api_key(profile=state.profile)
        from_stdin = source == "-"
        from_file = bool(source) or sample
        if from_stdin:
            if device is not None:
                raise UsageError("--device applies only to microphone input.")
        elif from_file and (sample_rate is not None or device is not None):
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

        def run(audio: FileSource | MicrophoneSource | StdinSource, rate: int) -> None:
            merged = config_builder.merge_streaming_params(
                flags=make_flags(rate), overrides=list(config_kv or []), config_file=config_file
            )
            params = config_builder.construct_streaming_params(merged)

            try:
                client.stream_audio(
                    api_key,
                    audio,
                    params=params,
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

        if from_stdin:
            # Raw PCM16 mono piped on stdin (e.g. `ffmpeg … -f s16le - | aai stream -`).
            stdin_src = StdinSource(sample_rate=sample_rate or TARGET_RATE)
            run(stdin_src, stdin_src.sample_rate)
        elif source and youtube.is_youtube_url(source):
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
