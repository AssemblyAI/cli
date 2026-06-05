from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from assemblyai.streaming.v3 import SpeechModel

from aai_cli import client, code_gen, config, config_builder, llm, output, youtube
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.follow import FollowRenderer
from aai_cli.help_text import examples_epilog
from aai_cli.microphone import MicrophoneSource
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.sources import TARGET_RATE, FileSource, StdinSource

app = typer.Typer()

DEFAULT_SPEECH_MODEL = SpeechModel.u3_rt_pro.value


@app.command(
    epilog=examples_epilog(
        [
            ("Stream from your microphone", "aai stream"),
            ("Stream the hosted sample", "aai stream --sample"),
            (
                "Summarize action items live as you talk",
                'aai stream --llm "summarize action items"',
            ),
            ("Print equivalent Python instead of running", "aai stream --show-code"),
        ]
    )
)
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
    llm_prompt: list[str] = typer.Option(
        None,
        "--llm",
        help="Run a prompt over the live transcript through LLM Gateway, refreshing the "
        "answer on every finalized turn. Repeatable: each prompt runs on the previous "
        "one's response (a chain).",
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    max_tokens: int = typer.Option(llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
    output_field: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: 'text' (finalized turns as plain lines, pipe-friendly) or 'json'.",
    ),
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
        text_mode, json_mode = output.stream_output_modes(output_field, json_mode)

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
            flags.update(config_builder.auth_header_flags(webhook_auth_header))
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
            gateway = code_gen.gateway_options(llm_prompt, model, max_tokens)
            output.print_code(code_gen.stream(merged, llm=gateway))
            return

        api_key = config.resolve_api_key(profile=state.profile)
        from_stdin = source == "-"
        from_file = bool(source) or sample
        if from_stdin:
            if device is not None:
                raise UsageError("--device applies only to microphone input.")
        elif from_file and (sample_rate is not None or device is not None):
            raise UsageError("--sample-rate and --device apply only to microphone input.")

        if llm_prompt and text_mode:
            raise UsageError(
                "--llm renders a live panel (or NDJSON when piped); it can't be combined "
                "with -o text."
            )

        renderer = StreamRenderer(json_mode=json_mode, text_mode=text_mode)
        # In --llm mode the answer is rendered live by a FollowRenderer instead of the
        # raw turns; transcript accumulates the finalized turns we re-run the chain over.
        follow = FollowRenderer(json_mode=json_mode) if llm_prompt else None
        transcript: list[str] = []

        def on_turn(event: object) -> None:
            if follow is None:
                renderer.turn(event)
                return
            # Live LLM mode: re-run the prompt chain over the growing transcript on every
            # finalized turn, refreshing one evolving answer (partials are ignored).
            if not getattr(event, "end_of_turn", False):
                return
            text = getattr(event, "transcript", "") or ""
            if not text:
                return
            transcript.append(text)
            answer = llm.run_chain(
                api_key,
                list(llm_prompt),
                transcript_text=" ".join(transcript),
                model=model,
                max_tokens=max_tokens,
            )
            follow(answer, len(transcript))

        def run(audio: FileSource | MicrophoneSource | StdinSource, rate: int) -> None:
            merged = config_builder.merge_streaming_params(
                flags=make_flags(rate), overrides=list(config_kv or []), config_file=config_file
            )
            params = config_builder.construct_streaming_params(merged)

            def drive() -> None:
                try:
                    client.stream_audio(
                        api_key,
                        audio,
                        params=params,
                        on_begin=None if follow is not None else renderer.begin,
                        on_turn=on_turn,
                        on_termination=None if follow is not None else renderer.termination,
                    )
                except KeyboardInterrupt:
                    # Ctrl-C is a normal "user stopped" signal -> exit 0.
                    if follow is None:
                        renderer.close()
                        renderer.stopped()
                except BrokenPipeError:
                    # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
                    raise typer.Exit(code=0) from None
                finally:
                    if follow is None:
                        renderer.close()

            # The FollowRenderer is a context manager (it owns the live panel); enter it
            # around the whole session so it stops cleanly and prints the final answer.
            if follow is not None:
                with follow:
                    drive()
            else:
                drive()

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
                device=device,
                capture_rate=sample_rate,
                # In --llm mode the FollowRenderer owns the screen, so skip the notice.
                on_open=(lambda: None) if follow is not None else renderer.listening,
            )
            run(mic, mic.sample_rate)

    run_command(ctx, body, json=json_out)
