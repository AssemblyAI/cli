from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from assemblyai.streaming.v3 import Encoding, NoiseSuppressionModel, SpeechModel

from aai_cli import (
    choices,
    client,
    code_gen,
    config,
    config_builder,
    help_panels,
    llm,
    output,
    youtube,
)
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.follow import FollowRenderer
from aai_cli.help_text import examples_epilog
from aai_cli.microphone import MicrophoneSource
from aai_cli.streaming.macos import MacSystemAudioSource
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.session import SourceOptions, StreamSession, validate_sources
from aai_cli.streaming.sources import TARGET_RATE, FileSource, StdinSource

app = typer.Typer()

DEFAULT_SPEECH_MODEL = SpeechModel.u3_rt_pro


def _dispatch(session: StreamSession, opts: SourceOptions) -> None:
    """Open the right audio source(s) for the flags and stream them."""
    if opts.from_system_audio:
        system = MacSystemAudioSource(on_open=session.on_open)
        if opts.system_audio_only:
            session.run(system, system.sample_rate, source_label="system")
        else:
            mic = MicrophoneSource(
                target_rate=TARGET_RATE,
                device=opts.device,
                capture_rate=opts.sample_rate,
                on_open=session.on_open,
            )
            session.run_parallel(
                [("system", system, system.sample_rate), ("you", mic, mic.sample_rate)]
            )
    elif opts.from_stdin:
        # Raw PCM16 mono piped on stdin (e.g. `ffmpeg … -f s16le - | aai stream -`).
        stdin_src = StdinSource(sample_rate=opts.sample_rate or TARGET_RATE)
        session.run(stdin_src, stdin_src.sample_rate)
    elif opts.source and youtube.is_youtube_url(opts.source):
        # Fetch the audio first, then stream the local file in real time.
        with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
            local = youtube.download_audio(opts.source, Path(td))
            session.run(FileSource(str(local)), TARGET_RATE)
    elif opts.from_file:
        file_audio = FileSource(client.resolve_audio_source(opts.source, sample=opts.sample))
        session.run(file_audio, file_audio.sample_rate)
    else:
        # Capture at the device's native rate (or --sample-rate override) and tell the
        # streaming API that rate, rather than forcing one the device may reject.
        # "Listening…" is announced once the device is open (see StreamSession.on_open),
        # not when the session opens — so early speech isn't lost in the gap.
        mic = MicrophoneSource(
            device=opts.device, capture_rate=opts.sample_rate, on_open=session.on_open
        )
        session.run(mic, mic.sample_rate)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Stream from your microphone", "aai stream"),
            ("Stream a file or URL in real time", "aai stream recording.wav"),
            ("Stream the hosted sample", "aai stream --sample"),
            ("Label speakers in the live transcript", "aai stream --speaker-labels"),
            (
                "Boost domain terms with keyterm prompts",
                'aai stream --keyterms-prompt "AssemblyAI" --keyterms-prompt "Claude"',
            ),
            (
                "Summarize action items live as you talk",
                'aai stream --llm "summarize action items"',
            ),
            ("Print equivalent Python instead of running", "aai stream --show-code"),
        ]
    ),
)
def stream(
    ctx: typer.Context,
    source: str | None = typer.Argument(
        None,
        help="Audio file path, URL, or YouTube URL to stream. Use - for raw PCM16/mono/16k "
        "on stdin. Omit to use the microphone.",
    ),
    sample: bool = typer.Option(False, "--sample", help="Stream the hosted wildfires.mp3 sample."),
    # audio capture
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        help="Force a microphone capture rate in Hz (default: device native).",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    device: int | None = typer.Option(
        None, "--device", help="Microphone device index.", rich_help_panel=help_panels.OPT_CAPTURE
    ),
    system_audio: bool = typer.Option(
        False,
        "--system-audio",
        help="macOS only: stream system/app audio and microphone as separate sessions.",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    system_audio_only: bool = typer.Option(
        False,
        "--system-audio-only",
        help="macOS only: stream system/app audio without the microphone.",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    # model & input
    speech_model: SpeechModel = typer.Option(
        DEFAULT_SPEECH_MODEL,
        "--speech-model",
        help="Streaming speech model.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    encoding: Encoding | None = typer.Option(
        None,
        "--encoding",
        help="Audio encoding.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    language_detection: bool | None = typer.Option(
        None,
        "--language-detection",
        help="Auto-detect the spoken language.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help="Domain preset (e.g. medical).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Prompt to bias the speech model (u3-pro).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    keyterms_prompt: list[str] | None = typer.Option(
        None,
        "--keyterms-prompt",
        help="Boost a key term (repeatable).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    # turn detection
    end_of_turn_confidence_threshold: float | None = typer.Option(
        None,
        "--end-of-turn-confidence-threshold",
        help="End-of-turn confidence (0-1).",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    min_turn_silence: int | None = typer.Option(
        None,
        "--min-turn-silence",
        help="Min silence to end a turn (ms).",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    max_turn_silence: int | None = typer.Option(
        None,
        "--max-turn-silence",
        help="Max silence before ending a turn (ms).",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    vad_threshold: float | None = typer.Option(
        None,
        "--vad-threshold",
        help="Voice-activity threshold (0-1).",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    format_turns: bool | None = typer.Option(
        None,
        "--format-turns/--no-format-turns",
        help="Punctuate/format finalized turns.",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    include_partial_turns: bool | None = typer.Option(
        None,
        "--include-partial-turns",
        help="Emit partial turns.",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    # speakers
    speaker_labels: bool | None = typer.Option(
        None,
        "--speaker-labels",
        help='Diarize speakers. With system audio the mic stays "You"; only the system '
        "audio is split into speakers.",
        rich_help_panel=help_panels.OPT_SPEAKERS,
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        help="Max speakers.",
        min=1,
        rich_help_panel=help_panels.OPT_SPEAKERS,
    ),
    # features
    voice_focus: NoiseSuppressionModel | None = typer.Option(
        None,
        "--voice-focus",
        help="Voice focus (noise suppression model).",
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    voice_focus_threshold: float | None = typer.Option(
        None,
        "--voice-focus-threshold",
        help="Voice-focus threshold (0-1).",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    inactivity_timeout: int | None = typer.Option(
        None,
        "--inactivity-timeout",
        help="Auto-close after N seconds idle.",
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    # guardrails
    filter_profanity: bool | None = typer.Option(
        None,
        "--filter-profanity",
        help="Mask profanity.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii: bool | None = typer.Option(
        None,
        "--redact-pii",
        help="Redact PII from turns.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_policy: str | None = typer.Option(
        None,
        "--redact-pii-policy",
        help="Comma-separated PII policies.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_sub: str | None = typer.Option(
        None,
        "--redact-pii-sub",
        help="Replace redacted PII with: hash or entity_name.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    # webhooks
    webhook_url: str | None = typer.Option(
        None, "--webhook-url", help="Webhook URL.", rich_help_panel=help_panels.OPT_WEBHOOKS
    ),
    webhook_auth_header: str | None = typer.Option(
        None,
        "--webhook-auth-header",
        help="Webhook auth header as NAME:VALUE.",
        rich_help_panel=help_panels.OPT_WEBHOOKS,
        metavar="NAME:VALUE",
    ),
    # llm transform
    llm_prompt: list[str] | None = typer.Option(
        None,
        "--llm",
        help="Run a prompt over the live transcript through LLM Gateway, refreshing the "
        "answer on every finalized turn. Repeatable: each prompt runs on the previous "
        "one's response (a chain).",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    llm_interval: float = typer.Option(
        30.0,
        "--llm-interval",
        help="Seconds between --llm summary refreshes (0 refreshes on every turn).",
        min=0.0,
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model.",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    # escape hatch
    config_kv: list[str] | None = typer.Option(
        None,
        "--config",
        help="Set any StreamingParameters field as KEY=VALUE (repeatable).",
        rich_help_panel=help_panels.OPT_ADVANCED,
        metavar="KEY=VALUE",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="JSON file of streaming fields.",
        rich_help_panel=help_panels.OPT_ADVANCED,
        exists=True,
        dir_okay=False,
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: text (finalized turns as plain lines, pipe-friendly) or json.",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not stream).",
    ),
) -> None:
    """Transcribe live audio in real time — from your mic, a file, a URL, or a pipe.

    Pass - as the source to read raw PCM16/mono/16k audio on stdin, e.g.
    ffmpeg -i input.mp4 -f s16le -ar 16000 -ac 1 - | aai stream -. --prompt biases the
    speech model. --llm runs a prompt over the live transcript in-process, refreshing the
    answer on every finalized turn; for a separate step instead, pipe the text out with
    -o text | aai llm -f "…".
    """

    def body(state: AppState, json_mode: bool) -> None:
        text_mode, json_mode = output.stream_output_modes(output_field, json_mode=json_mode)
        opts = SourceOptions(
            source=source,
            sample=sample,
            sample_rate=sample_rate,
            device=device,
            system_audio=system_audio,
            system_audio_only=system_audio_only,
        )
        # Every streaming flag except sample_rate, which is set per source at stream time.
        base_flags: dict[str, object] = {
            "speech_model": config_builder.enum_value(speech_model),
            "format_turns": format_turns if format_turns is not None else True,
            "encoding": config_builder.enum_value(encoding),
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
            "voice_focus": config_builder.enum_value(voice_focus),
            "voice_focus_threshold": voice_focus_threshold,
            "redact_pii": redact_pii,
            "redact_pii_policies": config_builder.split_csv(redact_pii_policy),
            "redact_pii_sub": redact_pii_sub,
            "inactivity_timeout": inactivity_timeout,
            "webhook_url": webhook_url,
            "prompt": prompt,
        }
        base_flags.update(config_builder.auth_header_flags(webhook_auth_header))

        if show_code:
            # Print-only: emit the canonical microphone-streaming script (16 kHz) from
            # the flags and exit without opening audio or authenticating. Raw stdout so
            # `--show-code > script.py` yields a runnable file.
            if opts.from_system_audio:
                raise UsageError("--show-code does not support macOS system audio capture yet.")
            merged = config_builder.merge_streaming_params(
                flags=base_flags | {"sample_rate": TARGET_RATE},
                overrides=config_kv,
                config_file=config_file,
            )
            gateway = code_gen.gateway_options(
                list(llm_prompt or []), model, max_tokens, interval=llm_interval
            )
            output.print_code(code_gen.stream(merged, llm=gateway))
            return

        api_key = config.resolve_api_key(profile=state.profile)
        validate_sources(opts, has_llm=bool(llm_prompt), text_mode=text_mode)

        llm_prompts = list(llm_prompt or [])
        session = StreamSession(
            api_key=api_key,
            base_flags=base_flags,
            overrides=config_kv,
            config_file=config_file,
            renderer=StreamRenderer(json_mode=json_mode, text_mode=text_mode),
            follow=FollowRenderer(json_mode=json_mode) if llm_prompts else None,
            llm_prompts=llm_prompts,
            model=model,
            max_tokens=max_tokens,
            llm_interval=llm_interval,
        )
        _dispatch(session, opts)

    run_command(ctx, body, json=json_out)
