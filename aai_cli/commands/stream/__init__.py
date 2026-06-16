from __future__ import annotations

from pathlib import Path

import typer
from assemblyai import PIISubstitutionPolicy
from assemblyai.streaming.v3 import Encoding, NoiseSuppressionModel, SpeechModel

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.stream import _exec as stream_exec
from aai_cli.core import choices, llm
from aai_cli.streaming.turn_presets import TurnDetectionPreset
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=20,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("stream",),
)

DEFAULT_SPEECH_MODEL = SpeechModel.u3_rt_pro


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Stream from your microphone", "assembly stream"),
            ("Stream a file or URL in real time", "assembly stream recording.wav"),
            ("Stream a list of files in turn", "ls *.wav | assembly stream --from-stdin"),
            ("Stream the hosted sample", "assembly stream --sample"),
            ("Label speakers in the live transcript", "assembly stream --speaker-labels"),
            ("Save a WAV of the audio while streaming", "assembly stream --save-audio out.wav"),
            ("Save the transcript text to a file", "assembly stream --save-transcript notes.txt"),
            (
                "Auto-name the transcript + WAV under a dir",
                'assembly stream --save-dir ~/recordings --name "Standup"',
            ),
            (
                "Name from content + save a summary note",
                'assembly stream --save-dir ~/recordings --auto-name --llm "summarize as a note"',
            ),
            (
                "Boost domain terms with keyterm prompts",
                'assembly stream --keyterms-prompt "AssemblyAI" --keyterms-prompt "Claude"',
            ),
            (
                "Summarize action items live as you talk",
                'assembly stream --llm "summarize action items"',
            ),
            ("Print equivalent Python instead of running", "assembly stream --show-code"),
        ]
    ),
)
def stream(
    ctx: typer.Context,
    source: str | None = typer.Argument(
        None,
        help="Audio file path, URL, or YouTube/podcast page URL to stream. Use - for raw "
        "PCM16/mono/16k on stdin. Omit to use the microphone.",
    ),
    sample: bool = typer.Option(False, "--sample", help="Stream the hosted wildfires.mp3 sample"),
    from_stdin: bool = typer.Option(
        False,
        "--from-stdin",
        help="Read a list of audio files/URLs on stdin (one per line) and stream each in turn",
    ),
    # audio capture
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        help="Audio rate in Hz (positive): capture rate for the mic, or the declared "
        "rate of raw PCM on stdin (default: device native / 16000)",
        min=1,
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    device: int | None = typer.Option(
        None, "--device", help="Microphone device index", rich_help_panel=help_panels.OPT_CAPTURE
    ),
    system_audio: bool = typer.Option(
        False,
        "--system-audio",
        help="macOS only: stream system/app audio and microphone as separate sessions",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    system_audio_only: bool = typer.Option(
        False,
        "--system-audio-only",
        help="macOS only: stream system/app audio without the microphone",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    # saving
    save_audio: Path | None = typer.Option(
        None,
        "--save-audio",
        help="Tee the streamed PCM to PATH as a 16-bit mono WAV while transcribing",
        rich_help_panel=help_panels.OPT_SAVING,
        dir_okay=False,
        # Click guardrail; flipping it changes no behavior a unit test can observe
        # (and the writable check is a no-op under the test runner's root uid).
        writable=True,  # pragma: no mutate
    ),
    save_transcript: Path | None = typer.Option(
        None,
        "--save-transcript",
        help="Write the finalized transcript to PATH, one turn per line",
        rich_help_panel=help_panels.OPT_SAVING,
        dir_okay=False,
        writable=True,  # pragma: no mutate
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Auto-name the transcript and a matching WAV under DIR/YYYY-MM-DD/ "
        "with a timestamped file; --system-audio saves one WAV per channel",
        rich_help_panel=help_panels.OPT_SAVING,
        file_okay=False,
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Title to slug into the --save-dir filename (e.g. a meeting title)",
        rich_help_panel=help_panels.OPT_SAVING,
    ),
    auto_name: bool = typer.Option(
        False,
        "--auto-name",
        help="With --save-dir, derive the filename from the transcript via the LLM",
        rich_help_panel=help_panels.OPT_SAVING,
    ),
    no_save_audio: bool = typer.Option(
        False,
        "--no-save-audio",
        help="With --save-dir, skip the WAV and save only the transcript",
        rich_help_panel=help_panels.OPT_SAVING,
    ),
    # model & input
    speech_model: SpeechModel = typer.Option(
        DEFAULT_SPEECH_MODEL,
        "--speech-model",
        help="Streaming speech model",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    encoding: Encoding | None = typer.Option(
        None,
        "--encoding",
        help="Audio encoding",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    language_detection: bool | None = typer.Option(
        None,
        "--language-detection",
        help="Auto-detect the spoken language",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help="Domain preset (e.g. medical)",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Prompt to bias the speech model (supported models only)",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    keyterms_prompt: list[str] | None = typer.Option(
        None,
        "--keyterms-prompt",
        help="Boost a key term (repeatable)",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    # turn detection
    turn_detection: TurnDetectionPreset | None = typer.Option(
        None,
        "--turn-detection",
        help="Turn-detection sensitivity preset",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    end_of_turn_confidence_threshold: float | None = typer.Option(
        None,
        # Not "--end-of-turn-confidence-threshold": at 34 chars the name can't render
        # un-clipped in an 80-column help screen, which made it unlearnable from --help.
        # The full SDK field stays reachable via --config.
        "--end-of-turn-confidence",
        help="End-of-turn confidence (0-1)",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    min_turn_silence: int | None = typer.Option(
        None,
        "--min-turn-silence",
        help="Min silence to end a turn (ms)",
        min=1,
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    max_turn_silence: int | None = typer.Option(
        None,
        "--max-turn-silence",
        help="Max silence before ending a turn (ms)",
        min=1,
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    vad_threshold: float | None = typer.Option(
        None,
        "--vad-threshold",
        help="Voice activity threshold (0-1)",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    format_turns: bool | None = typer.Option(
        None,
        "--format-turns/--no-format-turns",
        help="Format (punctuate) finalized turns",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    include_partial_turns: bool | None = typer.Option(
        None,
        "--include-partial-turns",
        help="Emit partial turns",
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
        help="Max speakers",
        min=1,
        rich_help_panel=help_panels.OPT_SPEAKERS,
    ),
    # features
    voice_focus: NoiseSuppressionModel | None = typer.Option(
        None,
        "--voice-focus",
        help="Voice focus (noise suppression model)",
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    voice_focus_threshold: float | None = typer.Option(
        None,
        "--voice-focus-threshold",
        help="Voice-focus threshold (0-1)",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    inactivity_timeout: int | None = typer.Option(
        None,
        "--inactivity-timeout",
        help="Auto-close after N seconds idle",
        min=1,
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    # guardrails
    filter_profanity: bool | None = typer.Option(
        None,
        "--filter-profanity",
        help="Mask profanity",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii: bool | None = typer.Option(
        None,
        "--redact-pii",
        help="Redact PII from turns",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_policy: str | None = typer.Option(
        None,
        "--redact-pii-policy",
        help="Comma-separated PII policies",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_sub: PIISubstitutionPolicy | None = typer.Option(
        None,
        "--redact-pii-sub",
        help="Replace redacted PII with: hash or entity_name",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    # webhooks
    webhook_url: str | None = typer.Option(
        None, "--webhook-url", help="Webhook URL", rich_help_panel=help_panels.OPT_WEBHOOKS
    ),
    webhook_auth_header: str | None = typer.Option(
        None,
        "--webhook-auth-header",
        help="Webhook auth header as NAME:VALUE",
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
        10.0,
        "--llm-interval",
        help="Seconds between --llm summary refreshes (0 refreshes on every turn)",
        min=0.0,
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens",
        min=1,
        rich_help_panel=help_panels.OPT_LLM,
    ),
    # escape hatch
    config_kv: list[str] | None = typer.Option(
        None,
        "--config",
        help="Set any StreamingParameters field as KEY=VALUE (repeatable)",
        rich_help_panel=help_panels.OPT_ADVANCED,
        metavar="KEY=VALUE",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="JSON file of streaming fields",
        rich_help_panel=help_panels.OPT_ADVANCED,
        exists=True,
        dir_okay=False,
    ),
    json_out: bool = options.json_option("Emit newline-delimited JSON events"),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: text (finalized turns as plain lines, pipe-friendly) or json",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not stream)",
    ),
) -> None:
    """Transcribe live audio in real time from a mic, file, URL, or pipe

    Pass - as the source to read raw PCM16/mono/16k audio on stdin, e.g.
    ffmpeg -i input.mp4 -f s16le -ar 16000 -ac 1 - | assembly stream -.

    --from-stdin instead reads a list of file paths/URLs on stdin (one per line)
    and streams each as its own realtime session, in turn.

    --prompt biases the speech model. --llm runs a prompt over the live transcript
    in-process, refreshing the answer on every finalized turn; for a separate step
    instead, pipe the text out with -o text | assembly llm -f "…".
    """
    opts = stream_exec.StreamOptions(
        source=source,
        sample=sample,
        from_stdin=from_stdin,
        sample_rate=sample_rate,
        device=device,
        system_audio=system_audio,
        system_audio_only=system_audio_only,
        speech_model=speech_model,
        encoding=encoding,
        language_detection=language_detection,
        domain=domain,
        prompt=prompt,
        keyterms_prompt=keyterms_prompt,
        end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
        min_turn_silence=min_turn_silence,
        max_turn_silence=max_turn_silence,
        turn_detection=turn_detection,
        vad_threshold=vad_threshold,
        format_turns=format_turns,
        include_partial_turns=include_partial_turns,
        speaker_labels=speaker_labels,
        max_speakers=max_speakers,
        voice_focus=voice_focus,
        voice_focus_threshold=voice_focus_threshold,
        inactivity_timeout=inactivity_timeout,
        filter_profanity=filter_profanity,
        redact_pii=redact_pii,
        redact_pii_policy=redact_pii_policy,
        redact_pii_sub=redact_pii_sub,
        webhook_url=webhook_url,
        webhook_auth_header=webhook_auth_header,
        llm_prompt=llm_prompt,
        llm_interval=llm_interval,
        model=model,
        max_tokens=max_tokens,
        config_kv=config_kv,
        config_file=config_file,
        output_field=output_field,
        show_code=show_code,
        save_audio=save_audio,
        save_transcript=save_transcript,
        save_dir=save_dir,
        name=name,
        auto_name=auto_name,
        no_save_audio=no_save_audio,
    )
    run_with_options(ctx, stream_exec.run_stream, opts, json=json_out)
