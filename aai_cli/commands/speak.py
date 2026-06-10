from __future__ import annotations

import sys
from pathlib import Path

import typer

from aai_cli import config, help_panels, options, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError, UsageError
from aai_cli.help_text import examples_epilog
from aai_cli.tts import audio, dialogue, session

app = typer.Typer()

# The streaming-TTS reference client defaults to the PocketTTS "jane" voice and
# English, so the CLI sends the same and a bare `aai speak` works out of the box.
# Override either with --voice/--language.
DEFAULT_VOICE = "jane"
DEFAULT_LANGUAGE = "English"


def _read_text(text: str | None) -> str:
    """The text to speak: the argument if non-blank, else stdin when piped."""
    if text is not None and text.strip():
        return text
    if text is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise UsageError(
        "No text to speak.",
        suggestion='Pass text as an argument: aai speak "Hello" — or pipe it via stdin.',
    )


def _output_audio(result: session.SpeakResult, out: Path | None) -> None:
    """Write a WAV when --out is given, else play through the speakers."""
    if out is not None:
        audio.write_wav(out, result.pcm, result.sample_rate)
    else:
        audio.play_pcm(result.pcm, result.sample_rate)


def _disposition(out: Path | None) -> str:
    return f"saved to {out}" if out is not None else "played"


def _emit_single(
    result: session.SpeakResult,
    cfg: session.SpeakConfig,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    """Single-voice result: a JSON object on stdout, or a human note on stderr."""
    duration = round(result.audio_duration_seconds, 3)
    if json_mode:
        output.emit_ndjson(
            {
                "voice": cfg.voice,
                "language": cfg.language,
                "sample_rate": result.sample_rate,
                "audio_duration_seconds": duration,
                "bytes": len(result.pcm),
                "out": str(out) if out is not None else None,
            }
        )
        return
    output.error_console.print(
        f"[aai.muted]Spoke {duration}s of audio ({_disposition(out)}).[/aai.muted]"
    )


def _emit_multi(
    result: session.SpeakResult,
    speakers: dict[str, str],
    segment_count: int,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    """Multi-voice result: a JSON object on stdout, or a human note on stderr."""
    duration = round(result.audio_duration_seconds, 3)
    if json_mode:
        output.emit_ndjson(
            {
                "mode": "multi",
                "speakers": speakers,
                "segments": segment_count,
                "sample_rate": result.sample_rate,
                "audio_duration_seconds": duration,
                "bytes": len(result.pcm),
                "out": str(out) if out is not None else None,
            }
        )
        return
    voices = ", ".join(f"{spk}={voice}" for spk, voice in speakers.items())
    output.error_console.print(
        f"[aai.muted]Spoke {duration}s across {len(speakers)} voices "
        f"({voices}) ({_disposition(out)}).[/aai.muted]"
    )


def _speak_single(
    api_key: str,
    text: str,
    voice: str,
    language: str,
    sample_rate: int | None,
    out: Path | None,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    cfg = session.SpeakConfig(text=text, voice=voice, language=language, sample_rate=sample_rate)
    with output.status("Synthesizing speech…", json_mode=json_mode, quiet=quiet):
        result = session.synthesize(
            api_key, cfg, on_warning=lambda m: output.emit_warning(m, json_mode=json_mode)
        )
    _output_audio(result, out)
    _emit_single(result, cfg, out, json_mode=json_mode)


def _speak_dialogue(
    api_key: str,
    text: str,
    bare_voice: str | None,
    overrides: dict[str, str],
    language: str,
    sample_rate: int | None,
    out: Path | None,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    segments = dialogue.parse_segments(text)
    if not segments:
        raise UsageError(
            "No text to speak.",
            suggestion="The input had speaker labels but no spoken text.",
        )
    if bare_voice is not None:
        output.emit_warning(
            "Ignoring bare --voice in multi-speaker mode; "
            "set a voice per speaker with --voice A=NAME.",
            json_mode=json_mode,
        )
    resolved, speakers = dialogue.assign_voices(
        segments, dialogue.DEFAULT_VOICE_ROTATION, overrides
    )
    with output.status("Synthesizing speech…", json_mode=json_mode, quiet=quiet):
        result = session.synthesize_dialogue(
            api_key,
            resolved,
            language=language,
            sample_rate=sample_rate,
            on_warning=lambda m: output.emit_warning(m, json_mode=json_mode),
        )
    _output_audio(result, out)
    _emit_multi(result, speakers, len(resolved), out, json_mode=json_mode)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Speak text aloud (sandbox only)", 'aai speak "Hello there, friend." --sandbox'),
            (
                "Pick a voice and language",
                'aai speak "Bonjour" --voice jane --language French --sandbox',
            ),
            (
                "Speak a diarized transcript, one voice per speaker",
                "aai transcribe meeting.mp3 --speaker-labels | aai speak --sandbox",
            ),
            (
                "Override a speaker's voice",
                "… | aai speak --voice A=vera --voice B=paul --sandbox",
            ),
            (
                "Save to a WAV instead of playing",
                'aai speak "Hello" --out /tmp/hello.wav --sandbox',
            ),
        ]
    ),
)
def speak(
    ctx: typer.Context,
    text: str | None = typer.Argument(None, help="Text to speak. Omit to read from stdin."),
    voice: list[str] = typer.Option(
        [],
        "--voice",
        help="Voice id, or SPEAKER=VOICE for diarized input (repeatable, e.g. --voice A=jane).",
    ),
    language: str = typer.Option(DEFAULT_LANGUAGE, "--language", help="Language of the text."),
    sample_rate: int | None = typer.Option(
        None, "--sample-rate", help="Output sample rate in Hz. Server default if omitted."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Write a WAV file instead of playing through the speakers."
    ),
    json_out: bool = options.json_option("Emit JSON metadata about the synthesized audio."),
) -> None:
    """Synthesize speech from text with AssemblyAI streaming TTS (sandbox only).

    Plays the audio through your speakers by default, or writes a WAV with --out.
    Speaker-labeled input (from 'aai transcribe --speaker-labels') is detected
    automatically: the labels are stripped and each speaker gets a different
    voice. This feature only exists in the sandbox today — run it with --sandbox.
    """

    def body(state: AppState, json_mode: bool) -> None:
        if not session.is_available():
            raise CLIError(
                "aai speak is only available in the sandbox.",
                error_type="unsupported_environment",
                exit_code=2,
                suggestion="Re-run with --sandbox (or --env sandbox000).",
            )
        spoken = _read_text(text)
        api_key = config.resolve_api_key(profile=state.profile)
        bare_voice, overrides = dialogue.parse_voice_overrides(voice)
        if dialogue.looks_like_speaker_labeled(spoken):
            _speak_dialogue(
                api_key,
                spoken,
                bare_voice,
                overrides,
                language,
                sample_rate,
                out,
                json_mode=json_mode,
                quiet=state.quiet,
            )
        else:
            _speak_single(
                api_key,
                spoken,
                bare_voice or DEFAULT_VOICE,
                language,
                sample_rate,
                out,
                json_mode=json_mode,
                quiet=state.quiet,
            )

    run_command(ctx, body, json=json_out)
