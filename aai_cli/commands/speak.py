from __future__ import annotations

import sys
from pathlib import Path

import typer

from aai_cli import config, help_panels, options, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError, UsageError
from aai_cli.help_text import examples_epilog
from aai_cli.tts import audio, session

app = typer.Typer()


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


def _emit_result(
    result: session.SpeakResult,
    config_: session.SpeakConfig,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    """Report what was produced: a JSON object on stdout, or a human note on stderr."""
    duration = round(result.audio_duration_seconds, 3)
    if json_mode:
        output.emit_ndjson(
            {
                "voice": config_.voice,
                "language": config_.language,
                "sample_rate": result.sample_rate,
                "audio_duration_seconds": duration,
                "bytes": len(result.pcm),
                "out": str(out) if out is not None else None,
            }
        )
        return
    where = f"saved to {out}" if out is not None else "played"
    output.error_console.print(f"[aai.muted]Spoke {duration}s of audio ({where}).[/aai.muted]")


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
                "Save to a WAV instead of playing",
                'aai speak "Hello" --out /tmp/hello.wav --sandbox',
            ),
            ("Speak text piped from stdin", 'echo "Hello" | aai speak --sandbox'),
        ]
    ),
)
def speak(
    ctx: typer.Context,
    text: str | None = typer.Argument(None, help="Text to speak. Omit to read from stdin."),
    voice: str | None = typer.Option(None, "--voice", help="Voice id. Server default if omitted."),
    language: str | None = typer.Option(
        None, "--language", help="Language. Server default if omitted."
    ),
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
    This feature only exists in the sandbox today — run it with --sandbox.
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
        cfg = session.SpeakConfig(
            text=spoken, voice=voice, language=language, sample_rate=sample_rate
        )
        with output.status("Synthesizing speech…", json_mode=json_mode, quiet=state.quiet):
            result = session.synthesize(
                api_key, cfg, on_warning=lambda m: output.emit_warning(m, json_mode=json_mode)
            )
        if out is not None:
            audio.write_wav(out, result.pcm, result.sample_rate)
        else:
            audio.play_pcm(result.pcm, result.sample_rate)
        _emit_result(result, cfg, out, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
