from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import help_panels, options, speak_exec
from aai_cli.context import run_command
from aai_cli.help_text import examples_epilog
from aai_cli.speak_exec import DEFAULT_LANGUAGE

app = typer.Typer()


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    # --sandbox is a root flag, so it must come before the subcommand in every example.
    epilog=examples_epilog(
        [
            ("Speak text aloud (sandbox only)", 'assembly --sandbox speak "Hello there, friend."'),
            (
                "Pick a voice and language",
                'assembly --sandbox speak "Bonjour" --voice jane --language French',
            ),
            (
                "Speak a diarized transcript, one voice per speaker",
                "assembly transcribe meeting.mp3 --speaker-labels | assembly --sandbox speak",
            ),
            (
                "Override a speaker's voice",
                "… | assembly --sandbox speak --voice A=vera --voice B=paul",
            ),
            (
                "Save to a WAV instead of playing",
                'assembly --sandbox speak "Hello" --out /tmp/hello.wav',
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
        help="Voice id (e.g. jane, michael, mary, paul, eve, george), or SPEAKER=VOICE "
        "for diarized input (repeatable, e.g. --voice A=jane).",
    ),
    language: str = typer.Option(DEFAULT_LANGUAGE, "--language", help="Language of the text."),
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        help="Output sample rate in Hz (positive). Server default if omitted.",
        min=1,
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Write a WAV file instead of playing through the speakers."
    ),
    json_out: bool = options.json_option("Emit JSON metadata about the synthesized audio."),
) -> None:
    """Synthesize speech from text with AssemblyAI streaming TTS (sandbox only).

    Plays the audio through your speakers by default, or writes a WAV with
    --out. Speaker-labeled input (from 'assembly transcribe
    --speaker-labels') is detected automatically: the labels are stripped
    and each speaker gets a different voice. This feature only exists in
    the sandbox today — run it as 'assembly --sandbox speak' (--sandbox
    goes before the subcommand).
    """

    opts = speak_exec.SpeakOptions(
        text=text,
        voice=voice,
        language=language,
        sample_rate=sample_rate,
        out=out,
    )
    run_command(
        ctx,
        lambda state, json_mode: speak_exec.run_speak(opts, state, json_mode=json_mode),
        json=json_out,
    )
