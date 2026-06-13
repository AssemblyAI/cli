from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.commands.dictate import _exec as dictate_exec
from aai_cli.context import run_command
from aai_cli.help_text import examples_epilog
from aai_cli.sync_stt import MAX_AUDIO_SECONDS

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=30,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("dictate",),
)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Dictate: Enter starts a recording, Enter transcribes it", "assembly dictate"),
            ("One utterance, then exit", "assembly dictate --once"),
            ("Dictate in Spanish", "assembly dictate --language es"),
            (
                "Bias recognition toward tricky terms",
                "assembly dictate --word-boost AssemblyAI --word-boost LeMUR",
            ),
            ("One JSON object per utterance", "assembly dictate --json"),
        ]
    ),
)
def dictate(
    ctx: typer.Context,
    language: str | None = typer.Option(
        None,
        "--language",
        help="ISO 639-1 language code, or a comma-separated list for "
        "code-switching audio (default: en).",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Custom transcription prompt (overrides --language).",
    ),
    word_boost: list[str] | None = typer.Option(
        None, "--word-boost", help="Bias recognition toward a term (repeatable)."
    ),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    once: bool = typer.Option(False, "--once", help="Transcribe one utterance, then exit."),
    max_seconds: float = typer.Option(
        float(MAX_AUDIO_SECONDS),
        "--max-seconds",
        help="Auto-stop a recording after this many seconds.",
        min=1.0,
        max=float(MAX_AUDIO_SECONDS),
    ),
    json_out: bool = options.json_option("Emit one JSON object per utterance."),
) -> None:
    """Dictate with a hotkey: record the mic, get the transcript back instantly.

    Press Enter (or Space) to start recording and press it again to stop; the
    utterance is sent to the AssemblyAI Sync API and the transcript prints
    immediately — no polling. Press q (or Esc/Ctrl-C) to finish. Each utterance
    can be up to 120 seconds long.
    """
    opts = dictate_exec.DictateOptions(
        language=language,
        prompt=prompt,
        word_boost=word_boost,
        device=device,
        once=once,
        max_seconds=max_seconds,
    )
    run_command(
        ctx,
        lambda state, json_mode: dictate_exec.run_dictate(opts, state, json_mode=json_mode),
        json=json_out,
    )
