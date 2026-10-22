from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.dictate import _exec as dictate_exec
from aai_cli.core import choices
from aai_cli.core.dictation import MAX_AUDIO_SECONDS
from aai_cli.ui.help_text import examples_epilog

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
            ("Record until SIGTERM, then print the cleaned-up text", "assembly dictate"),
            (
                "Stop the recording and transcribe (e.g. from a hotkey tool)",
                "kill -TERM $(pgrep -f 'assembly dictate')",
            ),
            (
                "Pipe the utterance into another command",
                'assembly dictate | assembly llm "write a conventional commit"',
            ),
            ("Dictate in Spanish", "assembly dictate --language es"),
            (
                "Bias recognition toward tricky terms",
                "assembly dictate --keyterm AssemblyAI --keyterm LeMUR",
            ),
            (
                "Reshape the text instead of the default cleanup",
                'assembly dictate --instruction "Format as a bulleted task list"',
            ),
            ("Print the verbatim transcript, unedited", "assembly dictate --verbatim"),
            ("Emit the utterance as a JSON object", "assembly dictate --json"),
            ("Pipe the bare transcript onward", "assembly dictate -o text | assembly llm -f"),
        ]
    ),
)
def dictate(
    ctx: typer.Context,
    language: str | None = typer.Option(
        None,
        "--language",
        help="ISO 639-1 language code, or a comma-separated list for "
        "code-switching audio (default: no language steering)",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Context for the transcription: what the audio is about (e.g. "
        '"A doctor dictating a visit note")',
    ),
    keyterm: list[str] | None = typer.Option(
        None, "--keyterm", help="Bias recognition toward a term, such as a name (repeatable)"
    ),
    word_boost: list[str] | None = typer.Option(
        None, "--word-boost", hidden=True, help="Deprecated: use --keyterm"
    ),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        help="Describe the shape you want the text in; replaces the default cleanup",
    ),
    rewrite_mode: choices.RewriteMode | None = typer.Option(
        None,
        "--rewrite-mode",
        help="Rewrite model tier: fast (default, lowest latency) or smart "
        "(follows detailed instructions more reliably)",
    ),
    verbatim: bool = typer.Option(
        False, "--verbatim", help="Print the verbatim transcript instead of the cleaned-up text"
    ),
    device: int | None = typer.Option(None, "--device", help="Microphone device index"),
    once: bool = typer.Option(
        False,
        "--once",
        hidden=True,
        help="Deprecated: recording one utterance and exiting is now the default",
    ),
    max_seconds: float = typer.Option(
        float(MAX_AUDIO_SECONDS),
        "--max-seconds",
        help="Auto-stop a recording after this many seconds",
        min=1.0,
        max=float(MAX_AUDIO_SECONDS),
    ),
    json_out: bool = options.json_option("Emit one JSON object per utterance"),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: text (the bare transcript per utterance, pipe-friendly) or json",
    ),
) -> None:
    """Signal-driven dictation: record the mic, get the transcript back

    Recording starts immediately and runs headless — no terminal needed — so a
    hotkey tool like Hammerspoon can launch it as a background task and send
    SIGTERM (kill -TERM, task:terminate()) to stop. On SIGTERM the utterance is
    sent to the AssemblyAI Dictation API, which returns send-ready text — filler
    removed, self-corrections resolved, punctuation applied — that prints right
    away, and dictate exits, so it flows straight to the next command in a pipe.
    --instruction asks for a different shape; --verbatim prints the unedited
    transcript. The recording can be up to 120 seconds long. Ctrl-C (SIGINT)
    cancels without transcribing.
    """
    opts = dictate_exec.DictateOptions(
        language=language,
        prompt=prompt,
        keyterms=keyterm,
        device=device,
        once=once,
        max_seconds=max_seconds,
        output_field=output_field,
        instruction=instruction,
        rewrite_mode=rewrite_mode,
        verbatim=verbatim,
        word_boost=word_boost,
    )
    run_with_options(ctx, dictate_exec.run_dictate, opts, json=json_out)
