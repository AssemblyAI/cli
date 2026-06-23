from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.control import _exec as control_exec
from aai_cli.core import llm
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=47,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("control",),
)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Control your Mac hands-free by voice", "assembly control"),
            ("Preview actions without touching the UI", "assembly control --dry-run"),
            ("Use a more capable model for the agent", "assembly control --model claude-opus-4-7"),
            ("Emit the loop as newline-delimited JSON", "assembly control --json"),
        ]
    ),
)
def control(
    ctx: typer.Context,
    device: int | None = typer.Option(
        None,
        "--device",
        help="Microphone device index",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        help="Microphone capture rate in Hz (default: device native)",
        min=1,
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model that decides the actions",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens per agent step",
        min=1,
        rich_help_panel=help_panels.OPT_LLM,
    ),
    max_steps: int = typer.Option(
        10,
        "--max-steps",
        help="Max action steps the agent may take per spoken instruction",
        min=1,
        rich_help_panel=help_panels.OPT_LLM,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan and observe only: refuse every UI-changing action",
    ),
    json_out: bool = options.json_option("Emit newline-delimited JSON events"),
) -> None:
    """Drive your Mac hands-free: speak an instruction, an agent acts on the UI

    Each spoken instruction is transcribed with Streaming STT and handed to an
    LLM agent that decides which UI actions to take — typing, key chords,
    clicking accessibility elements, launching apps — and performs them through a
    bundled native macOS helper, then speaks back a short confirmation.

    macOS only: the helper needs Apple's Swift compiler and the Accessibility +
    Microphone permissions granted to your terminal. Use --dry-run to watch the
    agent plan without it touching anything.
    """
    opts = control_exec.ControlOptions(
        device=device,
        sample_rate=sample_rate,
        model=model,
        max_tokens=max_tokens,
        max_steps=max_steps,
        dry_run=dry_run,
    )
    run_with_options(ctx, control_exec.run_control, opts, json=json_out)
