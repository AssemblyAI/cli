from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.code import _exec as code_exec
from aai_cli.ui.help_text import examples_epilog

# Flattened single-command sub-typer (same pattern as `assembly dev`/`assembly deploy`).
app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=65,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("code",),
)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Launch a coding agent in the current directory", "assembly code"),
            ("Open specific files", "assembly code api/index.py README.md"),
            ("Use a different gateway model", "assembly code --model gpt-5.1"),
        ]
    ),
)
def code(
    ctx: typer.Context,
    files: list[str] | None = typer.Argument(
        None, help="Files or directories to open in the agent", metavar="[FILES]..."
    ),
    model: str = typer.Option(
        code_exec.DEFAULT_MODEL, "--model", help="Gateway model for the agent"
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Launch an AI coding agent powered by the AssemblyAI LLM Gateway

    Starts aider pointed at the gateway (it edits your files over plain chat
    completions), so the same paid-plan key that runs `assembly llm` powers a coding
    agent in your terminal. Requires aider to be installed (`uv tool install aider-chat`).
    """
    opts = code_exec.CodeOptions(model=model, files=tuple(files or ()))
    run_with_options(ctx, code_exec.run_code, opts, json=json_out)
