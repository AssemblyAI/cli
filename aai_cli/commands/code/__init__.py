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
            ("Run one instruction and exit", 'assembly code -m "add a test for parse_url"'),
            ("Use a different gateway model", "assembly code --model gpt-5.1"),
        ]
    ),
)
def code(
    ctx: typer.Context,
    model: str = typer.Option(
        code_exec.DEFAULT_MODEL, "--model", help="Gateway model for the agent"
    ),
    message: str | None = typer.Option(
        None, "--message", "-m", help="Run one instruction and exit (non-interactive)"
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Launch an AI coding agent powered by the AssemblyAI LLM Gateway

    Starts opencode pointed at the gateway via a generated config, so the same paid-plan
    key that runs `assembly llm` powers a coding agent in your terminal. The AssemblyAI
    docs MCP and skills are wired in as context. Requires opencode (`npm i -g opencode-ai`).
    """
    opts = code_exec.CodeOptions(model=model, message=message)
    run_with_options(ctx, code_exec.run_code, opts, json=json_out)
