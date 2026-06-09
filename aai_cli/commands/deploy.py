# aai_cli/commands/deploy.py
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from aai_cli import help_panels, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.help_text import examples_epilog

# Flattened single-command sub-typer (same pattern as `aai dev`).
app = typer.Typer()

VERCEL = "vercel"


def _require_vercel() -> None:
    if shutil.which(VERCEL) is None:
        raise CLIError(
            "The Vercel CLI is required to deploy. Install it with `npm i -g vercel`.",
            error_type="missing_dependency",
            exit_code=1,
        )


def _confirmed(*, assume_yes: bool) -> bool:
    """True when the deploy should proceed: --yes, or an interactive yes.

    Refuses to guess in a non-interactive/agent session (would otherwise hang or
    deploy unintentionally)."""
    if assume_yes:
        return True
    if output.is_agentic():
        raise CLIError(
            "Refusing to deploy without confirmation in a non-interactive session. "
            "Pass --yes to deploy.",
            error_type="usage_error",
            exit_code=1,
        )
    return typer.confirm("Deploy this project to Vercel?")


def run_deploy(*, prod: bool, assume_yes: bool) -> None:
    """Confirm, then run `vercel deploy` in the current directory."""
    _require_vercel()
    if not _confirmed(assume_yes=assume_yes):
        output.console.print("Aborted.")
        return
    cmd = [VERCEL, "deploy", *(["--prod"] if prod else [])]
    result = subprocess.run(cmd, cwd=Path.cwd(), check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Deploy a preview to Vercel (asks first)", "aai deploy"),
            ("Deploy to production without prompting", "aai deploy --prod --yes"),
        ]
    ),
)
def deploy(
    ctx: typer.Context,
    prod: bool = typer.Option(False, "--prod", help="Deploy to production."),
    assume_yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Deploy the current project to Vercel.

    Asks for confirmation first, then runs `vercel deploy` (pass `--prod` to promote
    to production). Requires the Vercel CLI (`npm i -g vercel`).
    """

    def body(_state: AppState, _json_mode: bool) -> None:
        run_deploy(prod=prod, assume_yes=assume_yes)

    run_command(ctx, body)
