# aai_cli/commands/deploy.py
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer

from aai_cli import help_panels, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.help_text import examples_epilog

# Flattened single-command sub-typer (same pattern as `aai dev`).
app = typer.Typer()


@dataclass(frozen=True)
class Target:
    name: str  # human label, e.g. "Vercel"
    bin: str  # executable resolved via shutil.which
    install: str  # install hint shown when the CLI is missing

    def command(self, *, prod: bool) -> list[str]:
        if self.bin == "vercel":
            return ["vercel", "deploy", *(["--prod"] if prod else [])]
        return ["railway", "up"]  # Railway has no preview/prod split here


VERCEL = Target(name="Vercel", bin="vercel", install="npm i -g vercel")
RAILWAY = Target(name="Railway", bin="railway", install="npm i -g @railway/cli")


def _resolve_target(*, vercel: bool, railway: bool) -> Target:
    if vercel and railway:
        raise CLIError(
            "Pass either --vercel or --railway, not both.",
            error_type="usage_error",
            exit_code=1,
        )
    return RAILWAY if railway else VERCEL  # Vercel is the default


def _require_cli(target: Target) -> None:
    if shutil.which(target.bin) is None:
        raise CLIError(
            f"The {target.name} CLI is required to deploy. Install it with `{target.install}`.",
            error_type="missing_dependency",
            exit_code=1,
        )


def _confirmed(target: Target, *, assume_yes: bool) -> bool:
    """True when the deploy should proceed: --yes, or an interactive yes.

    Refuses to guess in a non-interactive/agent session."""
    if assume_yes:
        return True
    if output.is_agentic():
        raise CLIError(
            "Refusing to deploy without confirmation in a non-interactive session. "
            "Pass --yes to deploy.",
            error_type="usage_error",
            exit_code=1,
        )
    return typer.confirm(f"Deploy this project to {target.name}?")


def run_deploy(*, target: Target, prod: bool, assume_yes: bool) -> None:
    """Confirm, then run the target's deploy command in the current directory."""
    _require_cli(target)
    if not _confirmed(target, assume_yes=assume_yes):
        output.console.print("Aborted.")
        return
    result = subprocess.run(target.command(prod=prod), cwd=Path.cwd(), check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Deploy a preview to Vercel (asks first)", "aai deploy"),
            ("Deploy to production on Vercel", "aai deploy --prod --yes"),
            ("Deploy to Railway", "aai deploy --railway"),
        ]
    ),
)
def deploy(
    ctx: typer.Context,
    prod: bool = typer.Option(False, "--prod", help="Deploy to production (Vercel only)."),
    vercel: bool = typer.Option(False, "--vercel", help="Deploy to Vercel (the default)."),
    railway: bool = typer.Option(False, "--railway", help="Deploy to Railway."),
    assume_yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Deploy the current project to Vercel (default) or Railway.

    Asks for confirmation first, then runs the target's CLI (`vercel deploy` or
    `railway up`). Requires that target's CLI to be installed.
    """

    def body(_state: AppState, _json_mode: bool) -> None:
        run_deploy(
            target=_resolve_target(vercel=vercel, railway=railway),
            prod=prod,
            assume_yes=assume_yes,
        )

    run_command(ctx, body)
