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
    flag: str  # CLI selector, e.g. "--vercel"
    install: str  # full hint sentence shown when the CLI is missing
    deploy_args: tuple[str, ...]  # subcommand(s) appended after `bin`
    supports_prod: bool = False  # whether `--prod` adds a production flag
    post_deploy_args: tuple[str, ...] | None = None  # command run after a successful deploy
    requires_file: str | None = None  # a file that must exist in cwd before deploying
    setup_hint: str | None = None  # how to create `requires_file`

    def command(self, *, prod: bool) -> list[str]:
        argv = [self.bin, *self.deploy_args]
        if prod and self.supports_prod:
            argv.append("--prod")
        return argv


VERCEL = Target(
    name="Vercel",
    bin="vercel",
    flag="--vercel",
    install="Install it with `npm i -g vercel`.",
    deploy_args=("deploy",),
    supports_prod=True,
)
RAILWAY = Target(
    name="Railway",
    bin="railway",
    flag="--railway",
    install="Install it with `npm i -g @railway/cli`.",
    deploy_args=("up",),
    post_deploy_args=("domain",),
)
FLY = Target(
    name="Fly",
    bin="fly",
    flag="--fly",
    install="Install it with `brew install flyctl`.",
    deploy_args=("deploy",),
    requires_file="fly.toml",
    setup_hint="Run `fly launch` first to create your Fly app.",
)

TARGETS = (VERCEL, RAILWAY, FLY)


def _resolve_target(selected: list[Target]) -> Target:
    if len(selected) > 1:
        flags = " / ".join(t.flag for t in TARGETS)
        raise CLIError(
            f"Pass at most one deploy target ({flags}).",
            error_type="usage_error",
            exit_code=1,
        )
    return selected[0] if selected else VERCEL  # Vercel is the default


def _require_cli(target: Target) -> None:
    if shutil.which(target.bin) is None:
        raise CLIError(
            f"The {target.name} CLI is required to deploy. {target.install}",
            error_type="missing_dependency",
            exit_code=1,
        )


def _require_setup(target: Target) -> None:
    if target.requires_file is not None and not (Path.cwd() / target.requires_file).exists():
        raise CLIError(
            f"No {target.requires_file} in this directory. {target.setup_hint}",
            error_type="usage_error",
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
    _require_setup(target)
    if not _confirmed(target, assume_yes=assume_yes):
        output.console.print("Aborted.")
        return
    result = subprocess.run(target.command(prod=prod), cwd=Path.cwd(), check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)
    if target.post_deploy_args is not None:
        subprocess.run([target.bin, *target.post_deploy_args], cwd=Path.cwd(), check=False)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Deploy a preview to Vercel (asks first)", "aai deploy"),
            ("Deploy to production on Vercel", "aai deploy --prod --yes"),
            ("Deploy to Railway", "aai deploy --railway"),
            ("Deploy to Fly.io", "aai deploy --fly"),
        ]
    ),
)
def deploy(
    ctx: typer.Context,
    prod: bool = typer.Option(False, "--prod", help="Deploy to production (Vercel only)."),
    vercel: bool = typer.Option(False, "--vercel", help="Deploy to Vercel (the default)."),
    railway: bool = typer.Option(False, "--railway", help="Deploy to Railway."),
    fly: bool = typer.Option(False, "--fly", help="Deploy to Fly.io."),
    assume_yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Deploy the current project to Vercel (default), Railway, or Fly.io.

    Asks for confirmation first, then runs the target's CLI (`vercel deploy`,
    `railway up`, or `fly deploy`). Requires that target's CLI to be installed.
    (Render deploys from a connected Git repo — see the project README.)
    """

    def body(_state: AppState, _json_mode: bool) -> None:
        selected = [t for t, on in ((VERCEL, vercel), (RAILWAY, railway), (FLY, fly)) if on]
        run_deploy(target=_resolve_target(selected), prod=prod, assume_yes=assume_yes)

    run_command(ctx, body)
