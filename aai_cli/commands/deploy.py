# aai_cli/commands/deploy.py
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from aai_cli import help_panels, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError, UsageError
from aai_cli.help_text import examples_epilog
from aai_cli.init import procfile

# Flattened single-command sub-typer (same pattern as `assembly dev`).
app = typer.Typer()


@dataclass(frozen=True)
class Target:
    name: str  # human label, e.g. "Vercel"
    bin: str  # executable resolved via shutil.which
    flag: str  # CLI selector, e.g. "--vercel"
    install: str  # hint sentence shown when the CLI is missing (everywhere, or macOS-only)
    deploy_args: tuple[str, ...]  # subcommand(s) appended after `bin`
    supports_prod: bool = False  # whether `--prod` adds a production flag
    post_deploy_args: tuple[str, ...] | None = None  # command run after a successful deploy
    install_non_darwin: str | None = None  # hint off-macOS, when `install` is brew-specific

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
    # brew is macOS-specific; elsewhere point at the official install docs.
    install="Install it with `brew install flyctl`.",
    install_non_darwin="Install it: https://fly.io/docs/flyctl/install/",
    # `fly launch` does it all: creates the app, generates fly.toml (detecting the
    # shipped Dockerfile), and deploys — so no fly.toml needs to exist beforehand.
    deploy_args=("launch",),
)

TARGETS = (VERCEL, RAILWAY, FLY)


def _resolve_target(selected: list[Target]) -> Target:
    if len(selected) > 1:
        flags = " / ".join(t.flag for t in TARGETS)
        raise UsageError(f"Pass at most one deploy target ({flags}).")
    return selected[0] if selected else VERCEL  # Vercel is the default


def _install_hint(target: Target) -> str:
    """The platform-appropriate install hint: brew on macOS, docs URL elsewhere."""
    if target.install_non_darwin is not None and sys.platform != "darwin":
        return target.install_non_darwin
    return target.install


def _require_cli(target: Target) -> None:
    if shutil.which(target.bin) is None:
        raise CLIError(
            f"The {target.name} CLI is required to deploy. {_install_hint(target)}",
            error_type="missing_dependency",
            exit_code=1,
        )


def _confirmed(target: Target, *, assume_yes: bool) -> bool:
    """True when the deploy should proceed: --yes, or an interactive yes.

    Refuses to guess in a non-interactive/agent session."""
    if assume_yes:
        return True
    if output.is_agentic():
        raise UsageError(
            "Refusing to deploy without confirmation in a non-interactive session. "
            "Pass --yes to deploy."
        )
    return typer.confirm(f"Deploy this project to {target.name}?")


def run_deploy(*, target: Target, prod: bool, assume_yes: bool) -> None:
    """Confirm, then run the target's deploy command in the current directory."""
    if prod and not target.supports_prod:
        raise UsageError(
            "--prod is only supported for Vercel deploys.",
            suggestion=f"Drop --prod, or drop {target.flag} to deploy to Vercel.",
        )
    # Same not-a-project guard as `assembly dev`/`assembly share`, checked before CLI presence
    # so an empty directory says "run `assembly init`", not "install the Vercel CLI".
    procfile.require_procfile(Path.cwd())
    _require_cli(target)
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
            ("Deploy a preview to Vercel (asks first)", "assembly deploy"),
            ("Deploy to production on Vercel", "assembly deploy --prod --yes"),
            ("Deploy to Railway", "assembly deploy --railway"),
            ("Deploy to Fly.io", "assembly deploy --fly"),
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
    `railway up`, or `fly launch`). Requires that target's CLI to be installed.
    (Render deploys from a connected Git repo — see the project README.)
    """

    def body(_state: AppState, _json_mode: bool) -> None:
        selected = [t for t, on in ((VERCEL, vercel), (RAILWAY, railway), (FLY, fly)) if on]
        run_deploy(target=_resolve_target(selected), prod=prod, assume_yes=assume_yes)

    run_command(ctx, body)
