"""Run logic for `assembly deploy`: ship the current project to a PaaS target.

The command module (aai_cli/commands/deploy/__init__.py) only parses argv — it builds a
``DeployOptions`` and hands it to ``run_deploy`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive target resolution and the
subprocess orchestration by constructing options directly instead of
round-tripping argv.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from aai_cli import output
from aai_cli.context import AppState
from aai_cli.errors import CLIError, UsageError
from aai_cli.init import procfile


@dataclass(frozen=True)
class Target:
    """A deploy provider: its CLI binary, selector flag, install hint, and argv recipe."""

    name: str  # human label, e.g. "Vercel"
    bin: str  # executable resolved via shutil.which
    flag: str  # CLI selector, e.g. "--vercel"
    install: str  # hint sentence shown when the CLI is missing (everywhere, or macOS-only)
    deploy_args: tuple[str, ...]  # subcommand(s) appended after `bin`
    supports_prod: bool = False  # whether `--prod` adds a production flag
    post_deploy_args: tuple[str, ...] | None = None  # command run after a successful deploy
    install_non_darwin: str | None = None  # hint off-macOS, when `install` is brew-specific

    def command(self, *, prod: bool) -> list[str]:
        """The deploy argv, appending ``--prod`` when requested and the target supports it."""
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


@dataclass(frozen=True)
class DeployOptions:
    """Every `assembly deploy` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    prod: bool
    vercel: bool
    railway: bool
    fly: bool
    assume_yes: bool


def _resolve_target(opts: DeployOptions) -> Target:
    selected = [
        t for t, on in ((VERCEL, opts.vercel), (RAILWAY, opts.railway), (FLY, opts.fly)) if on
    ]
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


def run_deploy(opts: DeployOptions, _state: AppState, *, json_mode: bool) -> None:
    """Confirm, then run the target's deploy command in the current directory."""
    target = _resolve_target(opts)
    if opts.prod and not target.supports_prod:
        raise UsageError(
            "--prod is only supported for Vercel deploys.",
            suggestion=f"Drop --prod, or drop {target.flag} to deploy to Vercel.",
        )
    # Same not-a-project guard as `assembly dev`/`assembly share`, checked before CLI presence
    # so an empty directory says "run `assembly init`", not "install the Vercel CLI".
    procfile.require_procfile(Path.cwd())
    _require_cli(target)
    if not _confirmed(target, assume_yes=opts.assume_yes):
        aborted = {"status": "aborted", "target": target.name}
        output.emit(aborted, lambda _d: "Aborted.", json_mode=json_mode)
        return
    result = subprocess.run(target.command(prod=opts.prod), cwd=Path.cwd(), check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)
    if target.post_deploy_args is not None:
        subprocess.run([target.bin, *target.post_deploy_args], cwd=Path.cwd(), check=False)
