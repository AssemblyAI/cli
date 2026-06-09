from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from aai_cli import choices, output
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog
from aai_cli.steps import Step, render_steps

if TYPE_CHECKING:
    # Annotation only (PEP 563 string), so no runtime import. Import from
    # importlib.abc — that is the protocol `resources.files()` is typed to return.
    from importlib.abc import Traversable

app = typer.Typer(
    help="Set up your coding agent for AssemblyAI (docs MCP + skills).",
    no_args_is_help=True,
)

MCP_NAME = "assemblyai-docs"
MCP_URL = "https://mcp.assemblyai.com/docs"
SKILL_REPO = "AssemblyAI/assemblyai-skill"
_STEPS_HEADING = "AssemblyAI coding-agent setup:"


def _run(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    # stdin=DEVNULL so a child that would otherwise prompt (npx's "Ok to proceed?",
    # a `claude` confirmation) gets EOF and fails fast instead of hanging forever on
    # input the user can't see (its stdout is captured). timeout is a final backstop.
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout="",
            stderr=f"timed out after {timeout:.0f}s: {' '.join(cmd)}",
        )


def _proc_detail(proc: subprocess.CompletedProcess[str]) -> str:
    """The error text from a finished process: stderr if present, else stdout."""
    return (proc.stderr or proc.stdout).strip()


def _skills_root() -> Path:
    # Honor CLAUDE_CONFIG_DIR so install/status/remove agree with the agent's actual
    # config root rather than assuming ~/.claude.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills"


# --- docs MCP (registered via the `claude` CLI) ------------------------------


def _mcp_present() -> bool:
    return _run(["claude", "mcp", "get", MCP_NAME]).returncode == 0


def _install_mcp(scope: str, force: bool) -> Step:
    if shutil.which("claude") is None:
        return {
            "name": "mcp",
            "status": "skipped",
            "detail": (
                "Claude Code not found. Install it (https://claude.com/claude-code), "
                f"then run: claude mcp add --transport http --scope {scope} "
                f"{MCP_NAME} {MCP_URL}"
            ),
        }
    if _mcp_present():
        if not force:
            return {"name": "mcp", "status": "already", "detail": f"{MCP_NAME} already registered"}
        removed = _run(["claude", "mcp", "remove", MCP_NAME])
        if removed.returncode != 0:
            return {
                "name": "mcp",
                "status": "failed",
                "detail": f"could not remove existing {MCP_NAME}: " + _proc_detail(removed),
            }
    proc = _run(
        ["claude", "mcp", "add", "--transport", "http", "--scope", scope, MCP_NAME, MCP_URL]
    )
    if proc.returncode != 0:
        return {"name": "mcp", "status": "failed", "detail": _proc_detail(proc)}
    return {"name": "mcp", "status": "installed", "detail": f"{MCP_NAME} @ {scope} scope"}


def _mcp_status() -> Step:
    if shutil.which("claude") is None:
        return {"name": "mcp", "status": "unknown", "detail": "Claude Code not found"}
    present = _mcp_present()
    return {
        "name": "mcp",
        "status": "installed" if present else "not_installed",
        "detail": MCP_NAME,
    }


def _remove_mcp(scope: str | None) -> Step:
    if shutil.which("claude") is None:
        return {"name": "mcp", "status": "skipped", "detail": "Claude Code not found"}
    if not _mcp_present():
        return {"name": "mcp", "status": "not_installed", "detail": MCP_NAME}
    cmd = ["claude", "mcp", "remove", MCP_NAME]
    if scope is not None:
        cmd += ["--scope", scope]
    proc = _run(cmd)
    if proc.returncode != 0:
        return {"name": "mcp", "status": "failed", "detail": _proc_detail(proc)}
    return {"name": "mcp", "status": "removed", "detail": MCP_NAME}


# --- assemblyai skill (downloaded from its own repo via the `skills` CLI) -----

_SKILL_ADD = ["npx", "-y", "skills", "add", SKILL_REPO, "--global", "--yes"]
_SKILL_REMOVE = ["npx", "-y", "skills", "remove", "assemblyai", "--global"]
_SKILL_ADD_HINT = f"npx skills add {SKILL_REPO} --global"


def _skill_dir() -> Path:
    return _skills_root() / "assemblyai"


def _skill_installed() -> bool:
    return (_skill_dir() / "SKILL.md").exists()


def _install_skill(force: bool) -> Step:
    if shutil.which("npx") is None:
        return {
            "name": "skill",
            "status": "skipped",
            "detail": f"Node.js/npx not found. Install Node.js, then run: {_SKILL_ADD_HINT}",
        }
    # Idempotent like the MCP step: if the skill is already on disk and the user
    # didn't ask to --force, report `already` instead of silently re-downloading
    # it and always claiming `installed`.
    if _skill_installed() and not force:
        return {
            "name": "skill",
            "status": "already",
            "detail": f"assemblyai skill at {_skill_dir()}",
        }
    # --global: install at user scope (not project scope, which `skills` auto-selects
    # when run inside a project) so the skill lands in ~/.claude/skills where `status`
    # looks. npx -y skips its install prompt; the longer timeout covers the download.
    proc = _run(_SKILL_ADD, timeout=300)
    if proc.returncode != 0:
        return {"name": "skill", "status": "failed", "detail": _proc_detail(proc)}
    # Trust the filesystem, not the exit code: confirm the skill actually landed
    # where `status` looks, so the two commands can never disagree.
    if not _skill_installed():
        return {
            "name": "skill",
            "status": "failed",
            "detail": (
                f"'{' '.join(_SKILL_ADD[3:])}' reported success but no skill was found at "
                f"{_skill_dir()}. Install it manually: {_SKILL_ADD_HINT}"
            ),
        }
    return {"name": "skill", "status": "installed", "detail": str(_skill_dir())}


def _skill_status() -> Step:
    return {
        "name": "skill",
        "status": "installed" if _skill_installed() else "not_installed",
        "detail": str(_skill_dir()),
    }


def _remove_skill() -> Step:
    if not _skill_installed():
        return {"name": "skill", "status": "not_installed", "detail": str(_skill_dir())}
    if shutil.which("npx") is None:
        return {
            "name": "skill",
            "status": "skipped",
            "detail": "Node.js/npx not found. Remove manually: npx skills remove assemblyai --global",
        }
    # `skills` symlinks the skill into ~/.claude/skills from its own store, so let it
    # do the removal (a plain rmtree would choke on the symlink and orphan the store).
    proc = _run(_SKILL_REMOVE, timeout=120)
    if proc.returncode != 0 or _skill_installed():
        detail = _proc_detail(proc) or "skill still present after removal"
        return {"name": "skill", "status": "failed", "detail": detail}
    return {"name": "skill", "status": "removed", "detail": str(_skill_dir())}


# --- aai-cli skill (bundled in this package, copied into the agent) -----------

_CLI_SKILL_NAME = "aai-cli"


def _cli_skill_dir() -> Path:
    return _skills_root() / _CLI_SKILL_NAME


def _cli_skill_installed() -> bool:
    return (_cli_skill_dir() / "SKILL.md").exists()


def _bundled_cli_skill() -> Traversable:
    # Ships inside the wheel (force-included via [tool.hatch.build.targets.wheel]
    # artifacts). skills/ has no __init__.py, so navigate from the aai_cli package.
    from importlib import resources

    return resources.files("aai_cli") / "skills" / _CLI_SKILL_NAME


def _copy_tree(node: Traversable, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for child in node.iterdir():
        if child.name == "__pycache__" or child.name.endswith(".pyc"):
            continue
        out = dest / child.name
        if child.is_dir():
            _copy_tree(child, out)
        else:
            out.write_bytes(child.read_bytes())


def _install_cli_skill(force: bool) -> Step:
    # Bundled in the package, so no network/npx — just copy it into the agent's
    # skills dir. Idempotent: skip the copy when already present and not --force.
    dest = _cli_skill_dir()
    if _cli_skill_installed() and not force:
        return {"name": "aai-cli skill", "status": "already", "detail": f"aai-cli skill at {dest}"}
    src = _bundled_cli_skill()
    if not src.is_dir():
        return {
            "name": "aai-cli skill",
            "status": "failed",
            "detail": f"bundled aai-cli skill missing at {src} — this is a packaging bug.",
        }
    if dest.exists():
        shutil.rmtree(dest)
    _copy_tree(src, dest)
    if not _cli_skill_installed():
        return {
            "name": "aai-cli skill",
            "status": "failed",
            "detail": f"copied the bundled skill but {dest / 'SKILL.md'} is missing.",
        }
    return {"name": "aai-cli skill", "status": "installed", "detail": str(dest)}


def _cli_skill_status() -> Step:
    return {
        "name": "aai-cli skill",
        "status": "installed" if _cli_skill_installed() else "not_installed",
        "detail": str(_cli_skill_dir()),
    }


def _remove_cli_skill() -> Step:
    # We copied a real directory in (not a symlink into a store), so removal is a
    # plain rmtree of the destination.
    dest = _cli_skill_dir()
    if not _cli_skill_installed():
        return {"name": "aai-cli skill", "status": "not_installed", "detail": str(dest)}
    shutil.rmtree(dest, ignore_errors=True)
    if _cli_skill_installed():
        return {
            "name": "aai-cli skill",
            "status": "failed",
            "detail": "skill still present after removal",
        }
    return {"name": "aai-cli skill", "status": "removed", "detail": str(dest)}


def _render(data: dict[str, list[Step]]) -> str:
    return render_steps(data["steps"], heading=_STEPS_HEADING)


@app.command(
    epilog=examples_epilog(
        [
            ("Set up your coding agent for AssemblyAI", "aai setup install"),
            ("Install for the current project only", "aai setup install --scope project"),
            ("Reinstall everything even if already present", "aai setup install --force"),
        ]
    )
)
def install(
    ctx: typer.Context,
    scope: choices.Scope = typer.Option(
        choices.Scope.user,
        "--scope",
        help="Config scope to register the MCP under. Presence is detected across all scopes.",
    ),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already present."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Set up your coding agent for AssemblyAI by installing three things:

    the assemblyai-docs MCP server (live API docs, via `claude mcp add`), the AssemblyAI
    skill (via `npx skills add`), and the bundled aai-cli skill (copied from this package,
    no network). Each step is idempotent and skipped if already present unless --force.
    """

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [_install_mcp(scope, force), _install_skill(force), _install_cli_skill(force)]
        output.emit({"steps": steps}, _render, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Show what's set up", "aai setup status"),
            ("Print status as JSON", "aai setup status --json"),
        ]
    )
)
def status(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show whether the AssemblyAI MCP server and skills are set up in your coding agent."""

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [_mcp_status(), _skill_status(), _cli_skill_status()]
        output.emit({"steps": steps}, _render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Remove the AssemblyAI MCP server and skills", "aai setup remove"),
            ("Remove only from the project scope", "aai setup remove --scope project"),
        ]
    )
)
def remove(
    ctx: typer.Context,
    scope: choices.Scope | None = typer.Option(
        None,
        "--scope",
        help=(
            "Only remove the MCP from this scope. "
            "Default: remove from whichever scope it exists in."
        ),
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Remove the AssemblyAI MCP server and skills from your coding agent."""

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [_remove_mcp(scope), _remove_skill(), _remove_cli_skill()]
        output.emit({"steps": steps}, _render, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
