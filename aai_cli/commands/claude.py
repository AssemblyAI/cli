from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import typer

from aai_cli import output
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.help_text import examples_epilog
from aai_cli.steps import Step, render_steps

app = typer.Typer(
    help="Wire up Claude Code for AssemblyAI (docs MCP + skill).",
    no_args_is_help=True,
)

MCP_NAME = "assemblyai-docs"
MCP_URL = "https://mcp.assemblyai.com/docs"
SKILL_REPO = "AssemblyAI/assemblyai-skill"
_VALID_SCOPES = ("user", "project", "local")
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


_SKILL_ADD = ["npx", "-y", "skills", "add", SKILL_REPO, "--global", "--yes"]
_SKILL_REMOVE = ["npx", "-y", "skills", "remove", "assemblyai", "--global"]
_SKILL_ADD_HINT = f"npx skills add {SKILL_REPO} --global"

CLI_SKILL_REPO = "AssemblyAI/cli"
_CLI_SKILL_NAME = "aai-cli"
_CLI_SKILL_ADD = [
    "npx",
    "-y",
    "skills",
    "add",
    CLI_SKILL_REPO,
    "--skill",
    _CLI_SKILL_NAME,
    "--global",
    "--yes",
]
_CLI_SKILL_REMOVE = ["npx", "-y", "skills", "remove", _CLI_SKILL_NAME, "--global", "--yes"]
_CLI_SKILL_ADD_HINT = f"npx skills add {CLI_SKILL_REPO} --skill {_CLI_SKILL_NAME} --global"


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


def _skill_dir() -> Path:
    # Honor CLAUDE_CONFIG_DIR so install/status/remove agree with Claude Code's
    # actual config root rather than assuming ~/.claude.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills" / "assemblyai"


def _skill_installed() -> bool:
    return (_skill_dir() / "SKILL.md").exists()


def _mcp_status() -> Step:
    if shutil.which("claude") is None:
        return {"name": "mcp", "status": "unknown", "detail": "Claude Code not found"}
    present = _mcp_present()
    return {
        "name": "mcp",
        "status": "installed" if present else "not_installed",
        "detail": MCP_NAME,
    }


def _skill_status() -> Step:
    return {
        "name": "skill",
        "status": "installed" if _skill_installed() else "not_installed",
        "detail": str(_skill_dir()),
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


def _cli_skill_dir() -> Path:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills" / _CLI_SKILL_NAME


def _cli_skill_installed() -> bool:
    return (_cli_skill_dir() / "SKILL.md").exists()


def _install_cli_skill(force: bool) -> Step:
    if shutil.which("npx") is None:
        return {
            "name": "aai-cli skill",
            "status": "skipped",
            "detail": f"Node.js/npx not found. Install Node.js, then run: {_CLI_SKILL_ADD_HINT}",
        }
    if _cli_skill_installed() and not force:
        return {
            "name": "aai-cli skill",
            "status": "already",
            "detail": f"aai-cli skill at {_cli_skill_dir()}",
        }
    proc = _run(_CLI_SKILL_ADD, timeout=300)
    if proc.returncode != 0:
        return {"name": "aai-cli skill", "status": "failed", "detail": _proc_detail(proc)}
    if not _cli_skill_installed():
        return {
            "name": "aai-cli skill",
            "status": "failed",
            "detail": (
                f"'{' '.join(_CLI_SKILL_ADD[3:])}' reported success but no skill was found at "
                f"{_cli_skill_dir()}. Install it manually: {_CLI_SKILL_ADD_HINT}"
            ),
        }
    return {"name": "aai-cli skill", "status": "installed", "detail": str(_cli_skill_dir())}


def _cli_skill_status() -> Step:
    return {
        "name": "aai-cli skill",
        "status": "installed" if _cli_skill_installed() else "not_installed",
        "detail": str(_cli_skill_dir()),
    }


def _remove_cli_skill() -> Step:
    if not _cli_skill_installed():
        return {"name": "aai-cli skill", "status": "not_installed", "detail": str(_cli_skill_dir())}
    if shutil.which("npx") is None:
        return {
            "name": "aai-cli skill",
            "status": "skipped",
            "detail": f"Node.js/npx not found. Remove manually: {' '.join(_CLI_SKILL_REMOVE)}",
        }
    proc = _run(_CLI_SKILL_REMOVE, timeout=120)
    if proc.returncode != 0 or _cli_skill_installed():
        detail = _proc_detail(proc) or "skill still present after removal"
        return {"name": "aai-cli skill", "status": "failed", "detail": detail}
    return {"name": "aai-cli skill", "status": "removed", "detail": str(_cli_skill_dir())}


def _render(data: dict[str, list[Step]]) -> str:
    return render_steps(data["steps"], heading=_STEPS_HEADING)


@app.command(
    epilog=examples_epilog(
        [
            ("Wire AssemblyAI docs + skill into Claude Code", "aai claude install"),
            ("Install for the current project only", "aai claude install --scope project"),
        ]
    )
)
def install(
    ctx: typer.Context,
    scope: str = typer.Option(
        "user",
        "--scope",
        help=(
            "Config scope to register the MCP under: user, project, or local. "
            "Presence is detected across all scopes."
        ),
    ),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already present."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Install the AssemblyAI docs MCP server and skill into Claude Code."""

    def body(_state: AppState, json_mode: bool) -> None:
        if scope not in _VALID_SCOPES:
            raise UsageError(
                f"Invalid --scope '{scope}'. Choose one of: {', '.join(_VALID_SCOPES)}."
            )
        steps = [_install_mcp(scope, force), _install_skill(force), _install_cli_skill(force)]
        output.emit({"steps": steps}, _render, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Show whether Claude Code is wired up", "aai claude status"),
        ]
    )
)
def status(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show whether the AssemblyAI MCP server and skill are wired into Claude Code."""

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [_mcp_status(), _skill_status(), _cli_skill_status()]
        output.emit({"steps": steps}, _render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Remove the AssemblyAI MCP server and skill", "aai claude remove"),
        ]
    )
)
def remove(
    ctx: typer.Context,
    scope: str | None = typer.Option(
        None,
        "--scope",
        help=(
            "Only remove the MCP from this scope (user, project, or local). "
            "Default: remove from whichever scope it exists in."
        ),
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Remove the AssemblyAI MCP server and skill from Claude Code."""

    def body(_state: AppState, json_mode: bool) -> None:
        if scope is not None and scope not in _VALID_SCOPES:
            raise UsageError(
                f"Invalid --scope '{scope}'. Choose one of: {', '.join(_VALID_SCOPES)}."
            )
        steps = [_remove_mcp(scope), _remove_skill(), _remove_cli_skill()]
        output.emit({"steps": steps}, _render, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
