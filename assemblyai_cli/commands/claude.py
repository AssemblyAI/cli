from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict

import typer
from rich.markup import escape

from assemblyai_cli import output
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError

app = typer.Typer(
    help="Wire up Claude Code for AssemblyAI (docs MCP + skill).",
    no_args_is_help=True,
)

MCP_NAME = "assemblyai-docs"
MCP_URL = "https://mcp.assemblyai.com/docs"
SKILL_REPO = "AssemblyAI/assemblyai-skill"
_VALID_SCOPES = ("user", "project", "local")


class Step(TypedDict):
    """One line of setup output: a named step, its status, and a human detail."""

    name: str
    status: str
    detail: str


def _run(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess:
    # stdin=DEVNULL so a child that would otherwise prompt (npx's "Ok to proceed?",
    # a `claude` confirmation) gets EOF and fails fast instead of hanging forever on
    # input the user can't see (its stdout is captured). timeout is a final backstop.
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
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
                "detail": f"could not remove existing {MCP_NAME}: "
                + (removed.stderr or removed.stdout).strip(),
            }
    proc = _run(
        ["claude", "mcp", "add", "--transport", "http", "--scope", scope, MCP_NAME, MCP_URL]
    )
    if proc.returncode != 0:
        return {"name": "mcp", "status": "failed", "detail": (proc.stderr or proc.stdout).strip()}
    return {"name": "mcp", "status": "installed", "detail": f"{MCP_NAME} @ {scope} scope"}


def _install_skill() -> Step:
    if shutil.which("npx") is None:
        return {
            "name": "skill",
            "status": "skipped",
            "detail": (
                f"Node.js/npx not found. Install Node.js, then run: npx skills add {SKILL_REPO}"
            ),
        }
    # -y: skip npx's interactive "Ok to proceed?" prompt; longer timeout covers the download.
    proc = _run(["npx", "-y", "skills", "add", SKILL_REPO], timeout=300)
    if proc.returncode != 0:
        return {"name": "skill", "status": "failed", "detail": (proc.stderr or proc.stdout).strip()}
    # Trust the filesystem, not the exit code: confirm the skill actually landed
    # where `status` looks, so the two commands can never disagree.
    if not _skill_installed():
        return {
            "name": "skill",
            "status": "failed",
            "detail": (
                f"'npx skills add {SKILL_REPO}' reported success but no skill was found at "
                f"{_skill_dir()}. Install it manually: npx skills add {SKILL_REPO}"
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
        return {"name": "mcp", "status": "failed", "detail": (proc.stderr or proc.stdout).strip()}
    return {"name": "mcp", "status": "removed", "detail": MCP_NAME}


def _remove_skill() -> Step:
    target = _skill_dir()
    if not target.exists():
        return {"name": "skill", "status": "not_installed", "detail": str(target)}
    try:
        shutil.rmtree(target)
    except OSError as err:
        return {"name": "skill", "status": "failed", "detail": str(err)}
    return {"name": "skill", "status": "removed", "detail": str(target)}


def _render_steps(data: dict[str, list[Step]]) -> str:
    lines = [f"  {s['name']}: {s['status']} — {escape(s['detail'])}" for s in data["steps"]]
    return "AssemblyAI coding-agent setup:\n" + "\n".join(lines)


@app.command()
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
        steps = [_install_mcp(scope, force), _install_skill()]
        output.emit({"steps": steps}, _render_steps, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)


@app.command()
def status(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show whether the AssemblyAI MCP server and skill are wired into Claude Code."""

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [_mcp_status(), _skill_status()]
        output.emit({"steps": steps}, _render_steps, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command()
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
        steps = [_remove_mcp(scope), _remove_skill()]
        output.emit({"steps": steps}, _render_steps, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
