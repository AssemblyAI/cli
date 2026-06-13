"""Coding-agent integration probes (docs MCP + skills).

`assembly setup` installs these artifacts and `assembly doctor` reports whether
they are present. Command modules are independent of each other (import-linter
contract), so the constants and presence checks they share live here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MCP_NAME = "assemblyai-docs"
SKILL_NAME = "assemblyai"
CLI_SKILL_NAME = "aai-cli"


def run(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run an agent CLI command (``claude``/``npx``) with output captured and stdin
    closed, returning the finished process — a timeout becomes a synthetic exit 124
    rather than a hang or raise."""
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


def skills_root() -> Path:
    """The agent's skills directory (``$CLAUDE_CONFIG_DIR/skills`` or ``~/.claude/skills``)."""
    # Honor CLAUDE_CONFIG_DIR so install/status/remove agree with the agent's actual
    # config root rather than assuming ~/.claude.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills"


def mcp_present() -> bool:
    """Whether the docs MCP is registered. Callers must check `claude` is on PATH."""
    return run(["claude", "mcp", "get", MCP_NAME]).returncode == 0


def skill_dir() -> Path:
    """Where the published ``assemblyai`` skill installs."""
    return skills_root() / SKILL_NAME


def skill_installed() -> bool:
    """Whether the ``assemblyai`` skill is present on disk (its SKILL.md exists)."""
    return (skill_dir() / "SKILL.md").exists()


def cli_skill_dir() -> Path:
    """Where the bundled ``aai-cli`` skill installs."""
    return skills_root() / CLI_SKILL_NAME


def cli_skill_installed() -> bool:
    """Whether the bundled ``aai-cli`` skill is present on disk (its SKILL.md exists)."""
    return (cli_skill_dir() / "SKILL.md").exists()


def missing_components() -> list[str]:
    """Names of the `assembly setup install` artifacts that are not yet installed.

    Probes the docs MCP via the `claude` CLI, so callers must check `claude` is on
    PATH first.
    """
    missing: list[str] = []
    if not mcp_present():
        missing.append("docs MCP")
    if not skill_installed():
        missing.append("assemblyai skill")
    if not cli_skill_installed():
        missing.append("aai-cli skill")
    return missing
