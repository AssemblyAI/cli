"""Install/status/remove steps for `assembly setup`, shared with onboarding.

Command modules are import-linter-independent, so the step implementations live
here in the core layer; ``commands/setup.py`` drives them from the CLI and the
onboarding wizard (``onboard/sections.py``) reuses the install steps directly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli import coding_agent
from aai_cli.steps import Step, render_steps

if TYPE_CHECKING:
    # Annotation only (PEP 563 string), so no runtime import. Import from
    # importlib.abc — that is the protocol `resources.files()` is typed to return.
    from importlib.abc import Traversable

MCP_URL = "https://mcp.assemblyai.com/docs"
SKILL_REPO = "AssemblyAI/assemblyai-skill"
_STEPS_HEADING = "AssemblyAI coding-agent setup:"

# The subprocess wrapper, artifact names, and presence probes are shared with
# `assembly doctor`, so they live in aai_cli.coding_agent; the names below keep
# this module's call sites stable.
MCP_NAME = coding_agent.MCP_NAME
_run = coding_agent.run
_mcp_present = coding_agent.mcp_present
_skill_dir = coding_agent.skill_dir
_skill_installed = coding_agent.skill_installed
_cli_skill_dir = coding_agent.cli_skill_dir
_cli_skill_installed = coding_agent.cli_skill_installed


def _proc_detail(proc: subprocess.CompletedProcess[str]) -> str:
    """The error text from a finished process: stderr if present, else stdout."""
    return (proc.stderr or proc.stdout).strip()


# --- docs MCP (registered via the `claude` CLI) ------------------------------


def install_mcp(scope: str, *, force: bool) -> Step:
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


def mcp_status() -> Step:
    if shutil.which("claude") is None:
        return {"name": "mcp", "status": "unknown", "detail": "Claude Code not found"}
    present = _mcp_present()
    return {
        "name": "mcp",
        "status": "installed" if present else "not_installed",
        "detail": MCP_NAME,
    }


def remove_mcp(scope: str | None) -> Step:
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


def install_skill(*, force: bool) -> Step:
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


def skill_status() -> Step:
    return {
        "name": "skill",
        "status": "installed" if _skill_installed() else "not_installed",
        "detail": str(_skill_dir()),
    }


def remove_skill() -> Step:
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


def _bundled_cli_skill() -> Traversable:
    # Ships inside the wheel (force-included via [tool.hatch.build.targets.wheel]
    # artifacts). skills/ has no __init__.py, so navigate from the aai_cli package.
    from importlib import resources

    return resources.files("aai_cli") / "skills" / coding_agent.CLI_SKILL_NAME


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


def install_cli_skill(*, force: bool) -> Step:
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


def cli_skill_status() -> Step:
    return {
        "name": "aai-cli skill",
        "status": "installed" if _cli_skill_installed() else "not_installed",
        "detail": str(_cli_skill_dir()),
    }


def remove_cli_skill() -> Step:
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


def render(data: dict[str, list[Step]]) -> str:
    return render_steps(data["steps"], heading=_STEPS_HEADING)
