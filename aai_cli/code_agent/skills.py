"""Import installed agent skills (notably the `assemblyai` skill) into the agent.

`assembly setup` installs the `assemblyai` skill under the coding-agent config root
(`~/.claude/skills/assemblyai/`, honoring `CLAUDE_CONFIG_DIR`). deepagents can surface
skills to the model via progressive disclosure, but its `SkillsMiddleware` reads them
through a backend — and our main file backend is confined to the working directory.
So we give skills their *own* `FilesystemBackend` rooted at the skills directory and
inject a standalone `SkillsMiddleware`, independent of the cwd-scoped file tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.core import env

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware

# Mirrors aai_cli.app.coding_agent.skills_root without importing the app layer (a
# feature slice stays below it): the agent config root, overridable for tests/agents.
_CLAUDE_CONFIG_DIR = "CLAUDE_CONFIG_DIR"


def skills_root() -> Path:
    """Directory holding installed skills (one subdir per skill, each with SKILL.md)."""
    config_dir = env.get(_CLAUDE_CONFIG_DIR)
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills"


def _has_skills(root: Path) -> bool:
    """True when at least one ``<root>/<skill>/SKILL.md`` exists."""
    return root.is_dir() and any(child.joinpath("SKILL.md").is_file() for child in root.iterdir())


def build_skills_middleware(root: Path | None = None) -> AgentMiddleware | None:
    """A ``SkillsMiddleware`` over the installed skills, or ``None`` if none are present.

    Returns ``None`` (rather than an empty middleware) so the caller simply omits it
    from the stack when the user has run no `assembly setup` — the agent then starts
    with no skills section instead of an empty one.
    """
    root = root if root is not None else skills_root()
    if not _has_skills(root):
        return None

    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.skills import SkillsMiddleware

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    return SkillsMiddleware(backend=backend, sources=["/"])
