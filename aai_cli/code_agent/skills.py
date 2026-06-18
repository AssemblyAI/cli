"""Import installed agent skills (notably the `assemblyai` skill) into the agent.

`assembly setup` installs skills under the coding-agent config root
(`~/.claude/skills/<skill>/SKILL.md`, honoring `CLAUDE_CONFIG_DIR`). deepagents can
surface skills to the model via progressive disclosure, but its `SkillsMiddleware` reads
them through a backend — and our main file backend is confined to the working directory.
So we give skills their *own* `FilesystemBackend` rooted at the skills directory.

deepagents' stock skills prompt tells the model to open each `SKILL.md` with `read_file`,
but that tool is bound to the cwd-scoped backend and so can't reach a skill living under
`~/.claude/skills` (the model just gets ``File '/aai-cli/SKILL.md' not found``). We close
that gap with a dedicated read-only `read_skill` tool bound to the skills directory, and a
prompt that points the model at it instead of `read_file`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.core import env

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import BaseTool

# Mirrors aai_cli.app.coding_agent.skills_root without importing the app layer (a
# feature slice stays below it): the agent config root, overridable for tests/agents.
_CLAUDE_CONFIG_DIR = "CLAUDE_CONFIG_DIR"

READ_SKILL_TOOL_NAME = "read_skill"

# Skills prompt fragment. Must keep the three slots deepagents substitutes at runtime
# (`{skills_locations}`, `{skills_load_warnings}`, `{skills_list}`); the constructor
# raises if any is missing. The one behavioral change from deepagents' stock prompt is
# steering the model to `read_skill` — skills live outside the cwd sandbox, so the
# ordinary `read_file` tool can't open them.
_SKILLS_PROMPT = """## Skills

You have a library of skills — specialized instructions and workflows for specific tasks.

{skills_locations}{skills_load_warnings}
**Available skills:**

{skills_list}

**How to use a skill (progressive disclosure):** you see each skill's name, description, and
path above, but read its full instructions only when a skill matches the task. Read it with
the `read_skill` tool, passing the path shown above — e.g. `read_skill("/assemblyai/SKILL.md")`
— then follow what it says. Do **not** use `read_file` for these paths: skills live outside the
working directory, so only `read_skill` can reach them."""


def skills_root() -> Path:
    """Directory holding installed skills (one subdir per skill, each with SKILL.md)."""
    config_dir = env.get(_CLAUDE_CONFIG_DIR)
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills"


def _has_skills(root: Path) -> bool:
    """True when at least one ``<root>/<skill>/SKILL.md`` exists."""
    return root.is_dir() and any(child.joinpath("SKILL.md").is_file() for child in root.iterdir())


def _read_skill_file(root: Path, path: str) -> str:
    """Read ``path`` (as surfaced in the skills list) from under ``root``, guarding traversal.

    ``path`` is the backend-virtual path shown in the prompt (e.g. ``/assemblyai/SKILL.md``),
    so it is resolved relative to ``root``. A path that escapes ``root`` (``..`` segments) or
    names a missing file returns an error string the model can recover from rather than raising.
    """
    target = (root / path.lstrip("/")).resolve()
    if not target.is_relative_to(root.resolve()):
        return f"Error: '{path}' is outside the skills directory."
    if not target.is_file():
        return f"Error: skill file '{path}' not found."
    return target.read_text(encoding="utf-8")


def build_skill_reader(root: Path) -> BaseTool:
    """Wrap :func:`_read_skill_file` as the ``read_skill`` tool, bound to ``root``."""
    from langchain_core.tools import tool

    @tool(READ_SKILL_TOOL_NAME)
    def read_skill(path: str) -> str:
        """Read a skill's file (e.g. its SKILL.md) by the path shown in the skills list.
        Use this — not read_file — for any path under the skills library."""
        return _read_skill_file(root, path)

    return read_skill


def build_skills(root: Path | None = None) -> tuple[AgentMiddleware, BaseTool] | None:
    """The skills ``(middleware, read_skill tool)`` pair, or ``None`` if no skills are present.

    Returns ``None`` (rather than an empty middleware) so the caller simply omits both from
    the stack when the user has run no `assembly setup` — the agent then starts with no skills
    section and no `read_skill` tool instead of empty ones. The tool is paired with the
    middleware because the prompt the middleware injects directs the model to it.
    """
    root = root if root is not None else skills_root()
    if not _has_skills(root):
        return None

    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.skills import SkillsMiddleware

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    middleware = SkillsMiddleware(backend=backend, sources=["/"], system_prompt=_SKILLS_PROMPT)
    return middleware, build_skill_reader(root)
