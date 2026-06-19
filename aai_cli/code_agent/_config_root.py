"""The coding-agent config root, shared by the skills and memory backends.

`assembly setup` and the agent's middleware both anchor their on-disk state under
the coding-agent config root (`$CLAUDE_CONFIG_DIR` or `~/.claude`). Skills and
long-term memory each root their own `FilesystemBackend` there, so the resolution
lives here once rather than being duplicated per backend.

Mirrors `aai_cli.app.coding_agent.skills_root`'s root resolution without importing
the app layer (a feature slice stays below it).
"""

from __future__ import annotations

from pathlib import Path

from aai_cli.core import env

_CLAUDE_CONFIG_DIR = "CLAUDE_CONFIG_DIR"


def claude_config_root() -> Path:
    """The coding-agent config root: ``$CLAUDE_CONFIG_DIR`` if set, else ``~/.claude``."""
    config_dir = env.get(_CLAUDE_CONFIG_DIR)
    return Path(config_dir) if config_dir else Path.home() / ".claude"
