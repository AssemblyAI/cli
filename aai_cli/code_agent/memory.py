"""Long-term agent memory (deepagents-code parity).

deepagents' `MemoryMiddleware` loads memory files into the system prompt and lets the
agent persist learnings with `edit_file`. Like skills, it reads through a backend; we
give it its own `FilesystemBackend` rooted at a memories directory under the CLI's
config root, independent of the cwd-scoped file tools, so memory survives across
sessions and projects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.core import env

if TYPE_CHECKING:
    from deepagents.middleware.memory import MemoryMiddleware

_CLAUDE_CONFIG_DIR = "CLAUDE_CONFIG_DIR"


def memory_root() -> Path:
    """Directory where the agent's long-term memory files live (created on demand)."""
    config_dir = env.get(_CLAUDE_CONFIG_DIR)
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "code-memory"


# The single memory file the agent reads and appends learnings to. MemoryMiddleware
# loads each source as a *file* (a directory like "/" makes it raise is_directory), so
# this is a concrete path, not a folder.
_MEMORY_FILE = "memory.md"


def build_memory_middleware(root: Path | None = None) -> MemoryMiddleware:
    """A `MemoryMiddleware` reading/appending a single memory file under ``root``."""
    root = root if root is not None else memory_root()
    root.mkdir(parents=True, exist_ok=True)
    # Touch the file so the very first session has something to load (and a target to
    # append to); an absent source is skipped, but an empty file reads cleanly.
    (root / _MEMORY_FILE).touch(exist_ok=True)

    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.memory import MemoryMiddleware

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    return MemoryMiddleware(backend=backend, sources=[f"/{_MEMORY_FILE}"])
