# aai_cli/init/scaffold.py
from __future__ import annotations

import os
import stat
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.core.errors import CLIError
from aai_cli.init import templates

if TYPE_CHECKING:
    # Annotations only (PEP 563 strings), so no runtime import — `Traversable`'s
    # module location differs across 3.10/3.11 but that never matters at runtime.
    # Import from importlib.abc (not importlib.resources.abc): that is the protocol
    # variant `resources.files()` is typed to return, so the annotation matches.
    from importlib.abc import Traversable

PLACEHOLDER_KEY = "your_assemblyai_api_key_here"

# Template files stored under plain names -> their real dotted names on copy.
_DOTFILE_RENAMES = {
    "gitignore": ".gitignore",
    "env.example": ".env.example",
    "dockerignore": ".dockerignore",
}

# Never copy build/test detritus into the user's fresh project. (Loading a template's
# api/index.py during our own tests leaves a __pycache__ next to it.)
_SKIP_NAMES = {"__pycache__"}


def _template_root(template: str) -> Traversable:
    if not templates.is_template(template):
        raise CLIError(
            f"Unknown template {template!r}. Choose one of: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="unknown_template",
            exit_code=1,
        )
    # Navigate from the `aai_cli.init` package (templates/ has no __init__.py, so it
    # is not itself an importable package).
    root = resources.files("aai_cli.init") / "templates" / template
    # Defense in depth: the registry should only list shipped templates, but if it ever
    # drifts ahead of the on-disk directories, fail cleanly instead of with a traceback.
    if not root.is_dir():
        raise CLIError(
            f"Template {template!r} is registered but its files are missing. "
            "This is a packaging bug — please report it.",
            error_type="template_missing",
            exit_code=1,
        )
    return root


def target_conflict(target: Path) -> bool:
    """True when the target exists and is a non-empty directory."""
    return target.is_dir() and any(target.iterdir())


def existing_env_key(target: Path) -> str | None:
    """The real API key already configured in ``target/.env``, or None.

    Re-scaffolding (``assembly init --force``) rewrites ``.env``; when no key resolves
    for the new write, blindly writing the placeholder would silently wipe a key the
    user already configured. Returns None for a missing ``.env``, a blank value, or
    the placeholder itself — only a configured real key is worth preserving.
    """
    env_path = target / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith("ASSEMBLYAI_API_KEY="):
            value = line.removeprefix("ASSEMBLYAI_API_KEY=").strip()
            if value and value != PLACEHOLDER_KEY:
                return value
    return None


def _copy_tree(node: Traversable, dest: Path) -> None:
    for child in node.iterdir():
        if child.name in _SKIP_NAMES or child.name.endswith(".pyc"):
            continue
        name = _DOTFILE_RENAMES.get(child.name, child.name)
        out = dest / name
        if child.is_dir():
            # parents=True is an equivalent mutant here: the walk always creates a
            # node's parent before descending, so `dest` (and `out.parent`) already
            # exists. exist_ok is exercised by the idempotent re-scaffold test.
            out.mkdir(parents=True, exist_ok=True)  # pragma: no mutate
            _copy_tree(child, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)  # pragma: no mutate
            out.write_bytes(child.read_bytes())


def scaffold(
    template: str,
    target: Path,
    *,
    api_key: str | None,
    env_vars: dict[str, str] | None = None,
) -> Path:
    """Copy the template into `target` and write `.env`. Returns `target`.

    `env_vars` (the active environment's hosts) are appended to `.env` so the generated
    app targets the same AssemblyAI environment the key was minted for — otherwise a
    sandbox key would be rejected by the production defaults the templates fall back to.
    """
    root = _template_root(template)
    target.mkdir(parents=True, exist_ok=True)
    _copy_tree(root, target)
    lines = [
        f"ASSEMBLYAI_API_KEY={api_key or PLACEHOLDER_KEY}",
        *(f"{k}={v}" for k, v in (env_vars or {}).items()),
    ]
    env_path = target / ".env"
    # The .env holds the real API key, so create it readable/writable by the owner
    # only (0600) instead of the umask default (commonly 0644) — otherwise the key
    # would be world/group-readable on a shared host. Open with the 0600 mode so the
    # secret is never briefly world-readable; the explicit chmod then also tightens an
    # existing file when `assembly init --force` overwrites one (O_CREAT's mode is ignored
    # for a file that already exists).
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return target
