# aai_cli/init/scaffold.py
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.errors import CLIError
from aai_cli.init import templates

if TYPE_CHECKING:
    # Annotations only (PEP 563 strings), so no runtime import — `Traversable`'s
    # module location differs across 3.10/3.11 but that never matters at runtime.
    from importlib.resources.abc import Traversable

PLACEHOLDER_KEY = "your_assemblyai_api_key_here"

# Template files stored under plain names -> their real dotted names on copy.
_DOTFILE_RENAMES = {"gitignore": ".gitignore", "env.example": ".env.example"}

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


def _copy_tree(node: Traversable, dest: Path) -> None:
    for child in node.iterdir():
        if child.name in _SKIP_NAMES or child.name.endswith(".pyc"):
            continue
        name = _DOTFILE_RENAMES.get(child.name, child.name)
        out = dest / name
        if child.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            _copy_tree(child, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(child.read_bytes())


def scaffold(
    template: str, target: Path, *, api_key: str | None, base_url: str | None = None
) -> Path:
    """Copy the template into `target` and write `.env`. Returns `target`.

    `base_url`, when given, pins the generated app to the same AssemblyAI environment
    the key was minted for (e.g. a sandbox) — otherwise a sandbox key would be rejected
    by the production default.
    """
    root = _template_root(template)
    target.mkdir(parents=True, exist_ok=True)
    _copy_tree(root, target)
    lines = [f"ASSEMBLYAI_API_KEY={api_key or PLACEHOLDER_KEY}"]
    if base_url:
        lines.append(f"ASSEMBLYAI_BASE_URL={base_url}")
    (target / ".env").write_text("\n".join(lines) + "\n")
    return target
