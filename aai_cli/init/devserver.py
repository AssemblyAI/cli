# aai_cli/init/devserver.py
from __future__ import annotations

from pathlib import Path

from aai_cli import steps
from aai_cli.init import runner


def install_step(target: Path, *, no_install: bool, use_uv: bool) -> steps.Step:
    """Install deps (unless --no-install) and return the report row."""
    if no_install:
        return {"name": "install", "status": "skipped", "detail": "--no-install"}
    setup = runner.run_setup(target, use_uv=use_uv)
    if setup.returncode != 0:
        return {
            "name": "install",
            "status": "failed",
            "detail": (setup.stderr or setup.stdout).strip()[:300],
        }
    return {"name": "install", "status": "installed", "detail": "uv" if use_uv else "venv + pip"}


def dev_command(target: Path, web: list[str], *, use_uv: bool) -> list[str]:
    """The Procfile web process, run in the project venv with live reload.

    The Procfile's `web:` line starts with `python -m uvicorn …`. With uv, run it
    verbatim under `uv run`; without uv, swap a leading `python` for the project's
    venv interpreter so it runs inside the scaffolded `.venv`.
    """
    if use_uv:
        return ["uv", "run", *web, "--reload"]
    argv = list(web)
    if argv and argv[0] == "python":
        argv[0] = str(runner.venv_python(target))
    return [*argv, "--reload"]
