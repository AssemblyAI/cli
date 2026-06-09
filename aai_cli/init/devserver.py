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

    In the no-uv branch `web[0]` must be a `python -m`-runnable module; every current
    template's `web:` line starts with `uvicorn`.
    """
    prefix = ["uv", "run"] if use_uv else [str(runner.venv_python(target)), "-m"]
    return [*prefix, *web, "--reload"]
