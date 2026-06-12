# aai_cli/init/devserver.py
from __future__ import annotations

from pathlib import Path

from aai_cli import output, steps
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


def notify_port_change(requested: int, chosen: int, *, json_mode: bool, quiet: bool) -> None:
    """One stderr line when the requested port was busy and a neighbor was bound.

    `assembly dev`/`assembly share` silently substituting a free port would leave the
    user pointing tools at a dead port. Port 0 means "any free port", so no notice
    there, and ``--quiet`` suppresses it.
    """
    if quiet or requested in (0, chosen):
        return
    output.emit_warning(f"Port {requested} is in use; using {chosen}.", json_mode=json_mode)


# Local dev binds the loopback interface only. The template Procfile says
# `--host 0.0.0.0` — correct for the deploy targets (Railway/Fly route traffic into
# the container) but wrong for `assembly dev`/`assembly share`: the .env beside it holds a real
# API key, so the dev server must not listen on every interface of the machine.
LOCAL_HOST = "127.0.0.1"


def _override_host(argv: list[str], host: str) -> list[str]:
    """Rewrite (or add) the uvicorn ``--host`` argument so the server binds `host`."""
    out = list(argv)
    for index, arg in enumerate(out):
        if arg == "--host" and index + 1 < len(out):
            out[index + 1] = host
            return out
        if arg.startswith("--host="):
            out[index] = f"--host={host}"
            return out
    return [*out, "--host", host]


def dev_command(target: Path, web: list[str], *, use_uv: bool, host: str = LOCAL_HOST) -> list[str]:
    """The Procfile web process, run in the project venv with live reload.

    The Procfile's `web:` line starts with `python -m uvicorn …`. With uv, run it
    under `uv run`; without uv, swap a leading `python` for the project's venv
    interpreter so it runs inside the scaffolded `.venv`. In both cases the
    Procfile's `--host 0.0.0.0` is overridden to `host` (loopback by default) so a
    local dev run never exposes the server — and the key in `.env` — to the LAN.
    """
    argv = _override_host(web, host)
    if use_uv:
        return ["uv", "run", *argv, "--reload"]
    if argv and argv[0] == "python":
        argv[0] = str(runner.venv_python(target))
    return [*argv, "--reload"]
