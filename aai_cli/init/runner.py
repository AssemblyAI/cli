# aai_cli/init/runner.py
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from aai_cli.core.errors import CLIError
from aai_cli.ui import output


def has_uv() -> bool:
    return shutil.which("uv") is not None


def venv_python(target: Path) -> Path:
    if os.name == "nt":
        return target / ".venv" / "Scripts" / "python.exe"
    return target / ".venv" / "bin" / "python"


def env_setup_commands(target: Path, *, use_uv: bool) -> list[list[str]]:
    """Commands (run with cwd=target) to create a venv and install requirements.

    `--allow-existing` keeps the uv path idempotent: `assembly init` creates `.venv`,
    and a later `assembly dev` runs this setup again — without the flag, uv refuses
    with "A virtual environment already exists" instead of reusing it (the stdlib
    `python -m venv` path already reuses an existing venv).
    """
    if use_uv:
        return [
            ["uv", "venv", "--allow-existing"],
            ["uv", "pip", "install", "-r", "requirements.txt"],
        ]
    py = str(venv_python(target))
    return [
        [sys.executable, "-m", "venv", ".venv"],
        [py, "-m", "pip", "install", "-r", "requirements.txt"],
    ]


def serve_command(target: Path, *, port: int, use_uv: bool) -> list[str]:
    if use_uv:
        return ["uv", "run", "uvicorn", "api.index:app", "--port", str(port)]
    return [str(venv_python(target)), "-m", "uvicorn", "api.index:app", "--port", str(port)]


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(preferred: int, *, tries: int = 20) -> int:
    """The preferred port if free, else the next free port; OS-assigned when preferred is 0."""
    if preferred == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
    for candidate in range(preferred, preferred + tries):
        if not _port_open(candidate):
            return candidate
    raise CLIError(
        f"No free port found in {preferred}-{preferred + tries - 1}. "
        "Pass --port to choose another.",
        error_type="port_unavailable",
        exit_code=1,
    )


def wait_for_port(port: int, *, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.2)
    return False


def spawn(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen[str]:
    """Start a process without blocking.

    With `log_path`, the process's stdout+stderr are written to that file (text mode) —
    used to capture cloudflared's output for URL discovery. Without it, stdio is inherited.
    """
    if log_path is not None:
        # The child gets its own dup of the fd once Popen returns, so close the
        # parent's handle straight away instead of leaking it for the (long-lived)
        # process's whole lifetime.
        log = log_path.open("w")
        try:
            return subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log.close()
    return subprocess.Popen(command, cwd=cwd, env=env, text=True)


def run_setup(target: Path, *, use_uv: bool) -> subprocess.CompletedProcess[str]:
    """Run env-setup commands in order; return the first failure or the last success."""
    last = subprocess.CompletedProcess[str](args=[], returncode=0, stdout="", stderr="")
    for cmd in env_setup_commands(target, use_uv=use_uv):
        last = subprocess.run(cmd, cwd=target, capture_output=True, check=False, text=True)
        if last.returncode != 0:
            return last
    return last


def open_app_browser(port: int) -> None:
    """Open the app URL, saying where to point a browser when none can launch.

    `webbrowser.open` returns False on headless boxes (no display/$BROWSER); a
    silent False would leave the user staring at a running server with no URL.
    The hint goes to stderr so stdout stays clean for pipelines.
    """
    url = f"http://localhost:{port}"
    if not webbrowser.open(url):
        output.error_console.print(output.hint(f"Couldn't open a browser — visit {url}"))


def run_server(
    target: Path,
    *,
    command: list[str],
    port: int,
    env: dict[str, str] | None = None,
    open_browser: bool,
) -> int:
    """Run a prebuilt server command, wait for the port, open the browser, block until Ctrl-C.

    Returns the process exit code (0 on a clean Ctrl-C shutdown). `env=None` inherits
    the current environment; pass a full dict (e.g. `{**os.environ, "PORT": ...}`) to override.
    """
    proc = subprocess.Popen(command, cwd=target, env=env)
    try:
        if wait_for_port(port) and open_browser:
            open_app_browser(port)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        return 0
    return proc.returncode


def launch_and_open(target: Path, *, port: int, use_uv: bool, open_browser: bool) -> int:
    """Start the (init) dev server and open the browser; block until Ctrl-C."""
    return run_server(
        target,
        command=serve_command(target, port=port, use_uv=use_uv),
        port=port,
        open_browser=open_browser,
    )
