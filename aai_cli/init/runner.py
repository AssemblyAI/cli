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


def has_uv() -> bool:
    return shutil.which("uv") is not None


def venv_python(target: Path) -> Path:
    if os.name == "nt":
        return target / ".venv" / "Scripts" / "python.exe"
    return target / ".venv" / "bin" / "python"


def env_setup_commands(target: Path, *, use_uv: bool) -> list[list[str]]:
    """Commands (run with cwd=target) to create a venv and install requirements."""
    if use_uv:
        return [["uv", "venv"], ["uv", "pip", "install", "-r", "requirements.txt"]]
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
    return preferred


def wait_for_port(port: int, *, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.2)
    return False


def run_setup(target: Path, *, use_uv: bool) -> subprocess.CompletedProcess:
    """Run env-setup commands in order; return the first failure or the last success."""
    last = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    for cmd in env_setup_commands(target, use_uv=use_uv):
        last = subprocess.run(cmd, cwd=target, capture_output=True, text=True)
        if last.returncode != 0:
            return last
    return last


def launch_and_open(target: Path, *, port: int, use_uv: bool, open_browser: bool) -> int:
    """Start the dev server, wait for it, open the browser, and block until Ctrl-C.

    Returns the process exit code (0 on a clean Ctrl-C shutdown).
    """
    proc = subprocess.Popen(serve_command(target, port=port, use_uv=use_uv), cwd=target)
    try:
        if wait_for_port(port) and open_browser:
            webbrowser.open(f"http://localhost:{port}")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        return 0
    return proc.returncode
