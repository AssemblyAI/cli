import socket
import sys
from pathlib import Path

from aai_cli.init import runner


def test_has_uv_reflects_path(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/uv")
    assert runner.has_uv() is True
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    assert runner.has_uv() is False


def test_venv_python_path_per_platform(monkeypatch):
    target = Path("/proj")
    monkeypatch.setattr(runner.os, "name", "posix")
    assert runner.venv_python(target) == target / ".venv" / "bin" / "python"
    monkeypatch.setattr(runner.os, "name", "nt")
    assert runner.venv_python(target) == target / ".venv" / "Scripts" / "python.exe"


def test_env_setup_commands_uv():
    cmds = runner.env_setup_commands(Path("/proj"), use_uv=True)
    assert cmds == [["uv", "venv"], ["uv", "pip", "install", "-r", "requirements.txt"]]


def test_env_setup_commands_venv():
    target = Path("/proj")
    cmds = runner.env_setup_commands(target, use_uv=False)
    py = str(runner.venv_python(target))
    assert cmds == [
        [sys.executable, "-m", "venv", ".venv"],
        [py, "-m", "pip", "install", "-r", "requirements.txt"],
    ]


def test_serve_command_uv_and_venv():
    target = Path("/proj")
    assert runner.serve_command(target, port=3000, use_uv=True) == [
        "uv", "run", "uvicorn", "api.index:app", "--port", "3000",
    ]
    py = str(runner.venv_python(target))
    assert runner.serve_command(target, port=3000, use_uv=False) == [
        py, "-m", "uvicorn", "api.index:app", "--port", "3000",
    ]


def test_find_free_port_returns_preferred_when_open():
    port = runner.find_free_port(0)  # 0 -> OS assigns a free port
    assert isinstance(port, int) and port > 0


def test_find_free_port_skips_taken_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    taken = s.getsockname()[1]
    s.listen(1)
    try:
        chosen = runner.find_free_port(taken)
        assert chosen != taken
    finally:
        s.close()
