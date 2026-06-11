import socket
import subprocess
import sys
from pathlib import Path

import pytest

from aai_cli.errors import CLIError
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
    # --allow-existing keeps re-running setup idempotent: `assembly init` creates .venv,
    # and `assembly dev` in that project must reuse it, not fail on "already exists".
    cmds = runner.env_setup_commands(Path("/proj"), use_uv=True)
    assert cmds == [
        ["uv", "venv", "--allow-existing"],
        ["uv", "pip", "install", "-r", "requirements.txt"],
    ]


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
        "uv",
        "run",
        "uvicorn",
        "api.index:app",
        "--port",
        "3000",
    ]
    py = str(runner.venv_python(target))
    assert runner.serve_command(target, port=3000, use_uv=False) == [
        py,
        "-m",
        "uvicorn",
        "api.index:app",
        "--port",
        "3000",
    ]


# These two bind a real loopback socket, so they opt back into sockets past the
# suite-wide --disable-socket (see pyproject pytest config); 127.0.0.1 only, so the
# external-network block stays in force. test_find_free_port_raises_when_all_taken
# monkeypatches the probe and needs no socket.
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_find_free_port_returns_preferred_when_open():
    port = runner.find_free_port(0)  # 0 -> OS assigns a free port
    # A real OS-assigned ephemeral port, not a low/privileged one: pins the bind to
    # port 0 (binding to 1 would yield 1, or fail outright as non-root).
    assert isinstance(port, int) and port > 1024


@pytest.mark.allow_hosts(["127.0.0.1"])
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


def test_find_free_port_raises_when_all_taken(monkeypatch):
    monkeypatch.setattr(runner, "_port_open", lambda port: True)  # every port "in use"
    with pytest.raises(CLIError) as exc:
        runner.find_free_port(5000, tries=3)
    assert exc.value.error_type == "port_unavailable"
    assert exc.value.exit_code == 1
    # The message names the exact inclusive range probed: preferred .. preferred+tries-1.
    assert "5000-5002" in str(exc.value)


def test_wait_for_port_returns_true_when_port_opens(monkeypatch):
    calls = {"n": 0}
    slept = []

    def fake_open(port):
        calls["n"] += 1
        return calls["n"] >= 2  # closed on first poll, open on the second

    monkeypatch.setattr(runner, "_port_open", fake_open)
    monkeypatch.setattr(runner.time, "sleep", slept.append)
    assert runner.wait_for_port(3000, timeout=5.0) is True
    assert calls["n"] >= 2
    assert slept == [0.2]  # polls once at the 0.2s interval before the port opens


def test_wait_for_port_returns_false_on_timeout(monkeypatch):
    monkeypatch.setattr(runner, "_port_open", lambda port: False)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    # monotonic jumps past the deadline so the loop exits without the port opening.
    ticks = iter([0.0, 100.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    assert runner.wait_for_port(3000, timeout=1.0) is False


def test_run_setup_returns_last_success(monkeypatch):
    ran = []
    seen = {}

    def fake_run(cmd, cwd, capture_output, check, text):
        ran.append(cmd)
        seen.update(capture_output=capture_output, check=check, text=text)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_setup(Path("/proj"), use_uv=True)
    assert result.returncode == 0
    assert len(ran) == 2  # both env-setup commands ran
    # Output is captured as text, and a failing command is returned (not raised):
    # check must stay False so run_setup can report the failure itself.
    assert seen == {"capture_output": True, "check": False, "text": True}


def test_run_setup_with_no_commands_returns_success_sentinel(monkeypatch):
    # With no env-setup commands the seeded CompletedProcess is returned unchanged, so
    # an empty plan reads as success (returncode 0), never a spurious failure.
    monkeypatch.setattr(runner, "env_setup_commands", lambda *a, **k: [])
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("no command should run")
    )
    result = runner.run_setup(Path("/proj"), use_uv=True)
    assert result.returncode == 0


def test_run_setup_stops_at_first_failure(monkeypatch):
    ran = []

    def fake_run(cmd, cwd, capture_output, check, text):
        ran.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_setup(Path("/proj"), use_uv=True)
    assert result.returncode == 1
    assert result.stderr == "boom"
    assert len(ran) == 1  # short-circuits after the first failing command


class _FakeProc:
    def __init__(self, *, returncode=0, wait_raises=None):
        self.returncode = returncode
        self._wait_raises = wait_raises
        self.waited = 0
        self.terminated = False

    def wait(self):
        self.waited += 1
        if self._wait_raises and self.waited == 1:
            raise self._wait_raises
        return self.returncode

    def terminate(self):
        self.terminated = True


def test_launch_and_open_opens_browser_and_returns_exit_code(monkeypatch):
    proc = _FakeProc(returncode=0)
    opened = {}
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(runner, "wait_for_port", lambda port: True)
    monkeypatch.setattr(runner.webbrowser, "open", lambda url: opened.setdefault("url", url))
    rc = runner.launch_and_open(Path("/proj"), port=3000, use_uv=True, open_browser=True)
    assert rc == 0
    assert opened["url"] == "http://localhost:3000"


def test_launch_and_open_skips_browser_when_disabled(monkeypatch):
    proc = _FakeProc(returncode=2)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(runner, "wait_for_port", lambda port: True)

    def boom(url):
        raise AssertionError("browser should not open")

    monkeypatch.setattr(runner.webbrowser, "open", boom)
    rc = runner.launch_and_open(Path("/proj"), port=3000, use_uv=True, open_browser=False)
    assert rc == 2


def test_launch_and_open_handles_keyboard_interrupt(monkeypatch):
    proc = _FakeProc(wait_raises=KeyboardInterrupt())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(runner, "wait_for_port", lambda port: True)
    monkeypatch.setattr(runner.webbrowser, "open", lambda url: None)
    rc = runner.launch_and_open(Path("/proj"), port=3000, use_uv=True, open_browser=False)
    assert rc == 0  # clean Ctrl-C shutdown
    assert proc.terminated is True


def test_spawn_inherits_stdio_without_log(monkeypatch):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc(returncode=0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    runner.spawn(["echo", "hi"], cwd=Path("/proj"), env={"A": "B"})
    assert captured["cmd"] == ["echo", "hi"]
    assert captured["kwargs"]["cwd"] == Path("/proj")
    assert captured["kwargs"]["env"] == {"A": "B"}
    assert captured["kwargs"]["text"] is True
    assert "stdout" not in captured["kwargs"]  # inherited


def test_spawn_writes_to_log_when_given(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc(returncode=0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    log = tmp_path / "cf.log"
    runner.spawn(["cloudflared"], cwd=tmp_path, log_path=log)
    assert captured["kwargs"]["stderr"] is runner.subprocess.STDOUT
    assert captured["kwargs"]["text"] is True
    stdout = captured["kwargs"]["stdout"]
    # spawn writes the child's stdout to the log file...
    assert stdout.name == str(log)
    # ...and closes the parent's handle once Popen returns (the child keeps its dup),
    # so the file descriptor isn't leaked for the process's whole lifetime.
    assert stdout.closed is True


def test_run_server_passes_command_and_env(monkeypatch):
    captured = {}
    proc = _FakeProc(returncode=0)

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        return proc

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "wait_for_port", lambda port: True)
    monkeypatch.setattr(runner.webbrowser, "open", lambda url: None)
    rc = runner.run_server(
        Path("/proj"), command=["uvicorn", "x"], port=3000, env={"PORT": "3000"}, open_browser=False
    )
    assert rc == 0
    assert captured["cmd"] == ["uvicorn", "x"]
    assert captured["env"] == {"PORT": "3000"}
    assert captured["cwd"] == Path("/proj")
