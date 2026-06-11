import shlex
import subprocess
from pathlib import Path

from aai_cli.init import devserver


def test_install_step_skipped():
    step = devserver.install_step(Path("/proj"), no_install=True, use_uv=True)
    assert step == {"name": "install", "status": "skipped", "detail": "--no-install"}


def test_install_step_installed_uv(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    step = devserver.install_step(Path("/proj"), no_install=False, use_uv=True)
    assert step == {"name": "install", "status": "installed", "detail": "uv"}


def test_install_step_installed_venv(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    step = devserver.install_step(Path("/proj"), no_install=False, use_uv=False)
    assert step["detail"] == "venv + pip"


def test_install_step_failed_uses_stderr(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "out", "the-error"),
    )
    step = devserver.install_step(Path("/proj"), no_install=False, use_uv=True)
    assert step["status"] == "failed"
    assert step["detail"] == "the-error"


def test_install_step_failed_truncates_detail(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "x" * 500),
    )
    step = devserver.install_step(Path("/proj"), no_install=False, use_uv=True)
    assert len(step["detail"]) == 300


def test_dev_command_uv():
    cmd = devserver.dev_command(
        Path("/proj"), ["python", "-m", "uvicorn", "api.index:app"], use_uv=True
    )
    assert cmd == [
        "uv",
        "run",
        "python",
        "-m",
        "uvicorn",
        "api.index:app",
        "--host",
        "127.0.0.1",
        "--reload",
    ]


def test_dev_command_venv_swaps_python():
    from aai_cli.init import runner

    cmd = devserver.dev_command(
        Path("/proj"), ["python", "-m", "uvicorn", "api.index:app"], use_uv=False
    )
    assert cmd == [
        str(runner.venv_python(Path("/proj"))),
        "-m",
        "uvicorn",
        "api.index:app",
        "--host",
        "127.0.0.1",
        "--reload",
    ]


def test_dev_command_venv_leaves_non_python_first_token():
    # The `python`-swap only fires on a leading `python`; anything else passes through
    # (covers the False branch of the swap condition).
    cmd = devserver.dev_command(Path("/proj"), ["uvicorn", "api.index:app"], use_uv=False)
    assert cmd == ["uvicorn", "api.index:app", "--host", "127.0.0.1", "--reload"]


# The wildcard host exactly as the template Procfile spells it. Assembled via
# shlex (instead of a bare "0.0.0.0" literal) so ruff's S104 binding lint, which
# flags the standalone literal, stays meaningful in this file.
_PROCFILE_WEB = shlex.split("python -m uvicorn api.index:app --host 0.0.0.0 --port 3000")
WILDCARD_HOST = _PROCFILE_WEB[_PROCFILE_WEB.index("--host") + 1]


def test_dev_command_rewrites_procfile_host_to_loopback():
    # The template Procfile binds 0.0.0.0 (right for deploy targets); a local dev run
    # must rewrite it to loopback so the server (and the key in .env) never faces the LAN.
    cmd = devserver.dev_command(Path("/proj"), list(_PROCFILE_WEB), use_uv=True)
    assert WILDCARD_HOST not in cmd
    host_value = cmd[cmd.index("--host") + 1]
    assert host_value == "127.0.0.1"
    # Everything else from the Procfile line survives, in order, plus --reload.
    assert cmd == [
        "uv",
        "run",
        "python",
        "-m",
        "uvicorn",
        "api.index:app",
        "--host",
        "127.0.0.1",
        "--port",
        "3000",
        "--reload",
    ]


def test_dev_command_does_not_mutate_caller_argv():
    web = list(_PROCFILE_WEB)
    devserver.dev_command(Path("/proj"), web, use_uv=False)
    assert web == _PROCFILE_WEB


def test_dev_command_explicit_host_passes_through():
    # `assembly dev --host 0.0.0.0` is the deliberate opt-in to LAN exposure.
    cmd = devserver.dev_command(Path("/proj"), list(_PROCFILE_WEB), use_uv=True, host=WILDCARD_HOST)
    assert cmd[cmd.index("--host") + 1] == WILDCARD_HOST


def test_override_host_handles_equals_form():
    argv = devserver._override_host(
        ["uvicorn", "app", f"--host={WILDCARD_HOST}", "--port", "1"], "127.0.0.1"
    )
    assert argv == ["uvicorn", "app", "--host=127.0.0.1", "--port", "1"]


def test_override_host_tolerates_trailing_host_flag():
    # A malformed line ending in a bare `--host` has no value to rewrite; the
    # override appends an explicit bind instead of reading past the end.
    argv = devserver._override_host(["uvicorn", "app", "--host"], "127.0.0.1")
    assert argv == ["uvicorn", "app", "--host", "--host", "127.0.0.1"]


def test_override_host_appends_when_absent():
    # A Procfile line with no --host would otherwise rely on uvicorn's default;
    # the bind is made explicit so the printed URL always matches reality.
    argv = devserver._override_host(["uvicorn", "app"], "127.0.0.1")
    assert argv == ["uvicorn", "app", "--host", "127.0.0.1"]


def test_local_host_constant_is_loopback():
    assert devserver.LOCAL_HOST == "127.0.0.1"
