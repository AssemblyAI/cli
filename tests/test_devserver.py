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
    assert cmd == ["uv", "run", "python", "-m", "uvicorn", "api.index:app", "--reload"]


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
        "--reload",
    ]


def test_dev_command_venv_leaves_non_python_first_token():
    # The `python`-swap only fires on a leading `python`; anything else passes through
    # (covers the False branch of the swap condition).
    cmd = devserver.dev_command(Path("/proj"), ["uvicorn", "api.index:app"], use_uv=False)
    assert cmd == ["uvicorn", "api.index:app", "--reload"]
