import subprocess
import sys

import pytest

import assemblyai_cli.main as main_mod


def test_run_exits_clean_on_broken_pipe(monkeypatch):
    """A closed downstream pipe (`| head`) is success, not an error traceback."""

    def boom(*a, **k):
        raise BrokenPipeError

    monkeypatch.setattr(main_mod, "app", boom)
    # Don't dup2 the real stdout fd during the test; just verify the exit contract.
    monkeypatch.setattr("assemblyai_cli.stdio.silence_stdout", lambda: None)
    with pytest.raises(SystemExit) as exc:
        main_mod.run()
    assert exc.value.code == 0


def test_run_passes_through_normal_exit(monkeypatch):
    """Non-pipe exits keep their code (Typer raises SystemExit on normal completion)."""

    def normal(*a, **k):
        raise SystemExit(3)

    monkeypatch.setattr(main_mod, "app", normal)
    with pytest.raises(SystemExit) as exc:
        main_mod.run()
    assert exc.value.code == 3


def test_python_dash_m_entrypoint_runs():
    """`python -m assemblyai_cli` wires up the Typer app (exercises __main__.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "assemblyai_cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "aai" in result.stdout


def test_python_dash_m_version():
    result = subprocess.run(
        [sys.executable, "-m", "assemblyai_cli", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip()  # prints something (the version)
