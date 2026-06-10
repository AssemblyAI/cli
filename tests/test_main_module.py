import subprocess
import sys

import pytest

import aai_cli.main as main_mod


def test_command_line_requests_json_recognizes_every_form():
    f = main_mod._command_line_requests_json
    assert f(["whoami", "--json"])
    assert f(["transcribe", "a.mp3", "-o", "json"])
    assert f(["transcribe", "a.mp3", "--output", "json"])
    assert f(["transcribe", "a.mp3", "--output=json"])
    assert f(["transcribe", "a.mp3", "-ojson"])
    assert f(["whoami", "-j"])  # the short --json alias is recognized too
    # `-o json` is detected even when more tokens follow (pins the 1-element slice width).
    assert f(["transcribe", "-o", "json", "--speaker-labels"])


def test_command_line_requests_json_false_for_text_and_bare():
    f = main_mod._command_line_requests_json
    assert not f(["transcribe", "a.mp3", "-o", "text"])
    assert not f(["transcribe", "a.mp3"])
    assert not f([])


def test_run_exits_clean_on_broken_pipe(monkeypatch):
    """A closed downstream pipe (`| head`) is success, not an error traceback."""

    def boom(*a, **k):
        raise BrokenPipeError

    monkeypatch.setattr(main_mod, "app", boom)
    # Don't dup2 the real stdout fd during the test; just verify the exit contract.
    monkeypatch.setattr("aai_cli.stdio.silence_stdout", lambda: None)
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
    """`python -m aai_cli` wires up the Typer app (exercises __main__.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "aai_cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "aai" in result.stdout


def test_python_dash_m_version():
    result = subprocess.run(
        [sys.executable, "-m", "aai_cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip()  # prints something (the version)


def test_run_converts_click_epipe_exit_to_success(monkeypatch):
    """Typer's vendored Click swallows EPIPE itself (PacifyFlushWrapper + exit 1);
    run() must still honor the closed-pipe-is-success contract."""
    from typer._click.utils import PacifyFlushWrapper

    def epipe_path(*a, **k):
        monkeypatch.setattr(sys, "stdout", PacifyFlushWrapper(sys.stdout))
        raise SystemExit(1)

    monkeypatch.setattr(main_mod, "app", epipe_path)
    monkeypatch.setattr("aai_cli.stdio.silence_stdout", lambda: None)
    with pytest.raises(SystemExit) as exc:
        main_mod.run()
    assert exc.value.code == 0


def test_run_keeps_exit_1_when_stdout_is_not_pacified(monkeypatch):
    """A real failure exit (code 1, untouched stdout) must never be rewritten."""

    def failure(*a, **k):
        raise SystemExit(1)

    monkeypatch.setattr(main_mod, "app", failure)
    with pytest.raises(SystemExit) as exc:
        main_mod.run()
    assert exc.value.code == 1
