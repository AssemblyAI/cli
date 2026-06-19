"""The interactive "update now?" prompt that the startup notice offers."""

from __future__ import annotations

import io
import time
import types

from rich.console import Console

from aai_cli.core import config, config_store, procs, stdio
from aai_cli.ui import output, theme, update_check


def _tty_console() -> tuple[Console, io.StringIO]:
    # A theme-aware console reporting as a terminal, color env pinned for stable output
    # (mirrors the helper in test_update_check.py).
    buf = io.StringIO()
    return theme.make_console(file=buf, force_terminal=True, width=80, _environ={}), buf


def test_resolve_upgrade_command_uses_detected_channel(monkeypatch):
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "brew upgrade assembly")
    assert update_check.resolve_upgrade_command() == "brew upgrade assembly"


def test_resolve_upgrade_command_falls_back_to_install_script(monkeypatch):
    # Unknown install channel -> the canonical curl|sh installer, not an empty string.
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "")
    command = update_check.resolve_upgrade_command()
    assert command == update_check._INSTALL_SCRIPT_COMMAND
    assert "install.sh" in command


def test_upgrade_argv_runs_install_script_through_a_shell():
    # The fallback is a pipeline (curl … | sh), so it must go through `sh -c`, not be
    # split into bare argv (which would hand `|` and `sh` to curl as arguments).
    argv = update_check._upgrade_argv(update_check._INSTALL_SCRIPT_COMMAND)
    assert argv == ["sh", "-c", update_check._INSTALL_SCRIPT_COMMAND]


def test_upgrade_argv_splits_package_manager_command():
    assert update_check._upgrade_argv("brew upgrade assembly") == ["brew", "upgrade", "assembly"]


def test_run_foreground_inherits_stdio_and_returns_status(monkeypatch):
    calls = {}

    def fake_run(argv, *, check):
        calls["argv"] = argv
        calls["check"] = check
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr("aai_cli.core.procs.subprocess.run", fake_run)

    assert procs.run_foreground(["brew", "upgrade", "assembly"]) == 7
    assert calls["argv"] == ["brew", "upgrade", "assembly"]
    assert calls["check"] is False  # exit status is inspected, never raised


def _enable_prompt(tmp_path, monkeypatch) -> io.StringIO:
    """Cache a newer version, a tty stderr console, and an interactive stdin so the
    update notice renders and the upgrade prompt is reachable."""
    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)
    monkeypatch.setattr(stdio, "stdin_is_tty", lambda: True)
    return buf


def test_prompt_runs_upgrade_when_confirmed(tmp_path, monkeypatch):
    buf = _enable_prompt(tmp_path, monkeypatch)
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "brew upgrade assembly")

    confirm = {}

    def fake_confirm(text, *, default, err):
        confirm["text"] = text
        confirm["default"] = default
        confirm["err"] = err
        return True

    monkeypatch.setattr(update_check.typer, "confirm", fake_confirm)

    ran = {}

    def fake_run_foreground(argv):
        ran["argv"] = argv
        return 0

    monkeypatch.setattr(procs, "run_foreground", fake_run_foreground)

    update_check.maybe_notify(json_mode=False)

    assert ran["argv"] == ["brew", "upgrade", "assembly"]  # the detected channel ran
    assert "Update now?" in confirm["text"]  # the prompt actually asks
    assert confirm["default"] is False  # default-No: a bare Enter declines
    assert confirm["err"] is True  # prompt rides stderr, like the notice
    out = buf.getvalue()
    assert "Updated to" in out
    assert "9.9.9" in out
    assert "Restart" in out  # tells the user the new binary takes over next run


def test_prompt_skips_upgrade_when_declined(tmp_path, monkeypatch):
    buf = _enable_prompt(tmp_path, monkeypatch)
    monkeypatch.setattr(update_check.typer, "confirm", lambda *a, **k: False)

    ran = []

    def fake_run_foreground(argv):
        ran.append(argv)
        return 0

    monkeypatch.setattr(procs, "run_foreground", fake_run_foreground)

    update_check.maybe_notify(json_mode=False)

    assert ran == []  # declining runs nothing
    assert "Update available" in buf.getvalue()  # the notice still showed


def test_no_upgrade_prompt_when_stdin_not_a_tty(tmp_path, monkeypatch):
    buf = _enable_prompt(tmp_path, monkeypatch)
    monkeypatch.setattr(stdio, "stdin_is_tty", lambda: False)  # piped/redirected stdin

    asked = []
    monkeypatch.setattr(update_check.typer, "confirm", lambda *a, **k: asked.append(True))

    update_check.maybe_notify(json_mode=False)

    assert asked == []  # a non-interactive stdin is never prompted
    assert "Update available" in buf.getvalue()  # but the notice still renders


def test_prompt_reports_failure_when_upgrade_errors(tmp_path, monkeypatch):
    buf = _enable_prompt(tmp_path, monkeypatch)
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "brew upgrade assembly")
    monkeypatch.setattr(update_check.typer, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(procs, "run_foreground", lambda argv: 3)  # non-zero exit

    update_check.maybe_notify(json_mode=False)

    out = buf.getvalue()
    assert "Update failed" in out
    assert "brew upgrade assembly" in out  # the command to re-run by hand


def test_confirm_upgrade_treats_aborted_prompt_as_no(monkeypatch):
    # Ctrl-C (Abort) or Ctrl-D (EOFError) at the prompt must read as "no", never crash.
    for exc in (update_check.typer.Abort, EOFError):

        def boom(*a, _exc=exc, **k):
            raise _exc()

        monkeypatch.setattr(update_check.typer, "confirm", boom)
        assert update_check._confirm_upgrade() is False
