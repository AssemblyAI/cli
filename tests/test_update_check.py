"""Tests for the update-available notifier."""

from __future__ import annotations

import io
import sys
import time
import types

import httpx2
import pytest
from rich.console import Console

from aai_cli import __version__
from aai_cli.core import config
from aai_cli.ui import output, theme, update_check


def test_update_cache_roundtrips(tmp_path, monkeypatch):
    # Isolate config.toml to a temp dir so the real one is never touched.
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)

    assert config.get_update_cache() == (None, None)

    config.set_update_cache(last_check=1718000000.0, latest_version="0.2.0")
    assert config.get_update_cache() == (1718000000.0, "0.2.0")


def test_update_cache_records_check_even_when_version_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    # A failed fetch still records the timestamp (so we don't re-spawn every run)
    # but leaves the version unknown.
    config.set_update_cache(last_check=1718000001.0, latest_version=None)
    assert config.get_update_cache() == (1718000001.0, None)


@pytest.mark.parametrize(
    ("latest", "current", "expected"),
    [
        ("0.2.0", "0.1.0", True),
        ("0.1.1", "0.1.0", True),
        ("1.0.0", "0.9.9", True),
        ("0.1.0", "0.1.0", False),  # equal
        ("0.1.0", "0.2.0", False),  # older
        ("not-a-version", "0.1.0", False),  # unparseable -> never notify
    ],
)
def test_is_newer(latest, current, expected):
    assert update_check.is_newer(latest, current) is expected


@pytest.mark.parametrize(
    ("exe", "expected"),
    [
        ("/opt/homebrew/Cellar/assembly/0.1.0/libexec/bin/python", "brew upgrade assembly"),
        ("/usr/local/Cellar/assembly/0.1.0/libexec/bin/python", "brew upgrade assembly"),
        ("/usr/local/bin/python", "brew upgrade assembly"),
        # pipx/uv upgrade by *distribution* name (aai-cli), not the console command.
        ("/Users/x/.local/pipx/venvs/aai-cli/bin/python", "pipx upgrade aai-cli"),
        ("/Users/x/.local/share/uv/tools/aai-cli/bin/python", "uv tool upgrade aai-cli"),
        ("/usr/bin/python3", ""),  # unknown -> generic (empty)
    ],
)
def test_detect_upgrade_command(exe, expected, monkeypatch):
    # These are macOS-style install paths; pin the platform so the /usr/local
    # Intel-Homebrew heuristic applies (it is gated off on non-macOS).
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable", exe)
    assert update_check.detect_upgrade_command() == expected


def test_usr_local_bin_is_not_homebrew_off_macos(monkeypatch):
    # On Linux, /usr/local/bin/python is a source/manual build, not Homebrew — so we
    # must not tell the user to run `brew upgrade` (which they don't have).
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "executable", "/usr/local/bin/python")
    assert update_check.detect_upgrade_command() == ""


def test_usr_local_bin_is_homebrew_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable", "/usr/local/bin/python")
    assert update_check.detect_upgrade_command() == "brew upgrade assembly"


def _fake_response(payload: dict[str, object]) -> types.SimpleNamespace:
    return types.SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)


def test_fetch_and_cache_writes_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)

    # Untyped capture dict (mirrors the pattern in tests/test_telemetry.py).
    captured = {}

    def fake_get(url, *, headers, timeout, follow_redirects):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["follow_redirects"] = follow_redirects
        return _fake_response({"tag_name": "v0.4.0"})

    monkeypatch.setattr(httpx2, "get", fake_get)

    update_check.fetch_and_cache()

    last_check, latest = config.get_update_cache()
    assert latest == "0.4.0"  # 'v' stripped
    assert last_check is not None
    assert captured["url"] == update_check._RELEASES_URL
    assert "User-Agent" in captured["headers"]
    assert captured["timeout"] == 5.0  # the configured fetch timeout flows through
    assert captured["follow_redirects"] is True  # GitHub's latest-release URL redirects


def test_check_interval_is_24_hours():
    assert update_check._CHECK_INTERVAL_SECONDS == 86400


def test_fetch_and_cache_empty_tag_leaves_version_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    # An empty tag_name is present but unusable: it must NOT be cached as a version.
    monkeypatch.setattr(httpx2, "get", lambda url, **kwargs: _fake_response({"tag_name": ""}))

    update_check.fetch_and_cache()

    _, latest = config.get_update_cache()
    assert latest is None


def test_fetch_and_cache_swallows_errors_but_records_check(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)

    def boom(url, **kwargs):
        raise httpx2.HTTPError("network down")

    monkeypatch.setattr(httpx2, "get", boom)

    update_check.fetch_and_cache()  # must not raise

    last_check, latest = config.get_update_cache()
    assert latest is None  # unknown after a failed fetch
    assert last_check is not None  # but the attempt is recorded


def _tty_console() -> tuple[Console, io.StringIO]:
    # A theme-aware console (so aai.* styles resolve, like the real error_console)
    # that reports as a terminal, with color env pinned (see the
    # rich-color-tests-need-empty-environ project memory) so output is stable.
    # Returns the buffer too: Console.file is typed IO[str] (no .getvalue()).
    buf = io.StringIO()
    return theme.make_console(file=buf, force_terminal=True, width=80, _environ={}), buf


def test_maybe_notify_shows_box_for_newer_cached_version(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    # Cache says a newer version exists, checked just now (so no spawn).
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    monkeypatch.setattr(sys, "executable", "/opt/homebrew/Cellar/assembly/9/libexec/bin/python")

    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    update_check.maybe_notify(json_mode=False)

    out = buf.getvalue()
    assert "Update available" in out
    assert "9.9.9" in out
    assert "brew upgrade assembly" in out  # detected command, not a generic hint


def test_maybe_notify_silent_under_json(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)

    update_check.maybe_notify(json_mode=True)

    assert buf.getvalue() == ""  # JSON mode prints nothing


def test_maybe_notify_silent_when_not_a_tty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    buf = io.StringIO()
    con = theme.make_console(file=buf, force_terminal=False, _environ={})  # not a tty
    monkeypatch.setattr(output, "error_console", con)

    update_check.maybe_notify(json_mode=False)

    assert buf.getvalue() == ""


def test_maybe_notify_silent_in_ci_and_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)

    monkeypatch.setenv("CI", "1")
    update_check.maybe_notify(json_mode=False)
    assert buf.getvalue() == ""

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv(update_check.ENV_DISABLED, "1")
    update_check.maybe_notify(json_mode=False)
    assert buf.getvalue() == ""


def test_maybe_notify_no_box_when_cache_not_newer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version=__version__)  # equal
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    update_check.maybe_notify(json_mode=False)
    assert "Update available" not in buf.getvalue()


def test_maybe_notify_spawns_refresh_only_when_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    con, _buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    spawned: list[bool] = []
    monkeypatch.setattr(update_check, "spawn_refresh", lambda: spawned.append(True))

    # Never checked -> spawn.
    update_check.maybe_notify(json_mode=False)
    assert spawned == [True]
    spawned.clear()

    # Fresh check -> no spawn.
    config.set_update_cache(last_check=time.time(), latest_version=None)
    update_check.maybe_notify(json_mode=False)
    assert spawned == []

    # Stale check (>24h ago) -> spawn.
    config.set_update_cache(
        last_check=time.time() - update_check._CHECK_INTERVAL_SECONDS - 1, latest_version=None
    )
    update_check.maybe_notify(json_mode=False)
    assert spawned == [True]


def test_spawn_refresh_is_detached(monkeypatch):
    # Untyped capture dict (mirrors the pattern in tests/test_telemetry.py).
    calls = {}

    def fake_popen(args, *, stdout, stderr, start_new_session, env):
        calls["args"] = args
        calls["kwargs"] = {
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": start_new_session,
            "env": env,
        }
        return object()

    monkeypatch.setattr("aai_cli.core.procs.subprocess.Popen", fake_popen)
    update_check.spawn_refresh()

    assert calls["args"][:3] == [sys.executable, "-m", "aai_cli"]
    assert calls["args"][3] == "_update-check"
    assert calls["kwargs"]["start_new_session"] is True
    assert calls["kwargs"]["env"][update_check.ENV_DISABLED] == "1"  # child can't re-spawn


def test_notice_appears_after_a_real_command(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from aai_cli.main import app

    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    monkeypatch.setattr(sys, "executable", "/opt/homebrew/Cellar/assembly/9/libexec/bin/python")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)
    # is_terminal is a read-only property, so swap in a tty-reporting console and
    # read its buffer (CliRunner doesn't capture our substituted stderr console).
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)

    # `telemetry status` is a simple, side-effect-free command that runs through
    # run_command — exercising the maybe_notify hook end to end.
    result = CliRunner().invoke(app, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Update available" in buf.getvalue()


def test_no_notice_under_json(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from aai_cli.main import app

    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)

    result = CliRunner().invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert "Update available" not in buf.getvalue()


def test_maybe_notify_generic_hint_when_install_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")  # unknown -> no command
    con, buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    update_check.maybe_notify(json_mode=False)

    out = buf.getvalue()
    assert "Update available" in out
    assert "github.com/AssemblyAI/cli#installation" in out  # docs hint, not a command
    assert "brew upgrade" not in out


def test_fetch_and_cache_no_tag_leaves_version_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(httpx2, "get", lambda url, **kwargs: _fake_response({}))  # no tag_name

    update_check.fetch_and_cache()

    last_check, latest = config.get_update_cache()
    assert latest is None
    assert last_check is not None


def test_fetch_and_cache_swallows_cache_write_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(httpx2, "get", lambda url, **kwargs: _fake_response({"tag_name": "v0.4.0"}))

    def boom(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(config, "set_update_cache", boom)

    update_check.fetch_and_cache()  # the cache-write failure must be swallowed


def test_maybe_notify_swallows_config_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    con, _buf = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    def boom() -> tuple[float | None, str | None]:
        raise OSError("config unreadable")

    monkeypatch.setattr(config, "get_update_cache", boom)

    update_check.maybe_notify(json_mode=False)  # a config read failure must be swallowed
