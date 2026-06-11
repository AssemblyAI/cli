"""Tests for the update-available notifier."""

from __future__ import annotations

import io
import sys
import time
import types

import httpx2
import pytest
from rich.console import Console

from aai_cli import __version__, config, output, theme, update_check


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
        ("/Users/x/.local/pipx/venvs/aai-cli/bin/python", "pipx upgrade assembly"),
        ("/Users/x/.local/share/uv/tools/aai-cli/bin/python", "uv tool upgrade assembly"),
        ("/usr/bin/python3", ""),  # unknown -> generic (empty)
    ],
)
def test_detect_upgrade_command(exe, expected, monkeypatch):
    monkeypatch.setattr(sys, "executable", exe)
    assert update_check.detect_upgrade_command() == expected


def _fake_response(payload: dict[str, object]) -> types.SimpleNamespace:
    return types.SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)


def test_fetch_and_cache_writes_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)

    captured: dict[str, object] = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _fake_response({"tag_name": "v0.4.0"})

    monkeypatch.setattr(httpx2, "get", fake_get)

    update_check.fetch_and_cache()

    last_check, latest = config.get_update_cache()
    assert latest == "0.4.0"  # 'v' stripped
    assert last_check is not None
    assert captured["url"] == update_check._RELEASES_URL
    assert "User-Agent" in captured["headers"]


def test_fetch_and_cache_swallows_errors_but_records_check(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)

    def boom(url, **kwargs):
        raise httpx2.HTTPError("network down")

    monkeypatch.setattr(httpx2, "get", boom)

    update_check.fetch_and_cache()  # must not raise

    last_check, latest = config.get_update_cache()
    assert latest is None  # unknown after a failed fetch
    assert last_check is not None  # but the attempt is recorded


def _tty_console() -> Console:
    # A theme-aware console (so aai.* styles resolve, like the real error_console)
    # that reports as a terminal, with color env pinned (see the
    # rich-color-tests-need-empty-environ project memory) so output is stable.
    return theme.make_console(file=io.StringIO(), force_terminal=True, width=80, _environ={})


def test_maybe_notify_shows_box_for_newer_cached_version(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    # Cache says a newer version exists, checked just now (so no spawn).
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    monkeypatch.setattr(sys, "executable", "/opt/homebrew/Cellar/assembly/9/libexec/bin/python")

    con = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    update_check.maybe_notify(json_mode=False)

    out = con.file.getvalue()
    assert "Update available" in out
    assert "9.9.9" in out
    assert "brew upgrade assembly" in out  # detected command, not a generic hint


def test_maybe_notify_silent_under_json(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con = _tty_console()
    monkeypatch.setattr(output, "error_console", con)

    update_check.maybe_notify(json_mode=True)

    assert con.file.getvalue() == ""  # JSON mode prints nothing


def test_maybe_notify_silent_when_not_a_tty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con = theme.make_console(file=io.StringIO(), force_terminal=False, _environ={})  # not a tty
    monkeypatch.setattr(output, "error_console", con)

    update_check.maybe_notify(json_mode=False)

    assert con.file.getvalue() == ""


def test_maybe_notify_silent_in_ci_and_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version="9.9.9")
    con = _tty_console()
    monkeypatch.setattr(output, "error_console", con)

    monkeypatch.setenv("CI", "1")
    update_check.maybe_notify(json_mode=False)
    assert con.file.getvalue() == ""

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv(update_check.ENV_DISABLED, "1")
    update_check.maybe_notify(json_mode=False)
    assert con.file.getvalue() == ""


def test_maybe_notify_no_box_when_cache_not_newer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    config.set_update_cache(last_check=time.time(), latest_version=__version__)  # equal
    con = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    update_check.maybe_notify(json_mode=False)
    assert "Update available" not in con.file.getvalue()


def test_maybe_notify_spawns_refresh_only_when_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    con = _tty_console()
    monkeypatch.setattr(output, "error_console", con)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.ENV_DISABLED, raising=False)

    spawned: list[bool] = []
    monkeypatch.setattr(update_check, "spawn_refresh", lambda: spawned.append(True))

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
    calls: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(update_check.subprocess, "Popen", fake_popen)
    update_check.spawn_refresh()

    assert calls["args"][:3] == [sys.executable, "-m", "aai_cli"]
    assert calls["args"][3] == "_update-check"
    assert calls["kwargs"]["start_new_session"] is True
    assert calls["kwargs"]["env"][update_check.ENV_DISABLED] == "1"  # child can't re-spawn
