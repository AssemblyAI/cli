"""Tests for the update-available notifier."""

from __future__ import annotations

import sys

import pytest

from aai_cli import config, update_check


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
