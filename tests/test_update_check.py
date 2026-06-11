"""Tests for the update-available notifier."""

from __future__ import annotations

from aai_cli import config


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
