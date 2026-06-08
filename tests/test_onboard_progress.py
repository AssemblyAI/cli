from __future__ import annotations

from pathlib import Path

import pytest

from aai_cli import config


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    return tmp_path


def test_requests_made_starts_at_zero(tmp_config: Path) -> None:
    assert config.get_requests_made("default") == 0


def test_record_request_increments_and_persists(tmp_config: Path) -> None:
    assert config.record_request("default") == 1
    assert config.record_request("default") == 2
    assert config.get_requests_made("default") == 2
