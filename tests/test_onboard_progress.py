from __future__ import annotations

from aai_cli import config


def test_requests_made_starts_at_zero() -> None:
    assert config.get_requests_made("default") == 0


def test_record_request_increments_and_persists() -> None:
    assert config.record_request("default") == 1
    assert config.record_request("default") == 2
    assert config.get_requests_made("default") == 2
