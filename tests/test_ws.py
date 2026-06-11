"""Direct tests for the shared WebSocket-session helpers (aai_cli/ws.py).

The agent and TTS sessions both route through these; session-level behavior is
covered in test_agent_session_run.py / test_tts_session.py.
"""

from __future__ import annotations

import logging
import types

from aai_cli.errors import APIError, NotAuthenticated
from aai_cli.ws import (
    WEBSOCKETS_LOGGERS,
    auth_or_api_error,
    is_rejected_key,
    silence_websockets_logging,
)


class _HandshakeRejected(Exception):
    """Mimics websockets' InvalidStatus: a structured HTTP status on ``.response``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"server rejected WebSocket connection: HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


def test_is_rejected_key_false_for_handshake_403():
    # 403 also covers WAF/region/plan blocks, so it must NOT read as a rejected key.
    assert is_rejected_key(_HandshakeRejected(403)) is False


def test_is_rejected_key_true_for_handshake_401():
    assert is_rejected_key(_HandshakeRejected(401)) is True


def test_is_rejected_key_true_for_auth_worded_message():
    assert is_rejected_key(RuntimeError("Unauthorized")) is True


def test_is_rejected_key_false_for_generic_failure():
    assert is_rejected_key(RuntimeError("network unreachable")) is False


def test_auth_or_api_error_maps_rejected_key_to_not_authenticated():
    err = auth_or_api_error(RuntimeError("Unauthorized"), "Could not connect")
    assert isinstance(err, NotAuthenticated)
    assert err.exit_code == 4


def test_auth_or_api_error_wraps_other_failures_with_context():
    err = auth_or_api_error(RuntimeError("network unreachable"), "Could not connect")
    assert isinstance(err, APIError)
    assert err.message == "Could not connect: network unreachable"


def test_silence_websockets_logging_raises_both_loggers_to_critical():
    loggers = [logging.getLogger(name) for name in WEBSOCKETS_LOGGERS]
    previous = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.NOTSET)
        silence_websockets_logging()
        for lg in loggers:
            assert lg.level == logging.CRITICAL
            assert not lg.isEnabledFor(logging.ERROR)
    finally:
        for lg, level in zip(loggers, previous, strict=True):
            lg.setLevel(level)


def test_websockets_logger_names_cover_the_sync_client():
    # The sync client logs through "websockets.client"; the parent covers any
    # future child loggers.
    assert "websockets.client" in WEBSOCKETS_LOGGERS
    assert "websockets" in WEBSOCKETS_LOGGERS
