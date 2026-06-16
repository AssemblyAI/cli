"""Direct tests for the shared realtime-session hygiene helpers
(aai_cli/streaming/diagnostics.py): library-logger silencing and the
actionable classification of rejected WebSocket handshakes."""

from __future__ import annotations

import logging
import types
from collections.abc import Callable

import pytest

from aai_cli.core import debuglog
from aai_cli.core.errors import APIError, NotAuthenticated
from aai_cli.core.ws import WEBSOCKETS_LOGGERS
from aai_cli.streaming.diagnostics import (
    SDK_STREAMING_LOGGER,
    handshake_error,
    handshake_suggestion,
    open_authorized_ws,
    silence_streaming_logging,
)

# The logger the assemblyai SDK's sync streaming client actually emits through.
_SDK_CLIENT_LOGGER = "assemblyai.streaming.v3.client"


class _Spy(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def reset_levels():
    names = (SDK_STREAMING_LOGGER, _SDK_CLIENT_LOGGER, *WEBSOCKETS_LOGGERS)
    previous = {name: logging.getLogger(name).level for name in names}
    for name in names:
        logging.getLogger(name).setLevel(logging.NOTSET)
    yield
    for name, level in previous.items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.usefixtures("reset_levels")
def test_silence_streaming_logging_raises_sdk_and_websockets_loggers():
    silence_streaming_logging()
    for name in (SDK_STREAMING_LOGGER, *WEBSOCKETS_LOGGERS):
        assert logging.getLogger(name).level == logging.CRITICAL
        assert not logging.getLogger(name).isEnabledFor(logging.ERROR)


@pytest.mark.usefixtures("reset_levels")
def test_silence_streaming_logging_suppresses_the_sdk_client_logger():
    # The SDK logs "Connection failed: …" at ERROR through a *child* of the silenced
    # logger; prove a real record is dropped, not just that a level attribute moved.
    spy = _Spy()
    root = logging.getLogger()
    root.addHandler(spy)
    try:
        logging.getLogger(_SDK_CLIENT_LOGGER).error("Connection failed: handshake rejected")
        assert len(spy.records) == 1  # the spy sees the record before silencing
        silence_streaming_logging()
        logging.getLogger(_SDK_CLIENT_LOGGER).error("Connection failed: handshake rejected")
        assert len(spy.records) == 1  # …and nothing new after
    finally:
        root.removeHandler(spy)


@pytest.mark.usefixtures("reset_levels")
def test_silence_streaming_logging_stands_down_in_verbose_mode(monkeypatch):
    # Library logs are the requested output under -v/-vv; nothing gets clamped.
    monkeypatch.setattr(debuglog, "_verbosity", 1)
    silence_streaming_logging()
    for name in (SDK_STREAMING_LOGGER, *WEBSOCKETS_LOGGERS):
        assert logging.getLogger(name).level == logging.NOTSET


def test_sdk_streaming_logger_is_the_assemblyai_parent():
    assert SDK_STREAMING_LOGGER == "assemblyai.streaming"


def test_handshake_suggestion_names_whoami_env_and_host():
    text = handshake_suggestion("streaming.assemblyai.com")
    assert "assembly whoami" in text
    assert "--sandbox" in text
    assert "network/proxy access to streaming.assemblyai.com" in text


def test_handshake_suggestion_falls_back_when_host_is_empty():
    # e.g. the TTS host is empty outside the sandbox; never render "access to ."
    assert "network/proxy access to the streaming endpoint." in handshake_suggestion("")


class _SdkHandshake(Exception):
    """Mimics the assemblyai SDK's StreamingError: the HTTP status on ``.code``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"WebSocket handshake rejected (HTTP {status})")
        self.code = status


class _WsHandshake(Exception):
    """Mimics websockets' InvalidStatus: the status on ``.response.status_code``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"server rejected WebSocket connection: HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


def test_handshake_401_is_not_authenticated_with_suggestion():
    err = handshake_error(_SdkHandshake(401), "Streaming error", host="h.example")
    assert isinstance(err, NotAuthenticated)
    assert err.exit_code == 4
    assert err.rejected_key is True  # auto-login must not retry an env-provided key
    assert err.message == "Streaming error: WebSocket handshake rejected (HTTP 401)"
    assert err.suggestion == handshake_suggestion("h.example")


def test_handshake_403_is_api_error_with_suggestion():
    # 403 also covers WAF/region/plan blocks, so it stays exit 1 — but suggests.
    err = handshake_error(_WsHandshake(403), "Could not connect", host="h.example")
    assert isinstance(err, APIError)
    assert err.exit_code == 1
    assert err.message == "Could not connect: server rejected WebSocket connection: HTTP 403"
    assert err.suggestion == handshake_suggestion("h.example")


def test_handshake_status_read_from_websockets_response_shape():
    err = handshake_error(_WsHandshake(401), "Could not connect", host="h.example")
    assert isinstance(err, NotAuthenticated)


def test_non_handshake_errors_return_none():
    assert handshake_error(RuntimeError("socket dropped"), "Streaming error", host="h") is None
    # A WebSocket close code (e.g. 1008 policy violation) is not a handshake status.
    closed = types.SimpleNamespace(code=1008)
    assert handshake_error(closed, "Streaming error", host="h") is None
    # Other HTTP statuses (e.g. a 500 on the upgrade) are not auth-shaped.
    assert handshake_error(_WsHandshake(500), "Streaming error", host="h") is None


def _header_capture() -> tuple[Callable[..., str], dict[str, str]]:
    """A connect double plus the dict its Authorization header lands in."""
    captured: dict[str, str] = {}

    def _connect(url: str, *, additional_headers: dict[str, str], **kwargs: object) -> str:
        captured.update(additional_headers)
        return "ws"

    return _connect, captured


def test_open_authorized_ws_defaults_to_bearer_token():
    # The Voice Agent endpoint expects a Bearer token, so that's the default when
    # `bearer` is omitted entirely (not just when passed True).
    connect, captured = _header_capture()
    open_authorized_ws(connect, "secret", "wss://h/ws", message="m", host="h")
    assert captured["Authorization"] == "Bearer secret"


def test_open_authorized_ws_sends_raw_key_when_bearer_false():
    # Streaming endpoints (STT, TTS) authenticate with the raw API key — no "Bearer "
    # prefix. bearer=False must send the key verbatim, not a Bearer token.
    connect, captured = _header_capture()
    open_authorized_ws(connect, "secret", "wss://h/ws", message="m", host="h", bearer=False)
    assert captured["Authorization"] == "secret"
    assert "Bearer" not in captured["Authorization"]
