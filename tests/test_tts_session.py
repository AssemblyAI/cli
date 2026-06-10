from __future__ import annotations

import base64
import json

import pytest

from aai_cli import environments
from aai_cli.errors import APIError, NotAuthenticated
from aai_cli.tts import session


def _use_env(name: str) -> None:
    environments.set_active(environments.get(name))


def test_is_available_true_in_sandbox():
    _use_env("sandbox000")
    assert session.is_available() is True


def test_is_available_false_in_production():
    _use_env("production")
    assert session.is_available() is False


def test_ws_url_includes_set_params_only():
    _use_env("sandbox000")
    cfg = session.SpeakConfig(text="hi", voice="jane", language="English")
    url = session.ws_url(cfg.query_params())
    assert url.startswith("wss://streaming-tts.sandbox000.assemblyai-labs.com/v1/ws/?")
    assert "voice=jane" in url
    assert "language=English" in url
    assert "sample_rate" not in url


def test_ws_url_no_params_has_no_query_string():
    _use_env("sandbox000")
    url = session.ws_url(session.SpeakConfig(text="hi").query_params())
    assert url == "wss://streaming-tts.sandbox000.assemblyai-labs.com/v1/ws/"


def test_query_params_serializes_sample_rate_as_string():
    cfg = session.SpeakConfig(text="hi", sample_rate=16000)
    assert cfg.query_params() == {"sample_rate": "16000"}


class FakeWS:
    """A minimal stand-in for a websockets sync connection."""

    def __init__(self, incoming: list[str]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict] = []
        self.closed = False

    def recv(self) -> str:
        return self._incoming.pop(0)

    def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def close(self) -> None:
        self.closed = True


def _audio_frame(pcm: bytes, *, sample_rate: int = 24000, final: bool) -> str:
    return json.dumps(
        {
            "type": "Audio",
            "audio": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": sample_rate,
            "encoding": "pcm_s16le",
            "is_final_for_flush": final,
        }
    )


def _connect_returning(ws: FakeWS, captured: dict):
    def _connect(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    return _connect


def test_synthesize_drives_the_full_protocol():
    captured: dict = {}
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s1", "expires_at": 1}),
            _audio_frame(b"\x01\x02\x03\x04", final=False),
            _audio_frame(b"\x05\x06", final=True),
        ]
    )
    cfg = session.SpeakConfig(text="hello", voice="jane")
    result = session.synthesize("k", cfg, connect=_connect_returning(ws, captured))

    # Sends Generate(text), then Flush, then Terminate — in that order.
    assert [m["type"] for m in ws.sent] == ["Generate", "Flush", "Terminate"]
    assert ws.sent[0]["text"] == "hello"
    # Accumulates decoded PCM across chunks and stops on is_final_for_flush.
    assert result.pcm == b"\x01\x02\x03\x04\x05\x06"
    assert result.sample_rate == 24000
    # 6 bytes / 2 bytes-per-sample / 24000 Hz.
    assert result.audio_duration_seconds == pytest.approx(6 / 2 / 24000)
    # Auth header carries the key as a Bearer token.
    assert captured["kwargs"]["additional_headers"]["Authorization"] == "Bearer k"
    assert ws.closed is True


def test_synthesize_raises_on_missing_begin():
    ws = FakeWS([json.dumps({"type": "Audio"})])
    with pytest.raises(APIError):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_maps_error_frame_to_api_error():
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            json.dumps({"type": "Error", "error_code": 500, "error": "boom"}),
        ]
    )
    with pytest.raises(APIError, match="boom"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_invokes_on_warning_then_continues():
    seen: list[str] = []
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            json.dumps({"type": "Warning", "warning_code": 1, "warning": "slow"}),
            _audio_frame(b"\x01\x02", final=True),
        ]
    )
    result = session.synthesize(
        "k",
        session.SpeakConfig(text="hi"),
        connect=lambda *a, **k: ws,
        on_warning=seen.append,
    )
    assert seen == ["slow"]
    assert result.pcm == b"\x01\x02"


def test_synthesize_ignores_warning_without_callback():
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            json.dumps({"type": "Warning", "warning_code": 1, "warning": "slow"}),
            _audio_frame(b"\x01\x02", final=True),
        ]
    )
    # No on_warning: the warning is silently skipped, not an error.
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02"


def test_synthesize_maps_rejected_key_on_connect_to_not_authenticated():
    class Rejected(Exception):
        pass

    def _connect(*_a, **_k):
        raise Rejected("Unauthorized")

    with pytest.raises(NotAuthenticated):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=_connect)


def test_synthesize_maps_forbidden_connect_to_api_error():
    class Resp:
        status_code = 403

    class Forbidden(Exception):
        response = Resp()

    def _connect(*_a, **_k):
        raise Forbidden("Unauthorized")  # 403 -> NOT a rejected key

    with pytest.raises(APIError):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=_connect)
