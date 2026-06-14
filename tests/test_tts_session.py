from __future__ import annotations

import base64
import json

import pytest

from aai_cli.core import environments
from aai_cli.core.errors import APIError, CLIError, NotAuthenticated
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


def _set_attr(obj: object, name: str, value: object) -> None:
    # Indirect setattr: the attribute name is opaque to the type checker (so no
    # read-only error) and non-constant (so ruff's B010 leaves it as setattr),
    # while still hitting a frozen dataclass's blocking __setattr__ at runtime.
    setattr(obj, name, value)


def test_speak_config_is_immutable():
    cfg = session.SpeakConfig(text="hi")
    with pytest.raises(AttributeError):  # frozen=True -> assignment is blocked
        _set_attr(cfg, "text", "changed")


def test_speak_result_is_immutable():
    result = session.SpeakResult(pcm=b"\x01", sample_rate=24000, audio_duration_seconds=0.0)
    with pytest.raises(AttributeError):
        _set_attr(result, "sample_rate", 1)


class FakeWS:
    """A minimal stand-in for a websockets sync connection."""

    def __init__(self, incoming: list[str]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict[str, object]] = []
        self.closed = False
        self.recv_timeouts: list[float | None] = []

    def recv(self, timeout: float | None = None) -> str:
        self.recv_timeouts.append(timeout)
        return self._incoming.pop(0)

    def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def close(self) -> None:
        self.closed = True


def _audio_frame(pcm: bytes, *, final: bool) -> str:
    # The real server's Audio frames carry only the PCM payload and the final flag;
    # the sample rate is reported once, up front, in the Begin frame's configuration.
    return json.dumps(
        {
            "type": "Audio",
            "audio": base64.b64encode(pcm).decode("ascii"),
            "is_final": final,
        }
    )


def _begin_frame(*, sample_rate: int = 24000) -> str:
    return json.dumps(
        {
            "type": "Begin",
            "id": "s1",
            "expires_at": 1,
            "configuration": {"voice": "jane", "language": "english", "sample_rate": sample_rate},
        }
    )


def _connect_returning(ws: FakeWS, captured: dict[str, object]):
    def _connect(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    return _connect


def test_synthesize_drives_the_full_protocol():
    captured: dict = {}
    ws = FakeWS(
        [
            _begin_frame(sample_rate=24000),
            _audio_frame(b"\x01\x02\x03\x04", final=False),
            _audio_frame(b"\x05\x06", final=True),
        ]
    )
    cfg = session.SpeakConfig(text="hello", voice="jane")
    result = session.synthesize("k", cfg, connect=_connect_returning(ws, captured))

    # Sends Generate(text), then ForceFlushTextBuffer, then Terminate — in that order.
    assert [m["type"] for m in ws.sent] == ["Generate", "ForceFlushTextBuffer", "Terminate"]
    assert ws.sent[0]["text"] == "hello"
    # Accumulates decoded PCM across chunks and stops on the is_final frame.
    assert result.pcm == b"\x01\x02\x03\x04\x05\x06"
    # The sample rate comes from the Begin frame's configuration, not the Audio frames.
    assert result.sample_rate == 24000
    # 6 bytes / 2 bytes-per-sample / 24000 Hz.
    assert result.audio_duration_seconds == pytest.approx(6 / 2 / 24000)
    # Auth header carries the key as a Bearer token.
    assert captured["kwargs"]["additional_headers"]["Authorization"] == "Bearer k"
    # The frame-size cap is lifted: Audio frames can exceed websockets' 1 MiB default.
    assert captured["kwargs"]["max_size"] is None
    assert ws.closed is True


def test_synthesize_reads_sample_rate_from_begin_configuration():
    # A non-default rate in the Begin frame flows into the result and its duration,
    # proving the rate is read from Begin.configuration rather than hardcoded.
    ws = FakeWS([_begin_frame(sample_rate=16000), _audio_frame(b"\x01\x02\x03\x04", final=True)])
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.sample_rate == 16000
    assert result.audio_duration_seconds == pytest.approx(4 / 2 / 16000)


def test_synthesize_falls_back_to_default_rate_when_begin_omits_configuration():
    # Begin without a configuration block -> the 24 kHz fallback, not a KeyError.
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            _audio_frame(b"\x01\x02", final=True),
        ]
    )
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.sample_rate == 24000


def test_synthesize_raises_on_missing_begin():
    ws = FakeWS([json.dumps({"type": "Audio"})])
    with pytest.raises(APIError):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_maps_audio_frame_without_payload_to_api_error():
    # A malformed Audio frame (no "audio" key) must surface as a clean APIError
    # naming the defect, not a bare KeyError read as "TTS session failed: 'audio'".
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            json.dumps({"type": "Audio", "is_final": True}),
        ]
    )
    with pytest.raises(APIError, match="without an audio payload"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_maps_undecodable_audio_frame_to_api_error():
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            json.dumps({"type": "Audio", "audio": "!!!", "is_final": True}),
        ]
    )
    with pytest.raises(APIError, match="not valid base64"):
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


def test_synthesize_bounds_every_recv_with_the_frame_timeout():
    # Each frame wait is bounded (60s): an unbounded recv() would hang the command
    # forever if the server went silent mid-session.
    ws = FakeWS([_begin_frame(), _audio_frame(b"\x01\x02", final=True)])
    session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert ws.recv_timeouts == [60.0, 60.0]


def test_synthesize_maps_silent_server_to_clean_api_error():
    # websockets' sync recv raises TimeoutError when the bound expires; that must
    # surface as a clean APIError naming the stall, not a hang or a raw traceback.
    class SilentWS(FakeWS):
        def recv(self, timeout: float | None = None) -> str:
            raise TimeoutError

    ws = SilentWS([])
    with pytest.raises(APIError, match="stopped responding"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert ws.closed is True


def test_synthesize_silent_server_error_names_the_timeout_window():
    class SilentWS(FakeWS):
        def recv(self, timeout: float | None = None) -> str:
            raise TimeoutError

    with pytest.raises(APIError, match="no frame for 60s"):
        session.synthesize(
            "k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: SilentWS([])
        )


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
    _use_env("sandbox000")

    class Resp:
        status_code = 403

    class Forbidden(Exception):
        response = Resp()

    def _connect(*_a, **_k):
        raise Forbidden("Unauthorized")  # 403 -> NOT a rejected key

    with pytest.raises(APIError) as exc:
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=_connect)
    assert "Could not connect to the TTS service" in exc.value.message
    # The rejected handshake carries the actionable next steps, env host included.
    assert exc.value.suggestion is not None
    assert "assembly whoami" in exc.value.suggestion
    assert "--sandbox" in exc.value.suggestion
    assert "streaming-tts.sandbox000" in exc.value.suggestion


def test_synthesize_handshake_401_is_not_authenticated_with_suggestion():
    class Resp:
        status_code = 401

    class Rejected(Exception):
        response = Resp()

    def _connect(*_a, **_k):
        raise Rejected("server rejected WebSocket connection: HTTP 401")

    with pytest.raises(NotAuthenticated) as exc:
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=_connect)
    assert exc.value.exit_code == 4
    assert exc.value.rejected_key is True
    assert exc.value.suggestion is not None
    assert "assembly whoami" in exc.value.suggestion


def test_synthesize_error_frame_without_details_says_unknown():
    # An Error frame with no error_code / error message still produces a clean,
    # non-empty message (the "unknown" fallback), not a KeyError.
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            json.dumps({"type": "Error"}),
        ]
    )
    with pytest.raises(APIError, match="unknown"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_maps_unexpected_protocol_error_to_api_error():
    # recv() on an exhausted FakeWS raises IndexError inside _run_protocol; the
    # generic-exception arm maps it to a clean APIError rather than leaking.
    ws = FakeWS([json.dumps({"type": "Begin", "id": "s", "expires_at": 1})])
    with pytest.raises(APIError, match="TTS session failed"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


@pytest.mark.disable_socket
def test_synthesize_without_connect_uses_real_client_and_fails_cleanly():
    # No `connect` provided: synthesize imports websockets' real sync client and
    # attempts a connection. pytest-socket blocks socket creation, so this must
    # surface as a clean CLIError (mapped in diagnostics.open_authorized_ws),
    # never a raw socket error. disable_socket pins that blocked-at-creation behavior
    # on Windows too (the suite-wide conftest otherwise allows loopback there, which
    # would let the socket be created and then leak when the real connect is blocked).
    _use_env("sandbox000")
    with pytest.raises(CLIError):
        session.synthesize("k", session.SpeakConfig(text="hi"))


def test_synthesize_dialogue_concatenates_segments_with_silence():
    # One fresh fake socket per segment; record the voice each connection requested.
    sockets = [
        FakeWS([_begin_frame(sample_rate=24000), _audio_frame(b"\xaa\xbb", final=True)]),
        FakeWS([_begin_frame(sample_rate=24000), _audio_frame(b"\xcc\xdd", final=True)]),
    ]
    urls: list[str] = []

    def _connect(url: str, **_kwargs):
        urls.append(url)
        return sockets.pop(0)

    result = session.synthesize_dialogue(
        "k",
        [("jane", "Hello."), ("michael", "Hi.")],
        language="English",
        connect=_connect,
    )
    # Each segment connected with its own voice.
    assert "voice=jane" in urls[0]
    assert "voice=michael" in urls[1]
    # 0.25 s of silence (24000 * 0.25 * 2 = 12000 zero bytes) sits BETWEEN the two
    # segments' PCM, with none at the ends.
    gap = b"\x00" * 12000
    assert result.pcm == b"\xaa\xbb" + gap + b"\xcc\xdd"
    assert result.sample_rate == 24000
    # Pin the duration formula (len/2/rate) so its operators survive the mutation gate.
    assert result.audio_duration_seconds == pytest.approx(len(result.pcm) / 2 / 24000)


def test_synthesize_dialogue_single_segment_has_no_silence():
    ws = FakeWS([_begin_frame(sample_rate=24000), _audio_frame(b"\x01\x02", final=True)])
    result = session.synthesize_dialogue("k", [("jane", "Hi.")], connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02"  # no leading/trailing pad


def test_synthesize_dialogue_uses_server_sample_rate():
    # A non-default server rate must flow into the result (proving the per-segment
    # rate is read, not left at the default) and into the duration denominator.
    ws = FakeWS([_begin_frame(sample_rate=16000), _audio_frame(b"\x01\x02", final=True)])
    result = session.synthesize_dialogue("k", [("jane", "Hi.")], connect=lambda *a, **k: ws)
    assert result.sample_rate == 16000
    assert result.audio_duration_seconds == pytest.approx(2 / 2 / 16000)


def test_synthesize_dialogue_empty_segments_returns_silent_default():
    # No segments -> no audio at the default rate, and no crash. connect is omitted
    # entirely: the loop body never runs, so no connection is ever attempted.
    result = session.synthesize_dialogue("k", [])
    assert result.pcm == b""
    assert result.sample_rate == 24000
    assert result.audio_duration_seconds == 0.0
