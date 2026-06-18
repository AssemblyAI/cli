from __future__ import annotations

import json

import pytest

from aai_cli.core.errors import APIError, CLIError, NotAuthenticated
from aai_cli.tts import session
from tests._tts_session_helpers import (
    FakeWS,
    audio_chunk,
    audio_frame,
    begin_frame,
    connect_returning,
    flush_done_frame,
    use_env,
)


def test_is_available_true_in_sandbox():
    use_env("sandbox000")
    assert session.is_available() is True


def test_is_available_false_in_production():
    use_env("production")
    assert session.is_available() is False


def test_ws_url_includes_set_params_only():
    use_env("sandbox000")
    cfg = session.SpeakConfig(text="hi", voice="jane", language="English")
    url = session.ws_url(cfg.query_params())
    assert url.startswith("wss://streaming-tts.sandbox000.assemblyai-labs.com/v1/ws/?")
    assert "voice=jane" in url
    assert "language=English" in url
    assert "sample_rate" not in url


def test_ws_url_no_params_has_no_query_string():
    use_env("sandbox000")
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


def test_synthesize_drives_the_full_protocol():
    captured: dict = {}
    ws = FakeWS(
        [
            begin_frame(sample_rate=24000),
            audio_frame(b"\x01\x02\x03\x04", final=False),
            audio_frame(b"\x05\x06", final=True),
        ]
    )
    cfg = session.SpeakConfig(text="hello", voice="jane")
    result = session.synthesize("k", cfg, connect=connect_returning(ws, captured))

    # Sends Generate(text), then Flush, then Terminate — in that order. The flush tag
    # is "Flush" (the server's accepted tag, as the agent-cascade template sends),
    # not "ForceFlushTextBuffer", which the server rejects with a validation error.
    assert [m["type"] for m in ws.sent] == ["Generate", "Flush", "Terminate"]
    assert ws.sent[0]["text"] == "hello"
    # Accumulates decoded PCM across chunks and stops on the is_final frame.
    assert result.pcm == b"\x01\x02\x03\x04\x05\x06"
    # The sample rate comes from the Begin frame's configuration, not the Audio frames.
    assert result.sample_rate == 24000
    # 6 bytes / 2 bytes-per-sample / 24000 Hz.
    assert result.audio_duration_seconds == pytest.approx(6 / 2 / 24000)
    # Streaming TTS authenticates with the raw API key — no "Bearer " prefix (the
    # AssemblyAI streaming convention; the working agent-cascade template does the
    # same). A Bearer token makes the server reject the session with an Error frame.
    assert captured["kwargs"]["additional_headers"]["Authorization"] == "k"
    assert "Bearer" not in captured["kwargs"]["additional_headers"]["Authorization"]
    # The frame-size cap is lifted: Audio frames can exceed websockets' 1 MiB default.
    assert captured["kwargs"]["max_size"] is None
    # The keepalive pong deadline is disabled: a long input (e.g. a --url PDF) can leave
    # the server silent for >20s before the first Audio frame, and websockets' default
    # ping_timeout would close the still-alive connection with code 1011. _RECV_TIMEOUT_SECONDS
    # is the single liveness authority instead, so ping_timeout must be None.
    assert captured["kwargs"]["ping_timeout"] is None


def test_synthesize_stops_on_flush_done_when_audio_omits_is_final():
    # The live server ends a synthesis with a FlushDone frame and never sets is_final
    # on its Audio frames. Without handling FlushDone the loop blocks until the recv
    # timeout (the audio is silently lost), so FlushDone must end collection and return
    # every PCM byte gathered so far — then Terminate the session as usual.
    ws = FakeWS(
        [
            begin_frame(sample_rate=24000),
            audio_chunk(b"\x01\x02\x03\x04"),
            audio_chunk(b"\x05\x06"),
            flush_done_frame(),
        ]
    )
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02\x03\x04\x05\x06"
    assert [m["type"] for m in ws.sent] == ["Generate", "Flush", "Terminate"]
    assert ws.closed is True


def test_synthesize_streams_each_chunk_to_on_audio_as_it_arrives():
    # The whole point of streaming playback: every decoded Audio chunk is handed to
    # on_audio(chunk, sample_rate) the moment it arrives — one call per frame, in
    # order, with the server's reported rate — while the full PCM is still returned.
    ws = FakeWS(
        [
            begin_frame(sample_rate=16000),
            audio_chunk(b"\x01\x02"),
            audio_chunk(b"\x03\x04"),
            flush_done_frame(),
        ]
    )
    streamed: list[tuple[bytes, int]] = []
    result = session.synthesize(
        "k",
        session.SpeakConfig(text="hi"),
        connect=lambda *a, **k: ws,
        on_audio=lambda chunk, rate: streamed.append((chunk, rate)),
    )
    # One call per Audio frame, in arrival order, each carrying the Begin sample rate.
    assert streamed == [(b"\x01\x02", 16000), (b"\x03\x04", 16000)]
    # The buffered result is unchanged — streaming is additive, not a replacement.
    assert result.pcm == b"\x01\x02\x03\x04"


def test_synthesize_without_on_audio_still_returns_full_pcm():
    # The callback is optional: omitting it must not change the buffered result.
    ws = FakeWS([begin_frame(), audio_frame(b"\x01\x02", final=True)])
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02"


def test_synthesize_reads_sample_rate_from_begin_configuration():
    # A non-default rate in the Begin frame flows into the result and its duration,
    # proving the rate is read from Begin.configuration rather than hardcoded.
    ws = FakeWS([begin_frame(sample_rate=16000), audio_frame(b"\x01\x02\x03\x04", final=True)])
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.sample_rate == 16000
    assert result.audio_duration_seconds == pytest.approx(4 / 2 / 16000)


def test_synthesize_falls_back_to_default_rate_when_begin_omits_configuration():
    # Begin without a configuration block -> the 24 kHz fallback, not a KeyError.
    ws = FakeWS(
        [
            json.dumps({"type": "Begin", "id": "s", "expires_at": 1}),
            audio_frame(b"\x01\x02", final=True),
        ]
    )
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.sample_rate == 24000


def test_synthesize_raises_on_missing_begin():
    ws = FakeWS([json.dumps({"type": "Audio"})])
    with pytest.raises(APIError, match="did not start the session"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_session_start_error_frame_surfaces_its_reason():
    # When the very first frame is an Error (not Begin), the server is explaining why
    # it refused the session — surface that reason and code, not a generic "got 'Error'".
    ws = FakeWS([json.dumps({"type": "Error", "error_code": 401, "error": "bad token"})])
    with pytest.raises(APIError, match="bad token") as excinfo:
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert "401" in str(excinfo.value)


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
    ws = FakeWS([begin_frame(), audio_frame(b"\x01\x02", final=True)])
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
            audio_frame(b"\x01\x02", final=True),
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
            audio_frame(b"\x01\x02", final=True),
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
    use_env("sandbox000")

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
    use_env("sandbox000")
    with pytest.raises(CLIError):
        session.synthesize("k", session.SpeakConfig(text="hi"))


def test_default_connect_forwards_keepalive_and_frame_settings(monkeypatch):
    # The real-client factory must forward the frame cap and the keepalive pong deadline
    # verbatim to websockets — synthesize() passes ping_timeout=None to disable the deadline
    # so a long --url synthesis can't die with a 1011 "keepalive ping timeout" the moment the
    # server pauses >20s before the first frame. Sentinel (non-None) values are forwarded so a
    # mutant that hardcodes either argument can't slip through.
    captured: dict = {}

    def _fake_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("websockets.sync.client.connect", _fake_connect)
    session._default_connect(
        "wss://example/ws",
        additional_headers={"Authorization": "k"},
        max_size=4096,
        ping_timeout=42.0,
    )
    assert captured["url"] == "wss://example/ws"
    assert captured["kwargs"]["additional_headers"] == {"Authorization": "k"}
    assert captured["kwargs"]["max_size"] == 4096
    assert captured["kwargs"]["ping_timeout"] == 42.0


def test_synthesize_chunked_synthesizes_each_chunk_on_its_own_connection(monkeypatch):
    # PocketTTS is fed one chunk per connection — never the whole document in one Generate.
    monkeypatch.setattr("aai_cli.tts.text.chunk_text", lambda _t: ["First chunk.", "Second chunk."])
    sockets = [
        FakeWS([begin_frame(), audio_frame(b"\x01\x02", final=True)]),
        FakeWS([begin_frame(), audio_frame(b"\x03\x04", final=True)]),
    ]
    pool = list(sockets)
    result = session.synthesize_chunked(
        "k", session.SpeakConfig(text="ignored"), connect=lambda *a, **k: pool.pop(0)
    )
    # One Generate per chunk, each on its own connection, in order.
    assert [ws.sent[0]["text"] for ws in sockets] == ["First chunk.", "Second chunk."]
    assert not pool  # both connections were opened
    # PCM is concatenated across the chunk connections, and the duration recomputed from it.
    assert result.pcm == b"\x01\x02\x03\x04"
    assert result.audio_duration_seconds == pytest.approx(4 / 2 / 24000)


def test_synthesize_chunked_streams_each_chunk_to_on_audio(monkeypatch):
    # on_audio fires per chunk as it arrives (incremental playback), carrying each
    # connection's reported sample rate, across all chunks.
    monkeypatch.setattr("aai_cli.tts.text.chunk_text", lambda _t: ["a.", "b."])
    pool = [
        FakeWS([begin_frame(sample_rate=16000), audio_frame(b"\x01\x02", final=True)]),
        FakeWS([begin_frame(sample_rate=16000), audio_frame(b"\x03\x04", final=True)]),
    ]
    streamed: list[tuple[bytes, int]] = []
    session.synthesize_chunked(
        "k",
        session.SpeakConfig(text="x"),
        connect=lambda *a, **k: pool.pop(0),
        on_audio=lambda chunk, rate: streamed.append((chunk, rate)),
    )
    assert streamed == [(b"\x01\x02", 16000), (b"\x03\x04", 16000)]


def test_synthesize_chunked_single_chunk_passes_text_through():
    # A short input is one chunk: the original text is synthesized unchanged.
    captured: dict = {}
    ws = FakeWS([begin_frame(), audio_frame(b"\x01\x02", final=True)])
    session.synthesize_chunked(
        "k", session.SpeakConfig(text="Just one sentence."), connect=connect_returning(ws, captured)
    )
    assert ws.sent[0]["text"] == "Just one sentence."


def test_synthesize_chunked_empty_text_returns_silent_default():
    # Whitespace-only text yields no chunks -> no audio at the default rate, and no crash.
    # connect is omitted entirely: the loop never runs, so no connection is attempted.
    result = session.synthesize_chunked("k", session.SpeakConfig(text="   "))
    assert result.pcm == b""
    assert result.sample_rate == 24000
    assert result.audio_duration_seconds == 0.0
