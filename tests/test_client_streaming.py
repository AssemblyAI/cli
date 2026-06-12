"""client.stream_audio tests: handler wiring, error classification, teardown.

Non-streaming client wrapper tests (transcribe, list_transcripts, validate_key,
select_transcript_field, get_transcript) live in test_client.py.
"""

import types as _types

import pytest

from aai_cli import client
from aai_cli.errors import APIError


def _stream_params(sample_rate: int = 16000):
    from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

    return StreamingParameters(
        sample_rate=sample_rate,
        format_turns=True,
        speech_model=SpeechModel.universal_streaming_multilingual,
    )


class _FakeStreamingClient:
    last: "_FakeStreamingClient | None" = None

    def __init__(self, options):
        self.handlers = {}
        self.connected = False
        self.disconnected = False
        _FakeStreamingClient.last = self

    def on(self, event, handler):
        self.handlers[event] = handler

    def connect(self, params):
        self.connected = True
        self.params = params

    def stream(self, source):
        from assemblyai.streaming.v3 import StreamingEvents

        self.handlers[StreamingEvents.Turn](
            self, _types.SimpleNamespace(transcript="hi", end_of_turn=True)
        )

    def disconnect(self, terminate=False):
        self.disconnected = True
        self.terminate = terminate


def test_stream_audio_wires_handlers_and_streams(monkeypatch):
    monkeypatch.setattr(client, "StreamingClient", _FakeStreamingClient)
    turns = []
    begins = []
    client.stream_audio(
        "sk",
        [b"\x00"],
        params=_stream_params(),
        on_begin=lambda e: begins.append(e),
        on_turn=lambda e: turns.append(e.transcript),
    )
    assert turns == ["hi"]
    assert begins == []
    last = _FakeStreamingClient.last
    assert last is not None
    assert last.connected
    assert last.disconnected  # disconnected in finally
    assert last.params.sample_rate == 16000
    assert last.params.format_turns is True
    assert last.terminate is True  # graceful flush requested


def test_stream_audio_registers_begin_handler_when_provided(monkeypatch):
    # A provided on_begin must actually be wired to the Begin event (pins
    # `if on_begin is not None`); inverting it would leave Begin unhandled.
    class BeginClient(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Begin](self, _types.SimpleNamespace(id="sess_1"))

    monkeypatch.setattr(client, "StreamingClient", BeginClient)
    begins = []
    client.stream_audio(
        "sk",
        [b"\x00"],
        params=_stream_params(),
        on_begin=lambda e: begins.append(e.id),
    )
    assert begins == ["sess_1"]


def test_stream_audio_raises_on_error_event(monkeypatch):
    class ErrClient(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Error](self, "boom")

    monkeypatch.setattr(client, "StreamingClient", ErrClient)
    with pytest.raises(APIError):
        client.stream_audio("sk", [b"\x00"], params=_stream_params())


def test_stream_audio_forwards_termination(monkeypatch):
    class TermClient(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Termination](
                self, _types.SimpleNamespace(audio_duration_seconds=3.0)
            )

    monkeypatch.setattr(client, "StreamingClient", TermClient)
    seen = []
    client.stream_audio(
        "sk",
        [b"\x00"],
        params=_stream_params(),
        on_termination=lambda e: seen.append(e.audio_duration_seconds),
    )
    assert seen == [3.0]


def test_stream_audio_connect_error_becomes_apierror(monkeypatch):
    class ConnectFails(_FakeStreamingClient):
        def connect(self, params):
            raise RuntimeError("handshake refused")

    monkeypatch.setattr(client, "StreamingClient", ConnectFails)
    with pytest.raises(APIError):
        client.stream_audio("sk", [b"\x00"], params=_stream_params())


def test_stream_audio_connect_auth_error_becomes_not_authenticated(monkeypatch):
    from aai_cli.errors import NotAuthenticated

    class ConnectUnauthorized(_FakeStreamingClient):
        def connect(self, params):
            raise RuntimeError("401 Unauthorized: bad token")

    monkeypatch.setattr(client, "StreamingClient", ConnectUnauthorized)
    with pytest.raises(NotAuthenticated):
        client.stream_audio("sk_bad", [b"\x00"], params=_stream_params())


def test_stream_audio_auth_error_event_becomes_not_authenticated(monkeypatch):
    from aai_cli.errors import NotAuthenticated

    class AuthErrClient(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Error](self, "Unauthorized: invalid api key")

    monkeypatch.setattr(client, "StreamingClient", AuthErrClient)
    with pytest.raises(NotAuthenticated):
        client.stream_audio("sk_bad", [b"\x00"], params=_stream_params())


def test_stream_audio_handshake_403_event_carries_suggestion(monkeypatch):
    # The SDK reports a rejected handshake as an Error event (StreamingError with the
    # HTTP status on .code); the CLI must add the actionable next steps, not stop at
    # "Streaming error: WebSocket handshake rejected (HTTP 403)".
    from assemblyai.streaming.v3 import StreamingError

    class Handshake403Client(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Error](
                self, StreamingError("WebSocket handshake rejected (HTTP 403)", code=403)
            )

    monkeypatch.setattr(client, "StreamingClient", Handshake403Client)
    with pytest.raises(APIError) as exc:
        client.stream_audio("sk", [b"\x00"], params=_stream_params())
    assert exc.value.message == "Streaming error: WebSocket handshake rejected (HTTP 403)"
    assert exc.value.suggestion is not None
    assert "assembly whoami" in exc.value.suggestion
    assert "--sandbox" in exc.value.suggestion
    # The suggestion names the active environment's streaming host (production here).
    assert "streaming.assemblyai.com" in exc.value.suggestion


def test_stream_audio_handshake_401_event_is_not_authenticated_with_suggestion(monkeypatch):
    from assemblyai.streaming.v3 import StreamingError

    from aai_cli.errors import NotAuthenticated

    class Handshake401Client(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Error](
                self, StreamingError("WebSocket handshake rejected (HTTP 401)", code=401)
            )

    monkeypatch.setattr(client, "StreamingClient", Handshake401Client)
    with pytest.raises(NotAuthenticated) as exc:
        client.stream_audio("sk_bad", [b"\x00"], params=_stream_params())
    assert exc.value.exit_code == 4
    assert exc.value.rejected_key is True
    assert exc.value.suggestion is not None
    assert "assembly whoami" in exc.value.suggestion


def test_stream_audio_silences_sdk_and_websockets_loggers(monkeypatch):
    # The streaming setup must raise the library loggers above ERROR so a reader-thread
    # failure can't dump a duplicate log line or raw traceback next to the CLIError.
    import logging

    names = ("assemblyai.streaming", "websockets", "websockets.client")
    previous = {name: logging.getLogger(name).level for name in names}
    for name in names:
        logging.getLogger(name).setLevel(logging.NOTSET)
    try:
        monkeypatch.setattr(client, "StreamingClient", _FakeStreamingClient)
        client.stream_audio("sk", [b"\x00"], params=_stream_params(), on_turn=lambda e: None)
        for name in names:
            assert logging.getLogger(name).level == logging.CRITICAL, name
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def test_stream_audio_mid_stream_error_becomes_apierror(monkeypatch):
    class StreamFails(_FakeStreamingClient):
        def stream(self, source):
            raise RuntimeError("socket dropped")

    monkeypatch.setattr(client, "StreamingClient", StreamFails)
    with pytest.raises(APIError):
        client.stream_audio("sk", [b"\x00"], params=_stream_params())
    last = StreamFails.last
    assert last is not None
    assert last.disconnected  # still disconnected in finally


def test_stream_audio_swallows_broken_pipe_in_callback(monkeypatch):
    # A closed downstream pipe makes a turn write raise BrokenPipeError on the SDK's
    # reader thread; the guard must swallow it instead of dumping a thread traceback.
    monkeypatch.setattr(client, "StreamingClient", _FakeStreamingClient)
    # never touch the real stdout fd during the test
    monkeypatch.setattr("aai_cli.stdio.silence_stdout", lambda: None)

    def on_turn(_event):
        raise BrokenPipeError

    client.stream_audio("sk", [b"\x00"], params=_stream_params(), on_turn=on_turn)  # no raise


def test_stream_audio_passes_through_clierror(monkeypatch):
    from aai_cli.errors import CLIError

    class StreamRaisesCLIError(_FakeStreamingClient):
        def stream(self, source):
            raise CLIError("boom", error_type="x", exit_code=2)

    monkeypatch.setattr(client, "StreamingClient", StreamRaisesCLIError)
    with pytest.raises(CLIError) as exc:
        client.stream_audio("sk", [b"\x00"], params=_stream_params())
    assert exc.value.exit_code == 2  # not rewrapped into APIError


def test_stream_audio_accepts_params(monkeypatch):
    from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

    from aai_cli import client

    captured = {}

    class FakeSC:
        def __init__(self, *a, **k):
            pass

        def on(self, *a, **k):
            pass

        def connect(self, params):
            captured["params"] = params

        def stream(self, source):
            pass

        def disconnect(self, terminate=True):
            pass

    monkeypatch.setattr("aai_cli.client.StreamingClient", FakeSC)
    params = StreamingParameters(
        sample_rate=16000, speech_model=SpeechModel.universal_streaming_multilingual
    )
    client.stream_audio("sk", iter([b""]), params=params)
    assert captured["params"] is params


def test_stream_audio_flushes_termination_on_disconnect(monkeypatch):
    class DeferredTermClient(_FakeStreamingClient):
        def stream(self, source):
            pass  # nothing dispatched during stream; the server flushes on terminate

        def disconnect(self, terminate=False):
            self.disconnected = True
            self.terminate = terminate
            if terminate:
                from assemblyai.streaming.v3 import StreamingEvents

                self.handlers[StreamingEvents.Termination](
                    self, _types.SimpleNamespace(audio_duration_seconds=5.0)
                )

    monkeypatch.setattr(client, "StreamingClient", DeferredTermClient)
    seen = []
    client.stream_audio(
        "sk",
        [b"\x00"],
        params=_stream_params(),
        on_termination=lambda e: seen.append(e.audio_duration_seconds),
    )
    assert seen == [5.0]
