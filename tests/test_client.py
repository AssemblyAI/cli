import types as _types
from unittest.mock import MagicMock, patch

import assemblyai as aai
import pytest

from assemblyai_cli import client
from assemblyai_cli.errors import APIError


def _stream_params(sample_rate: int = 16000):
    from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

    return StreamingParameters(
        sample_rate=sample_rate,
        format_turns=True,
        speech_model=SpeechModel.universal_streaming_multilingual,
    )


def test_validate_key_true_on_success():
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.return_value = MagicMock()
        assert client.validate_key("sk_good") is True


def test_validate_key_false_on_auth_error():
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError(
            "Authentication error, API token missing/invalid"
        )
        assert client.validate_key("sk_bad") is False


def test_validate_key_raises_on_other_sdk_error():
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError("server exploded")
        with pytest.raises(APIError):
            client.validate_key("sk")


def test_validate_key_raises_on_network_error():
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.side_effect = ConnectionError("boom")
        with pytest.raises(APIError):
            client.validate_key("sk")


def test_list_transcripts_returns_dict_rows():
    item = MagicMock()
    item.model_dump.return_value = {"id": "t1", "status": "completed", "created": "2026-01-01"}
    resp = MagicMock()
    resp.transcripts = [item]
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.return_value = resp
        rows = client.list_transcripts("sk", limit=5)
    assert rows == [{"id": "t1", "status": "completed", "created": "2026-01-01"}]
    item.model_dump.assert_called_once_with(mode="json")


def test_list_transcripts_supports_pydantic_v1_items():
    # assemblyai's transcription models are pydantic v1: no model_dump, but .json().
    import types

    item = types.SimpleNamespace(json=lambda: '{"id": "t2", "status": "queued"}')
    resp = MagicMock()
    resp.transcripts = [item]
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.return_value = resp
        rows = client.list_transcripts("sk", limit=5)
    assert rows == [{"id": "t2", "status": "queued"}]


def test_list_transcripts_auth_error_becomes_apierror():
    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError("nope")
        with pytest.raises(APIError):
            client.list_transcripts("sk")


def test_list_transcripts_rejected_key_becomes_not_authenticated():
    from assemblyai_cli.errors import NotAuthenticated

    with patch.object(client.aai, "Transcriber") as T:
        T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError(
            "Authentication error, API token missing/invalid"
        )
        with pytest.raises(NotAuthenticated):
            client.list_transcripts("sk_bad")


def test_resolve_audio_source_sample_explicit_and_missing():
    from assemblyai_cli.errors import UsageError

    assert client.resolve_audio_source(None, sample=True) == client.SAMPLE_AUDIO_URL
    assert client.resolve_audio_source("clip.mp3", sample=False) == "clip.mp3"
    with pytest.raises(UsageError):
        client.resolve_audio_source(None, sample=False)


def test_transcribe_blocks_and_returns_transcript():
    fake_transcript = MagicMock()
    fake_transcript.status = client.aai.TranscriptStatus.completed
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.return_value = fake_transcript

    cfg = aai.TranscriptionConfig(speaker_labels=True)
    with patch.object(client.aai, "Transcriber", return_value=fake_transcriber):
        result = client.transcribe("sk", "audio.mp3", config=cfg)

    fake_transcriber.transcribe.assert_called_once_with("audio.mp3", config=cfg)
    assert result is fake_transcript


def test_transcribe_raises_on_error_status():
    fake_transcript = MagicMock()
    fake_transcript.status = client.aai.TranscriptStatus.error
    fake_transcript.error = "decode failed"
    fake_transcript.id = "t_err"
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.return_value = fake_transcript

    with patch.object(client.aai, "Transcriber", return_value=fake_transcriber):
        with pytest.raises(APIError) as exc:
            client.transcribe("sk", "audio.mp3", config=aai.TranscriptionConfig())
    assert exc.value.transcript_id == "t_err"


def test_select_transcript_field_srt_uses_sdk():
    t = MagicMock()
    t.export_subtitles_srt.return_value = "1\n00:00:00,000 --> 00:00:02,000\nhello world\n"
    assert client.select_transcript_field(t, "srt") == (
        "1\n00:00:00,000 --> 00:00:02,000\nhello world\n"
    )
    t.export_subtitles_srt.assert_called_once_with()


def test_select_transcript_field_srt_network_error_becomes_apierror():
    t = MagicMock()
    t.export_subtitles_srt.side_effect = RuntimeError("connection reset")
    with pytest.raises(APIError):
        client.select_transcript_field(t, "srt")


def test_select_transcript_field_srt_auth_error_becomes_not_authenticated():
    from assemblyai_cli.errors import NotAuthenticated

    t = MagicMock()
    t.export_subtitles_srt.side_effect = RuntimeError("HTTP 401 Unauthorized")
    with pytest.raises(NotAuthenticated):
        client.select_transcript_field(t, "srt")


def test_get_transcript_calls_sdk():
    fake = MagicMock()
    with patch.object(client.aai.Transcript, "get_by_id", return_value=fake) as g:
        result = client.get_transcript("sk", "t_123")
    g.assert_called_once_with("t_123")
    assert result is fake


def test_get_transcript_generic_error_becomes_apierror():
    with patch.object(client.aai.Transcript, "get_by_id", side_effect=RuntimeError("boom")):
        with pytest.raises(APIError):
            client.get_transcript("sk", "t_x")


def test_get_transcript_auth_error_becomes_not_authenticated():
    from assemblyai_cli.errors import NotAuthenticated

    with patch.object(
        client.aai.Transcript, "get_by_id", side_effect=RuntimeError("HTTP 401 Unauthorized")
    ):
        with pytest.raises(NotAuthenticated):
            client.get_transcript("sk_bad", "t_x")


def test_transcribe_network_error_becomes_apierror():
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.side_effect = RuntimeError("connection reset")
    with patch.object(client.aai, "Transcriber", return_value=fake_transcriber):
        with pytest.raises(APIError):
            client.transcribe("sk", "audio.mp3", config=aai.TranscriptionConfig())


def test_transcribe_auth_error_becomes_not_authenticated():
    from assemblyai_cli.errors import NotAuthenticated

    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.side_effect = RuntimeError("Invalid API key")
    with patch.object(client.aai, "Transcriber", return_value=fake_transcriber):
        with pytest.raises(NotAuthenticated):
            client.transcribe("sk_bad", "audio.mp3", config=aai.TranscriptionConfig())


class _FakeStreamingClient:
    last = None

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
    client.stream_audio(
        "sk", [b"\x00"], params=_stream_params(), on_turn=lambda e: turns.append(e.transcript)
    )
    assert turns == ["hi"]
    assert _FakeStreamingClient.last.connected
    assert _FakeStreamingClient.last.disconnected  # disconnected in finally
    assert _FakeStreamingClient.last.params.sample_rate == 16000
    assert _FakeStreamingClient.last.params.format_turns is True
    assert _FakeStreamingClient.last.terminate is True  # graceful flush requested


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
    from assemblyai_cli.errors import NotAuthenticated

    class ConnectUnauthorized(_FakeStreamingClient):
        def connect(self, params):
            raise RuntimeError("401 Unauthorized: bad token")

    monkeypatch.setattr(client, "StreamingClient", ConnectUnauthorized)
    with pytest.raises(NotAuthenticated):
        client.stream_audio("sk_bad", [b"\x00"], params=_stream_params())


def test_stream_audio_auth_error_event_becomes_not_authenticated(monkeypatch):
    from assemblyai_cli.errors import NotAuthenticated

    class AuthErrClient(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Error](self, "Unauthorized: invalid api key")

    monkeypatch.setattr(client, "StreamingClient", AuthErrClient)
    with pytest.raises(NotAuthenticated):
        client.stream_audio("sk_bad", [b"\x00"], params=_stream_params())


def test_stream_audio_mid_stream_error_becomes_apierror(monkeypatch):
    class StreamFails(_FakeStreamingClient):
        def stream(self, source):
            raise RuntimeError("socket dropped")

    monkeypatch.setattr(client, "StreamingClient", StreamFails)
    with pytest.raises(APIError):
        client.stream_audio("sk", [b"\x00"], params=_stream_params())
    assert StreamFails.last.disconnected  # still disconnected in finally


def test_stream_audio_swallows_broken_pipe_in_callback(monkeypatch):
    # A closed downstream pipe makes a turn write raise BrokenPipeError on the SDK's
    # reader thread; the guard must swallow it instead of dumping a thread traceback.
    monkeypatch.setattr(client, "StreamingClient", _FakeStreamingClient)
    # never touch the real stdout fd during the test
    monkeypatch.setattr("assemblyai_cli.stdio.silence_stdout", lambda: None)

    def on_turn(_event):
        raise BrokenPipeError

    client.stream_audio("sk", [b"\x00"], params=_stream_params(), on_turn=on_turn)  # no raise


def test_stream_audio_passes_through_clierror(monkeypatch):
    from assemblyai_cli.errors import CLIError

    class StreamRaisesCLIError(_FakeStreamingClient):
        def stream(self, source):
            raise CLIError("boom", error_type="x", exit_code=2)

    monkeypatch.setattr(client, "StreamingClient", StreamRaisesCLIError)
    with pytest.raises(CLIError) as exc:
        client.stream_audio("sk", [b"\x00"], params=_stream_params())
    assert exc.value.exit_code == 2  # not rewrapped into APIError


def test_transcribe_passes_prebuilt_config(monkeypatch):
    import assemblyai as aai

    from assemblyai_cli import client

    captured = {}

    class FakeTranscriber:
        def transcribe(self, audio, config=None):
            captured["audio"] = audio
            captured["config"] = config
            t = MagicMock()
            t.status = aai.TranscriptStatus.completed
            return t

    monkeypatch.setattr(aai, "Transcriber", lambda: FakeTranscriber())
    cfg = aai.TranscriptionConfig(speaker_labels=True)
    client.transcribe("sk", "audio.mp3", config=cfg)
    assert captured["audio"] == "audio.mp3"
    assert captured["config"] is cfg


def test_stream_audio_accepts_params(monkeypatch):
    from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

    from assemblyai_cli import client

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

    monkeypatch.setattr("assemblyai_cli.client.StreamingClient", FakeSC)
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
