import types as _types

import assemblyai as aai
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


def test_validate_key_true_on_success(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.return_value = mocker.MagicMock()
    assert client.validate_key("sk_good") is True
    # The probe asks for a single row — it only needs to confirm the key authenticates.
    params = T.return_value.list_transcripts.call_args.args[0]
    assert params.limit == 1


def test_validate_key_false_on_auth_error(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError(
        "Authentication error, API token missing/invalid"
    )
    assert client.validate_key("sk_bad") is False


def test_validate_key_raises_on_other_sdk_error(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError("server exploded")
    with pytest.raises(APIError):
        client.validate_key("sk")


def test_validate_key_raises_on_network_error(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = ConnectionError("boom")
    with pytest.raises(APIError):
        client.validate_key("sk")


def test_list_transcripts_returns_dict_rows(mocker):
    item = mocker.MagicMock()
    item.model_dump.return_value = {"id": "t1", "status": "completed", "created": "2026-01-01"}
    resp = mocker.MagicMock()
    resp.transcripts = [item]
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.return_value = resp
    rows = client.list_transcripts("sk", limit=5)
    assert rows == [{"id": "t1", "status": "completed", "created": "2026-01-01"}]
    item.model_dump.assert_called_once_with(mode="json")


def test_list_transcripts_supports_pydantic_v1_items(mocker):
    # assemblyai's transcription models are pydantic v1: no model_dump, but .json().
    import types

    item = types.SimpleNamespace(json=lambda: '{"id": "t2", "status": "queued"}')
    resp = mocker.MagicMock()
    resp.transcripts = [item]
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.return_value = resp
    rows = client.list_transcripts("sk", limit=5)
    assert rows == [{"id": "t2", "status": "queued"}]


def test_list_transcripts_auth_error_becomes_apierror(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError("nope")
    with pytest.raises(APIError):
        client.list_transcripts("sk")


def test_list_transcripts_rejected_key_becomes_not_authenticated(mocker):
    from aai_cli.errors import NotAuthenticated

    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError(
        "Authentication error, API token missing/invalid"
    )
    with pytest.raises(NotAuthenticated):
        client.list_transcripts("sk_bad")


def test_transcribe_blocks_and_returns_transcript(mocker):
    fake_transcript = mocker.MagicMock()
    fake_transcript.status = client.aai.TranscriptStatus.completed
    fake_transcriber = mocker.MagicMock()
    fake_transcriber.transcribe.return_value = fake_transcript

    cfg = aai.TranscriptionConfig(speaker_labels=True)
    mocker.patch.object(client.aai, "Transcriber", autospec=True, return_value=fake_transcriber)
    result = client.transcribe("sk", "audio.mp3", config=cfg)

    fake_transcriber.transcribe.assert_called_once_with("audio.mp3", config=cfg)
    assert result is fake_transcript


def test_transcribe_raises_on_error_status(mocker):
    fake_transcript = mocker.MagicMock()
    fake_transcript.status = client.aai.TranscriptStatus.error
    fake_transcript.error = "decode failed"
    fake_transcript.id = "t_err"
    fake_transcriber = mocker.MagicMock()
    fake_transcriber.transcribe.return_value = fake_transcript

    mocker.patch.object(client.aai, "Transcriber", autospec=True, return_value=fake_transcriber)
    with pytest.raises(APIError) as exc:
        client.transcribe("sk", "audio.mp3", config=aai.TranscriptionConfig())
    assert exc.value.transcript_id == "t_err"
    assert exc.value.message == "decode failed"  # surfaces the SDK's error verbatim


def test_transcribe_error_status_without_message_uses_fallback(mocker):
    # When the SDK reports an error status but no error text, fall back to a generic
    # message (pins the `transcript.error or "Transcription failed."`).
    fake_transcript = mocker.MagicMock()
    fake_transcript.status = client.aai.TranscriptStatus.error
    fake_transcript.error = None
    fake_transcript.id = "t_err"
    fake_transcriber = mocker.MagicMock()
    fake_transcriber.transcribe.return_value = fake_transcript

    mocker.patch.object(client.aai, "Transcriber", autospec=True, return_value=fake_transcriber)
    with pytest.raises(APIError) as exc:
        client.transcribe("sk", "audio.mp3", config=aai.TranscriptionConfig())
    assert exc.value.message == "Transcription failed."


def test_select_transcript_field_utterances_formats_speakers(mocker):
    import types

    t = mocker.MagicMock()
    t.utterances = [
        types.SimpleNamespace(speaker="A", text="hello"),
        types.SimpleNamespace(speaker="B", text="hi there"),
    ]
    assert (
        client.select_transcript_field(t, "utterances") == "Speaker A: hello\nSpeaker B: hi there"
    )


def test_select_transcript_field_utterances_falls_back_to_text_when_absent(mocker):
    t = mocker.MagicMock()
    t.utterances = []  # no diarization -> plain transcript text
    t.text = "plain transcript"
    assert client.select_transcript_field(t, "utterances") == "plain transcript"


def test_select_transcript_field_utterances_ignores_non_list_value(mocker):
    t = mocker.MagicMock()
    t.utterances = "bad"
    t.text = "plain transcript"
    assert client.select_transcript_field(t, "utterances") == "plain transcript"


def test_select_transcript_field_srt_uses_sdk(mocker):
    t = mocker.MagicMock()
    t.export_subtitles_srt.return_value = "1\n00:00:00,000 --> 00:00:02,000\nhello world\n"
    assert client.select_transcript_field(t, "srt") == (
        "1\n00:00:00,000 --> 00:00:02,000\nhello world\n"
    )
    t.export_subtitles_srt.assert_called_once_with()


def test_select_transcript_field_srt_network_error_becomes_apierror(mocker):
    t = mocker.MagicMock()
    t.export_subtitles_srt.side_effect = RuntimeError("connection reset")
    with pytest.raises(APIError):
        client.select_transcript_field(t, "srt")


def test_select_transcript_field_srt_auth_error_becomes_not_authenticated(mocker):
    from aai_cli.errors import NotAuthenticated

    t = mocker.MagicMock()
    t.export_subtitles_srt.side_effect = RuntimeError("HTTP 401 Unauthorized")
    with pytest.raises(NotAuthenticated):
        client.select_transcript_field(t, "srt")


def test_get_transcript_calls_sdk(mocker):
    fake = mocker.MagicMock()
    g = mocker.patch.object(client.aai.Transcript, "get_by_id", return_value=fake)
    result = client.get_transcript("sk", "t_123")
    g.assert_called_once_with("t_123")
    assert result is fake


def test_get_transcript_generic_error_becomes_apierror(mocker):
    mocker.patch.object(client.aai.Transcript, "get_by_id", side_effect=RuntimeError("boom"))
    with pytest.raises(APIError):
        client.get_transcript("sk", "t_x")


def test_get_transcript_auth_error_becomes_not_authenticated(mocker):
    from aai_cli.errors import NotAuthenticated

    mocker.patch.object(
        client.aai.Transcript, "get_by_id", side_effect=RuntimeError("HTTP 401 Unauthorized")
    )
    with pytest.raises(NotAuthenticated):
        client.get_transcript("sk_bad", "t_x")


def test_transcribe_network_error_becomes_apierror(mocker):
    fake_transcriber = mocker.MagicMock()
    fake_transcriber.transcribe.side_effect = RuntimeError("connection reset")
    mocker.patch.object(client.aai, "Transcriber", autospec=True, return_value=fake_transcriber)
    with pytest.raises(APIError):
        client.transcribe("sk", "audio.mp3", config=aai.TranscriptionConfig())


def test_transcribe_auth_error_becomes_not_authenticated(mocker):
    from aai_cli.errors import NotAuthenticated

    fake_transcriber = mocker.MagicMock()
    fake_transcriber.transcribe.side_effect = RuntimeError("Invalid API key")
    mocker.patch.object(client.aai, "Transcriber", autospec=True, return_value=fake_transcriber)
    with pytest.raises(NotAuthenticated):
        client.transcribe("sk_bad", "audio.mp3", config=aai.TranscriptionConfig())


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


def test_transcribe_passes_prebuilt_config(monkeypatch, mocker):
    import assemblyai as aai

    from aai_cli import client

    captured = {}

    class FakeTranscriber:
        def transcribe(self, audio, config=None):
            captured["audio"] = audio
            captured["config"] = config
            t = mocker.MagicMock()
            t.status = aai.TranscriptStatus.completed
            return t

    monkeypatch.setattr(aai, "Transcriber", lambda: FakeTranscriber())
    cfg = aai.TranscriptionConfig(speaker_labels=True)
    client.transcribe("sk", "audio.mp3", config=cfg)
    assert captured["audio"] == "audio.mp3"
    assert captured["config"] is cfg


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
