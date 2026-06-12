"""Non-streaming client wrapper tests (transcribe, list, validate, fields).

client.stream_audio tests live in test_client_streaming.py.
"""

import assemblyai as aai
import pytest

from aai_cli import client
from aai_cli.errors import APIError


def test_validate_key_true_on_success(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.return_value = mocker.MagicMock()
    assert client.validate_key("sk_good") is True
    # The probe asks for a single row — it only needs to confirm the key authenticates.
    params = T.return_value.list_transcripts.call_args.args[0]
    assert params.limit == 1


def test_list_transcript_params_serialize_without_model_config():
    # assemblyai==0.64.4 + pydantic==2.13.4: the SDK's own ListTranscriptParameters
    # leaks a spurious `model_config` field into .dict(exclude_none=True) — exactly
    # what the SDK serializes onto the query string. The helper must drop it.
    params = client._list_transcript_params(3)
    assert params.dict(exclude_none=True) == {"limit": 3}


def test_validate_key_probe_serializes_without_model_config(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.return_value = mocker.MagicMock()
    assert client.validate_key("sk_good") is True
    params = T.return_value.list_transcripts.call_args.args[0]
    # No junk `model_config` query param on the wire (and still a one-row probe).
    assert params.dict(exclude_none=True) == {"limit": 1}


def test_list_transcripts_params_serialize_without_model_config(mocker):
    resp = mocker.MagicMock()
    resp.transcripts = []
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.return_value = resp
    assert client.list_transcripts("sk", limit=7) == []
    params = T.return_value.list_transcripts.call_args.args[0]
    assert params.dict(exclude_none=True) == {"limit": 7}


def test_validate_key_sdk_error_message_is_one_clean_line(mocker):
    # httpx-backed SDK failures embed a multi-line repr; the CLI error must keep the
    # reason but collapse it to one line and drop the `Request: <…>` tail.
    raw = (
        "failed to retrieve transcripts: \n"
        "Reason: [Errno -3] Temporary failure in name resolution\n"
        "Request: <Request('GET', 'https://api.assemblyai.com/v2/transcript?limit=1')>"
    )
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError(raw)
    with pytest.raises(APIError) as exc:
        client.validate_key("sk")
    assert exc.value.message == (
        "Could not validate key: failed to retrieve transcripts: "
        "Reason: [Errno -3] Temporary failure in name resolution"
    )


def test_validate_key_network_error_message_is_one_clean_line(mocker):
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = ConnectionError(
        "connection refused\nRequest: <Request('GET', 'https://api.assemblyai.com/x')>"
    )
    with pytest.raises(APIError) as exc:
        client.validate_key("sk")
    assert exc.value.message == "Network error contacting AssemblyAI: connection refused"


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


def test_list_transcripts_error_is_clean_without_request_repr_or_doubled_prefix(mocker):
    # The generic _sdk_errors wrap must compact the httpx repr ("Request: <…>")
    # and drop the SDK's own "failed to retrieve transcripts:" preamble, which
    # doubled up with the wrapper's "Could not list transcripts:" prefix.
    raw = (
        "failed to retrieve transcripts: \n"
        "Reason: Host not in allowlist\n"
        "Request: <Request('GET', 'https://api.assemblyai.com/v2/transcript?limit=10')>"
    )
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = aai.types.AssemblyAIError(raw)
    with pytest.raises(APIError) as exc:
        client.list_transcripts("sk")
    assert exc.value.message == "Could not list transcripts: Reason: Host not in allowlist"
    assert exc.value.suggestion == "Check your network and try again."


def test_get_transcript_error_strips_id_bearing_sdk_preamble(mocker):
    mocker.patch.object(
        client.aai.Transcript,
        "get_by_id",
        side_effect=RuntimeError("failed to retrieve transcript t_x: server exploded"),
    )
    with pytest.raises(APIError) as exc:
        client.get_transcript("sk", "t_x")
    assert exc.value.message == "Could not fetch transcript t_x: server exploded"


def test_sdk_error_without_preamble_keeps_reason_verbatim(mocker):
    # Reasons that don't carry the SDK's preamble pass through unchanged.
    T = mocker.patch.object(client.aai, "Transcriber", autospec=True)
    T.return_value.list_transcripts.side_effect = ValueError("connection reset by peer")
    with pytest.raises(APIError) as exc:
        client.list_transcripts("sk")
    assert exc.value.message == "Could not list transcripts: connection reset by peer"


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
