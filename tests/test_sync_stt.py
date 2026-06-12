"""The Sync STT HTTP boundary: request shape, error normalization, parsing."""

import dataclasses

import httpx2 as httpx
import pytest

from aai_cli import environments, sync_stt
from aai_cli.errors import APIError, NotAuthenticated


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client
    seen_kwargs = {}

    def fake_client(*args, **kwargs):
        seen_kwargs.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sync_stt.httpx, "Client", fake_client)
    return seen_kwargs


def _ok_handler(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["model"] = request.headers.get("x-aai-model")
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "text": "Hi, I'm calling about my order.",
                "confidence": 0.87,
                "audio_duration_ms": 1500,
                "session_id": "eb92c4ff",
            },
        )

    return handler


def test_posts_pcm_and_config_to_the_active_environment(monkeypatch):
    seen = {}
    client_kwargs = _patch_transport(monkeypatch, _ok_handler(seen))
    result = sync_stt.transcribe_pcm("sk_key_pcm", b"\x01\x02pcm-bytes", sample_rate=16000)
    assert seen["url"] == "https://sync.assemblyai.com/transcribe"
    assert seen["auth"] == "sk_key_pcm"
    assert seen["model"] == "u3-sync-pro"
    # Multipart body: a raw-PCM audio part plus the JSON config part.
    assert b"\x01\x02pcm-bytes" in seen["body"]
    assert b"audio/pcm" in seen["body"]
    assert b'"sample_rate": 16000' in seen["body"]
    assert b'"channels": 1' in seen["body"]
    # Optional knobs are omitted entirely when unset.
    assert b"language_code" not in seen["body"]
    assert b"prompt" not in seen["body"]
    assert b"word_boost" not in seen["body"]
    # Generous timeout: the upload can carry up to 2 minutes of audio.
    assert client_kwargs["timeout"] == 60.0
    assert result == sync_stt.SyncTranscript(
        text="Hi, I'm calling about my order.",
        confidence=0.87,
        audio_duration_ms=1500,
        session_id="eb92c4ff",
    )


def test_targets_the_sandbox_host_when_active(monkeypatch):
    seen = {}
    _patch_transport(monkeypatch, _ok_handler(seen))
    environments.set_active(environments.get("sandbox000"))
    sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert seen["url"] == "https://sync.sandbox000.assemblyai-labs.com/transcribe"


def test_optional_config_fields_are_sent_when_set(monkeypatch):
    seen = {}
    _patch_transport(monkeypatch, _ok_handler(seen))
    sync_stt.transcribe_pcm(
        "sk",
        b"pcm",
        sample_rate=44100,
        channels=2,
        language_code=["en", "es"],
        prompt="Transcribe verbatim.",
        word_boost=["AssemblyAI", "LeMUR"],
    )
    assert b'"sample_rate": 44100' in seen["body"]
    assert b'"channels": 2' in seen["body"]
    assert b'"language_code": ["en", "es"]' in seen["body"]
    assert b'"prompt": "Transcribe verbatim."' in seen["body"]
    assert b'"word_boost": ["AssemblyAI", "LeMUR"]' in seen["body"]


def test_single_language_code_is_sent_as_a_string(monkeypatch):
    seen = {}
    _patch_transport(monkeypatch, _ok_handler(seen))
    sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000, language_code="es")
    assert b'"language_code": "es"' in seen["body"]


def test_sync_transcript_is_immutable(monkeypatch):
    seen = {}
    _patch_transport(monkeypatch, _ok_handler(seen))
    result = sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    field_name = dataclasses.fields(result)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, field_name, "mutated")


def test_missing_optional_response_fields_parse_as_none(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "bare"})

    _patch_transport(monkeypatch, handler)
    result = sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert result == sync_stt.SyncTranscript(
        text="bare", confidence=None, audio_duration_ms=None, session_id=None
    )


@pytest.mark.parametrize("status", [401, 403])
def test_auth_rejection_raises_not_authenticated(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "Invalid API key"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated) as exc:
        sync_stt.transcribe_pcm("bad", b"pcm", sample_rate=16000)
    assert exc.value.rejected_key is True


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_and_capacity_are_retryable_api_errors(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error_code": "capacity_exceeded", "message": "server at cap"}
        )

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert "busy" in exc.value.message
    assert str(status) in exc.value.message
    assert "server at cap (capacity_exceeded)" in exc.value.message
    assert "try again" in (exc.value.suggestion or "")


def test_audio_error_carries_error_code_and_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error_code": "audio_too_short", "message": "audio below 80 ms"}
        )

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert "Sync transcription failed (400)" in exc.value.message
    assert "audio below 80 ms (audio_too_short)" in exc.value.message


def test_error_detail_reads_detail_field(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "missing audio part"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert "missing audio part" in exc.value.message


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"upstream proxy says no", "upstream proxy says no"),  # non-JSON body
        (b"", "HTTP 500"),  # empty body -> bare status
        (b'["weird"]', "weird"),  # JSON but not an object -> raw text
        (b'{"unrelated": true}', '{"unrelated": true}'),  # object without message/detail
    ],
)
def test_error_detail_falls_back_to_raw_body_or_status(monkeypatch, body, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert expected in exc.value.message


def test_success_with_unparseable_body_is_a_clean_api_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert "not valid JSON" in exc.value.message


@pytest.mark.parametrize("payload", [{"words": []}, ["list"]])
def test_success_without_transcript_text_is_a_clean_api_error(monkeypatch, payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert "unexpected response shape" in exc.value.message


def test_network_failure_is_a_clean_api_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        sync_stt.transcribe_pcm("sk", b"pcm", sample_rate=16000)
    assert "Could not reach the Sync API" in exc.value.message
    assert "network" in (exc.value.suggestion or "")
