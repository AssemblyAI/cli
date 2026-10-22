"""The Dictation API HTTP boundary (aai_cli/core/dictation.py): the documented
request shape (config part first), error normalization across both error body
shapes, and response parsing."""

import dataclasses

import httpx2 as httpx
import pytest

from aai_cli.core import dictation, environments
from aai_cli.core.errors import APIError, NotAuthenticated

OK_BODY = {
    "text": "Um, I'm calling about, uh, my order.",
    "llm_response": "I'm calling about my order.",
    "llm_error": None,
    "confidence": 0.87,
    "audio_duration_ms": 1500,
    "session_id": "eb92c4ff",
}


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client
    seen_kwargs = {}

    def fake_client(*args, **kwargs):
        seen_kwargs.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(dictation.httpx, "Client", fake_client)
    return seen_kwargs


def _ok_handler(seen, body=OK_BODY):
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        return httpx.Response(200, json=body)

    return handler


def _respond(status, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **kwargs)

    return handler


def test_posts_config_then_pcm_to_the_live_endpoint(monkeypatch):
    seen = {}
    client_kwargs = _patch_transport(monkeypatch, _ok_handler(seen))
    result = dictation.transcribe_live("sk_key_pcm", b"\x01\x02pcm-bytes", sample_rate=16000)
    assert seen["url"] == "https://dictation.assemblyai.com/v1/transcribe/live"
    # The raw key, and no Sync-era model header.
    assert seen["auth"] == "sk_key_pcm"
    assert "x-aai-model" not in seen["headers"]
    body = seen["body"]
    # The config part must precede the audio part, or the live endpoint answers 400.
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert b"application/json" in body
    assert b"\x01\x02pcm-bytes" in body
    assert b"audio/pcm" in body
    assert b'{"sample_rate": 16000, "channels": 1}' in body
    # Generous timeout: the upload can carry up to 2 minutes of audio, then the rewrite.
    assert client_kwargs["timeout"] == 90.0
    assert result == dictation.DictationResult(
        text="Um, I'm calling about, uh, my order.",
        llm_response="I'm calling about my order.",
        llm_error=None,
        confidence=0.87,
        audio_duration_ms=1500,
        session_id="eb92c4ff",
    )


def test_targets_the_sandbox_host_when_active(monkeypatch):
    seen = {}
    _patch_transport(monkeypatch, _ok_handler(seen))
    environments.set_active(environments.get("sandbox000"))
    dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert seen["url"] == "https://dictation.sandbox000.assemblyai-labs.com/v1/transcribe/live"


def test_config_fields_use_the_documented_names_when_set(monkeypatch):
    seen = {}
    _patch_transport(monkeypatch, _ok_handler(seen))
    config = dictation.DictationConfig(
        language_codes=["en", "es"],
        stt_prompt="A doctor dictating a visit note.",
        keyterms=["AssemblyAI", "LeMUR"],
        llm_instruction="Format as a bulleted list",
        rewrite_mode="smart",
    )
    dictation.transcribe_live("sk", b"pcm", sample_rate=44100, channels=2, config=config)
    assert (
        b'{"sample_rate": 44100, "channels": 2, "language_codes": ["en", "es"], '
        b'"stt_prompt": "A doctor dictating a visit note.", '
        b'"keyterms_prompt": ["AssemblyAI", "LeMUR"], '
        b'"llm_instruction": "Format as a bulleted list", "rewrite_mode": "smart"}'
    ) in seen["body"]


def test_result_is_immutable(monkeypatch):
    _patch_transport(monkeypatch, _ok_handler({}))
    result = dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    field_name = dataclasses.fields(result)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, field_name, "mutated")


def test_final_text_prefers_the_rewrite():
    result = dictation.DictationResult("raw", "clean", None, None, None, None)
    assert result.final_text == "clean"


def test_final_text_falls_back_to_the_transcript_when_the_rewrite_failed():
    result = dictation.DictationResult("raw", None, "timeout", None, None, None)
    assert result.final_text == "raw"


def test_missing_optional_response_fields_parse_as_none(monkeypatch):
    _patch_transport(monkeypatch, _respond(200, json={"text": "bare"}))
    result = dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert result == dictation.DictationResult(
        text="bare",
        llm_response=None,
        llm_error=None,
        confidence=None,
        audio_duration_ms=None,
        session_id=None,
    )


def test_failed_rewrite_is_still_a_result(monkeypatch):
    body = {"text": "raw", "llm_response": None, "llm_error": "timeout"}
    _patch_transport(monkeypatch, _respond(200, json=body))
    result = dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert (result.llm_response, result.llm_error) == (None, "timeout")


@pytest.mark.parametrize("status", [401, 403])
def test_auth_rejection_raises_not_authenticated(monkeypatch, status):
    _patch_transport(monkeypatch, _respond(status, json={"detail": "Invalid API key"}))
    with pytest.raises(NotAuthenticated) as exc:
        dictation.transcribe_live("bad", b"pcm", sample_rate=16000)
    assert exc.value.rejected_key is True


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_rate_limit_capacity_and_upstream_are_retryable(monkeypatch, status):
    body = {"error_code": "capacity_exceeded", "error": "server at cap"}
    _patch_transport(monkeypatch, _respond(status, json=body))
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert exc.value.message == (
        f"The Dictation API is busy ({status}): server at cap (capacity_exceeded)"
    )
    assert exc.value.suggestion == "Wait a moment and try again."


def test_dictation_error_shape_carries_error_and_code(monkeypatch):
    body = {"error_code": "bad_request", "error": "`config` must be sent before `audio`"}
    _patch_transport(monkeypatch, _respond(400, json=body))
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert exc.value.message == (
        "Dictation failed (400): `config` must be sent before `audio` (bad_request)"
    )


def test_detail_shape_wins_over_error(monkeypatch):
    # Docs: check `detail` first, falling back to `error`.
    body = {"status": 400, "title": "Bad Request", "detail": "unknown field", "error": "x"}
    _patch_transport(monkeypatch, _respond(400, json=body))
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert exc.value.message == "Dictation failed (400): unknown field"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"upstream proxy says no", "upstream proxy says no"),  # non-JSON body
        (b"", "HTTP 500"),  # empty body -> bare status
        (b'["weird"]', '["weird"]'),  # JSON but not an object -> raw text
        (b'{"unrelated": true}', '{"unrelated": true}'),  # object without detail/error
    ],
)
def test_error_detail_falls_back_to_raw_body_or_status(monkeypatch, body, expected):
    _patch_transport(monkeypatch, _respond(500, content=body))
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert exc.value.message == f"Dictation failed (500): {expected}"


def test_success_with_unparseable_body_is_a_clean_api_error(monkeypatch):
    _patch_transport(monkeypatch, _respond(200, content=b"not-json"))
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert "not valid JSON (HTTP 200)" in exc.value.message


@pytest.mark.parametrize("payload", [{"words": []}, ["list"]])
def test_success_without_text_is_a_clean_api_error(monkeypatch, payload):
    _patch_transport(monkeypatch, _respond(200, json=payload))
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert "unexpected response shape" in exc.value.message


def test_network_failure_is_a_clean_api_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        dictation.transcribe_live("sk", b"pcm", sample_rate=16000)
    assert "Could not reach the Dictation API" in exc.value.message
    assert "network" in (exc.value.suggestion or "")


def test_config_is_immutable():
    config = dictation.DictationConfig()
    field_name = dataclasses.fields(config)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, field_name, "mutated")
