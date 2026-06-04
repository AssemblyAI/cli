import types

import httpx
import openai
import pytest

from assemblyai_cli import llm
from assemblyai_cli.errors import APIError, NotAuthenticated

_REQUEST = httpx.Request("POST", f"{llm.GATEWAY_BASE_URL}/chat/completions")


def _response(content: "str | None" = "hi there", usage=None):
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=usage)


class FakeCompletions:
    def __init__(self, result=None, error=None, seen=None):
        self._result = result
        self._error = error
        self._seen = seen if seen is not None else {}

    def create(self, **kwargs):
        self._seen.update(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


def _fake_client(monkeypatch, *, result=None, error=None):
    seen = {}
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions(result, error, seen))
    )
    monkeypatch.setattr(llm, "_client", lambda api_key: client)
    return seen


def test_complete_sends_model_and_messages(monkeypatch):
    seen = _fake_client(monkeypatch, result=_response("hi there"))
    resp = llm.complete(
        "sk_live", model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}]
    )
    assert llm.content_of(resp) == "hi there"
    assert seen["model"] == "claude-sonnet-4-6"
    assert seen["messages"] == [{"role": "user", "content": "x"}]
    assert seen["extra_body"] is None  # no transcript_id -> no extra body


def test_complete_passes_transcript_id_as_extra_body(monkeypatch):
    seen = _fake_client(monkeypatch, result=_response())
    llm.complete("sk", model="m", messages=[], transcript_id="t_42")
    assert seen["extra_body"] == {"transcript_id": "t_42"}


def test_complete_auth_error_maps_to_not_authenticated(monkeypatch):
    err = openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=_REQUEST), body=None
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(NotAuthenticated):
        llm.complete("sk", model="m", messages=[])


def test_complete_permission_error_maps_to_not_authenticated(monkeypatch):
    err = openai.PermissionDeniedError(
        "forbidden", response=httpx.Response(403, request=_REQUEST), body=None
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(NotAuthenticated):
        llm.complete("sk", model="m", messages=[])


def test_complete_bad_request_maps_to_api_error(monkeypatch):
    err = openai.BadRequestError(
        "missing model", response=httpx.Response(400, request=_REQUEST), body=None
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(APIError):
        llm.complete("sk", model="m", messages=[])


def test_complete_connection_error_maps_to_api_error(monkeypatch):
    _fake_client(monkeypatch, error=openai.APIConnectionError(request=_REQUEST))
    with pytest.raises(APIError):
        llm.complete("sk", model="m", messages=[])


def test_content_of_missing_raises():
    with pytest.raises(APIError):
        llm.content_of(types.SimpleNamespace(choices=[]))


def test_content_of_none_returns_empty():
    assert llm.content_of(_response(content=None)) == ""


def test_usage_of_variants():
    assert llm.usage_of(_response(usage=None)) is None
    assert llm.usage_of(_response(usage={"total_tokens": 5})) == {"total_tokens": 5}
    model = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 9})
    assert llm.usage_of(_response(usage=model)) == {"total_tokens": 9}


def test_build_messages_transcript_id_uses_tag():
    msgs = llm.build_messages("summarize", transcript_id="t_1")
    assert msgs == [{"role": "user", "content": f"summarize\n\n{llm.TRANSCRIPT_TAG}"}]


def test_build_messages_inline_text():
    msgs = llm.build_messages("summarize", transcript_text="hello world")
    assert msgs[0]["content"] == "summarize\n\nTranscript:\nhello world"


def test_build_messages_with_system_prompt():
    msgs = llm.build_messages("hi", system="be terse")
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_transform_transcript_roundtrips(monkeypatch):
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["transcript_id"] = transcript_id
        seen["messages"] = messages
        return _response("SUMMARY")

    monkeypatch.setattr(llm, "complete", fake_complete)
    out = llm.transform_transcript("sk", prompt="summarize", transcript_id="t_9")
    assert out == "SUMMARY"
    assert seen["transcript_id"] == "t_9"
    assert llm.TRANSCRIPT_TAG in seen["messages"][0]["content"]
