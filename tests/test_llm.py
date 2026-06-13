import types
from typing import cast

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from aai_cli.core import environments, llm
from aai_cli.core.errors import APIError, NotAuthenticated

_GATEWAY_BASE = environments.get(environments.DEFAULT_ENV).llm_gateway_base
_REQUEST = httpx.Request("POST", f"{_GATEWAY_BASE}/chat/completions")


def _response(content: "str | None" = "hi there", usage=None) -> ChatCompletion:
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return cast(ChatCompletion, types.SimpleNamespace(choices=[choice], usage=usage))


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


def test_client_targets_active_gateway_base():
    from aai_cli.core import environments

    client = llm._client("sk_live")
    assert str(client.base_url).rstrip("/") == environments.active().llm_gateway_base.rstrip("/")


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


def test_complete_bad_key_401_raises_clean_auth_failure(monkeypatch):
    # A plain 401 with no entitlement hint is just a rejected key: surface the same
    # clean exit-4 auth_failure transcribe gives, not the gateway's raw 401 body.
    err = openai.AuthenticationError(
        "Authentication error, API token missing/invalid",
        response=httpx.Response(401, request=_REQUEST),
        body=None,
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(NotAuthenticated) as exc:
        llm.complete("sk", model="m", messages=[])
    assert exc.value.exit_code == 4
    assert exc.value.rejected_key is True
    # Not the raw gateway passthrough (which would leak the 401 body).
    assert "access denied" not in exc.value.message


def test_complete_entitlement_401_still_points_at_billing(monkeypatch):
    # A 401 that *does* read as an entitlement block keeps the exit-1 billing pointer
    # rather than the rejected-key path.
    err = openai.AuthenticationError(
        "Your account has no access to the LLM Gateway.",
        response=httpx.Response(401, request=_REQUEST),
        body=None,
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(APIError, match="access denied") as exc:
        llm.complete("sk", model="m", messages=[])
    assert exc.value.suggestion is not None and "paid plan" in exc.value.suggestion


def test_complete_entitlement_denial_suggests_paid_plan(monkeypatch):
    # The gateway's own entitlement block ("no access to LLM Gateway") is the one
    # 401/403 where pointing at billing is right.
    err = openai.PermissionDeniedError(
        "Your account has no access to the LLM Gateway.",
        response=httpx.Response(403, request=_REQUEST),
        body=None,
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(APIError, match="access denied") as exc:
        llm.complete("sk", model="m", messages=[])
    assert exc.value.suggestion is not None and "paid plan" in exc.value.suggestion


def test_complete_proxy_denial_does_not_suggest_paid_plan(monkeypatch):
    # A corporate-proxy 403 says nothing about plans; sending the user to billing
    # would mislead — they need to look at their key/network instead.
    err = openai.PermissionDeniedError(
        "Host not in allowlist", response=httpx.Response(403, request=_REQUEST), body=None
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(APIError, match="access denied") as exc:
        llm.complete("sk", model="m", messages=[])
    assert exc.value.suggestion == llm._ACCESS_DENIED_SUGGESTION


@pytest.mark.parametrize("hint", ["entitlement", "plan", "upgrade", "billing", "no access"])
def test_denial_suggestion_matches_each_entitlement_hint(hint):
    assert (
        llm._denial_suggestion(Exception(f"denied: {hint} required")) == llm._PAID_PLAN_SUGGESTION
    )


class _DenialWithBody(Exception):
    def __init__(self, message: str, body: object) -> None:
        super().__init__(message)
        self.body = body


def test_denial_suggestion_reads_the_response_body_too():
    # The entitlement marker can live in the structured body rather than str(exc).
    exc = _DenialWithBody("403 Forbidden", body={"error": "upgrade your plan"})
    assert llm._denial_suggestion(exc) == llm._PAID_PLAN_SUGGESTION


def test_complete_bad_request_maps_to_api_error(monkeypatch):
    err = openai.BadRequestError(
        "missing model", response=httpx.Response(400, request=_REQUEST), body=None
    )
    _fake_client(monkeypatch, error=err)
    with pytest.raises(APIError) as exc:
        llm.complete("sk", model="m", messages=[])
    assert exc.value.suggestion is not None and "network" in exc.value.suggestion


def test_complete_connection_error_maps_to_api_error(monkeypatch):
    _fake_client(monkeypatch, error=openai.APIConnectionError(request=_REQUEST))
    with pytest.raises(APIError):
        llm.complete("sk", model="m", messages=[])


def test_content_of_missing_raises():
    with pytest.raises(APIError):
        llm.content_of(cast(ChatCompletion, types.SimpleNamespace(choices=[])))


def test_content_of_none_returns_empty():
    assert llm.content_of(_response(content=None)) == ""


def test_usage_of_variants():
    # The OpenAI SDK always deserializes `usage` into a CompletionUsage (a
    # pydantic model with model_dump) or None — never a raw dict.
    assert llm.usage_of(_response(usage=None)) is None
    model = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 9})
    assert llm.usage_of(_response(usage=model)) == {"total_tokens": 9}


def test_build_messages_transcript_id_uses_tag():
    msgs = llm.build_messages("summarize", transcript_id="t_1")
    assert msgs == [{"role": "user", "content": f"summarize\n\n{llm.TRANSCRIPT_TAG}"}]


def test_build_messages_inline_text():
    msgs = llm.build_messages("summarize", transcript_text="hello world")
    assert msgs == [{"role": "user", "content": "summarize\n\nTranscript:\nhello world"}]


def test_build_messages_with_system_prompt():
    msgs = llm.build_messages("hi", system="be terse")
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_transform_transcript_roundtrips(monkeypatch):
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["transcript_id"] = transcript_id
        seen["messages"] = messages
        return _response("SUMMARY")

    monkeypatch.setattr(llm, "complete", fake_complete)
    out = llm.transform_transcript("sk", prompt="summarize", transcript_id="t_9")
    assert out == "SUMMARY"
    assert seen["transcript_id"] == "t_9"
    assert llm.TRANSCRIPT_TAG in seen["messages"][0]["content"]


def test_run_chain_single_prompt_runs_over_transcript(monkeypatch):
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["messages"] = messages
        seen["transcript_id"] = transcript_id
        return _response("SUMMARY")

    monkeypatch.setattr(llm, "complete", fake_complete)
    out = llm.run_chain("sk", ["summarize"], transcript_text="hola mundo", model="m", max_tokens=50)
    assert out == "SUMMARY"
    # No transcript_id in live mode -> the text is inlined into the prompt.
    assert seen["transcript_id"] is None
    content = seen["messages"][-1]["content"]
    assert "summarize" in content and "hola mundo" in content


def test_run_chain_threads_output_through_prompts(monkeypatch):
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        calls.append(messages[-1]["content"])
        return _response(f"out{len(calls)}")

    monkeypatch.setattr(llm, "complete", fake_complete)
    out = llm.run_chain(
        "sk",
        ["summarize", "translate to french"],
        transcript_text="hola mundo",
        model="m",
        max_tokens=50,
    )
    assert out == "out2"  # final step's output
    assert len(calls) == 2
    assert "summarize" in calls[0] and "hola mundo" in calls[0]
    # Second prompt runs over the FIRST step's output, not the transcript.
    assert "translate to french" in calls[1] and "out1" in calls[1]


def test_run_chain_empty_prompts_do_not_call_gateway(monkeypatch):
    def fail_complete(*args, **kwargs):
        raise AssertionError("gateway should not be called")

    monkeypatch.setattr(llm, "complete", fail_complete)
    assert llm.run_chain_steps("sk", []) == []
    assert llm.run_chain("sk", [], transcript_text="hola mundo") == ""


def test_run_chain_steps_uses_transcript_id_then_prior_output(monkeypatch):
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        calls.append({"content": messages[-1]["content"], "transcript_id": transcript_id})
        return _response(f"out{len(calls)}")

    monkeypatch.setattr(llm, "complete", fake_complete)
    steps = llm.run_chain_steps(
        "sk",
        ["summarize", "translate"],
        transcript_id="t_1",
        model="m",
        max_tokens=50,
    )
    assert steps == [
        {"prompt": "summarize", "output": "out1"},
        {"prompt": "translate", "output": "out2"},
    ]
    assert calls[0]["transcript_id"] == "t_1"
    assert llm.TRANSCRIPT_TAG in calls[0]["content"]
    assert calls[1]["transcript_id"] is None
    assert "out1" in calls[1]["content"]
