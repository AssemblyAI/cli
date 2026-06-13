"""`assembly llm --config KEY=VALUE` — the open-ended gateway-field escape hatch."""

import pytest
from typer.testing import CliRunner

from aai_cli.core import config, llm
from aai_cli.core.errors import UsageError
from aai_cli.main import app
from tests.test_llm import _fake_client, _response

runner = CliRunner()


# --- parse_gateway_overrides --------------------------------------------------


def test_overrides_parse_json_typed_values():
    assert llm.parse_gateway_overrides(
        ["temperature=0.2", 'stop=["END"]', "logprobs=true", "n=2"]
    ) == {"temperature": 0.2, "stop": ["END"], "logprobs": True, "n": 2}


def test_overrides_fall_back_to_literal_strings():
    # Not valid JSON -> the literal text, so enum-ish values need no quoting.
    assert llm.parse_gateway_overrides(["reasoning_effort=low"]) == {"reasoning_effort": "low"}


def test_overrides_key_is_stripped_and_empty_value_allowed():
    assert llm.parse_gateway_overrides([" user =alex"]) == {"user": "alex"}


def test_overrides_reject_pair_without_equals():
    with pytest.raises(UsageError) as exc:
        llm.parse_gateway_overrides(["temperature"])
    assert "KEY=VALUE" in exc.value.message
    assert "'temperature'" in exc.value.message
    assert "temperature=0.2" in (exc.value.suggestion or "")


def test_overrides_reject_empty_key():
    with pytest.raises(UsageError):
        llm.parse_gateway_overrides(["=0.2"])


def test_overrides_empty_input_yields_empty_dict():
    assert llm.parse_gateway_overrides([]) == {}


# --- complete() plumbing -------------------------------------------------------


def test_complete_merges_extra_with_transcript_id(monkeypatch):
    seen = _fake_client(monkeypatch, result=_response())
    llm.complete(
        "sk",
        model="m",
        messages=[],
        transcript_id="t_42",
        extra={"temperature": 0.2},
    )
    assert seen["extra_body"] == {"transcript_id": "t_42", "temperature": 0.2}


def test_complete_without_extras_sends_no_extra_body(monkeypatch):
    seen = _fake_client(monkeypatch, result=_response())
    llm.complete("sk", model="m", messages=[], extra={})
    assert seen["extra_body"] is None


# --- command wiring ------------------------------------------------------------


def test_llm_config_flags_reach_the_gateway(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["extra"] = extra
        return _response("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app,
        [
            "llm",
            "hi",
            "--config",
            "temperature=0.2",
            "--config",
            "reasoning_effort=low",
            "-o",
            "text",
        ],
    )
    assert result.exit_code == 0
    assert seen["extra"] == {"temperature": 0.2, "reasoning_effort": "low"}


def test_llm_bad_config_pair_fails_before_any_request(monkeypatch):
    config.set_api_key("default", "sk_live")
    called = []
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: called.append(1),
    )
    result = runner.invoke(app, ["llm", "hi", "--config", "broken"])
    assert result.exit_code == 2
    assert "KEY=VALUE" in result.output
    assert called == []


def test_llm_follow_passes_config_to_every_refresh(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen.append(extra)
        return _response("ans")

    monkeypatch.setattr("aai_cli.commands.llm._exec.gateway.complete", fake_complete)
    result = runner.invoke(
        app,
        ["llm", "-f", "summarize", "--config", "temperature=0.1", "--json"],
        input="turn one\nturn two\n",
    )
    assert result.exit_code == 0
    assert seen == [{"temperature": 0.1}, {"temperature": 0.1}]
