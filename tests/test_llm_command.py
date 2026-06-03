import json
import types

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_api_key("default", "sk_live")


def _payload(content="four"):
    # Mimics the OpenAI SDK response object the command reads via content_of/usage_of.
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage={"total_tokens": 3})


def test_llm_help_lists_command():
    result = runner.invoke(app, ["llm", "--help"])
    assert result.exit_code == 0
    assert "gateway" in result.output.lower()


def test_llm_list_models_exits_without_network(monkeypatch):
    called = {"ran": False}
    monkeypatch.setattr(
        "assemblyai_cli.commands.llm.gateway.complete",
        lambda *a, **k: called.__setitem__("ran", True),
    )
    result = runner.invoke(app, ["llm", "--list-models"])
    assert result.exit_code == 0
    assert "claude-sonnet-4-6" in result.output
    assert called["ran"] is False


def test_llm_sends_prompt_and_prints_output(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["model"] = model
        seen["messages"] = messages
        seen["transcript_id"] = transcript_id
        return _payload("4")

    monkeypatch.setattr("assemblyai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "What is 2+2?", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["output"] == "4"
    assert data["model"] == "claude-sonnet-4-6"
    assert seen["transcript_id"] is None
    assert seen["messages"][0]["content"] == "What is 2+2?"


def test_llm_transcript_id_injected(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["transcript_id"] = transcript_id
        seen["content"] = messages[0]["content"]
        return _payload("summary")

    monkeypatch.setattr("assemblyai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "summarize", "--transcript-id", "t_7", "--json"])
    assert result.exit_code == 0
    assert seen["transcript_id"] == "t_7"
    assert "{{ transcript }}" in seen["content"]


def test_llm_missing_prompt_exits_2(monkeypatch):
    _auth()
    monkeypatch.setattr("assemblyai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm"])
    assert result.exit_code == 2


def test_llm_unauthenticated_exits_2():
    result = runner.invoke(app, ["llm", "hello"])
    assert result.exit_code == 2


def test_llm_passes_model_and_max_tokens(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return _payload()

    monkeypatch.setattr("assemblyai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app, ["llm", "hi", "--model", "gemini-2.5-flash", "--max-tokens", "42", "--json"]
    )
    assert result.exit_code == 0
    assert seen["model"] == "gemini-2.5-flash"
    assert seen["max_tokens"] == 42
