import json
import types

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def test_stream_llm_refreshes_live_over_growing_transcript(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {"texts": []}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hola", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript="mundo", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript="partial", end_of_turn=False))  # ignored
            on_turn(types.SimpleNamespace(transcript="no-eot"))  # missing flag -> not final

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["texts"].append(transcript_text)
        seen["prompts"] = prompts
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return f"answer:{transcript_text}"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(
        app,
        [
            "stream",
            "--llm",
            "translate to english",
            "--model",
            "gpt-4.1",
            "--max-tokens",
            "50",
            "--json",
        ],
    )
    assert result.exit_code == 0
    # One refresh per finalized turn, over the growing transcript (partials ignored).
    assert seen["texts"] == ["hola", "hola mundo"]
    assert seen["prompts"] == ["translate to english"]
    assert seen["model"] == "gpt-4.1"
    assert seen["max_tokens"] == 50
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"turns": 1, "output": "answer:hola"} in lines
    assert {"turns": 2, "output": "answer:hola mundo"} in lines
    # Live mode replaces the raw turn envelopes; only follow refreshes reach stdout.
    assert '"type"' not in result.output


def test_stream_llm_chains_multiple_prompts(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True))

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["prompts"] = prompts
        return "done"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(
        app, ["stream", "--llm", "summarize", "--llm", "translate to french", "--json"]
    )
    assert result.exit_code == 0
    assert seen["prompts"] == ["summarize", "translate to french"]


def test_stream_llm_rejects_output_text(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not stream")),
    )
    result = runner.invoke(app, ["stream", "--llm", "summarize", "-o", "text"])
    assert result.exit_code == 2  # --llm renders a panel/NDJSON; -o text is contradictory


def test_stream_without_prompt_does_not_transform(monkeypatch):
    config.set_api_key("default", "sk_live")
    called = {"ran": False}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True))

    def fake_run_chain(*a, **k):
        called["ran"] = True
        return "x"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    assert called["ran"] is False  # no --llm -> no gateway call


def test_stream_show_code_with_llm_emits_follow_loop(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not stream")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _boom)
    result = runner.invoke(app, ["stream", "--llm", "summarize", "--show-code"])
    assert result.exit_code == 0
    assert "from openai import OpenAI" in result.output
    assert "summarize" in result.output
    assert "run_chain" in result.output  # the live transcribe->LLM-per-turn loop
