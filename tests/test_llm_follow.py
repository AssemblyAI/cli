import json
import types

from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_api_key("default", "sk_live")


def _payload(content="four"):
    # Mimics the OpenAI SDK response object the command reads via content_of/usage_of.
    # `usage` is a CompletionUsage-like model (model_dump), not a raw dict.
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    usage = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 3})
    return types.SimpleNamespace(choices=[choice], usage=usage)


def test_llm_follow_summarizes_each_turn(monkeypatch):
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        calls.append(messages[-1]["content"])
        return _payload(f"summary-{len(calls)}")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app,
        ["llm", "summarize action items", "--follow", "--json"],
        input="we ship friday\nbob owns the deploy\n",
    )
    assert result.exit_code == 0
    updates = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    # One update per finalized turn, full transcript accumulating each time.
    assert len(updates) == 2
    assert "we ship friday" in calls[0]
    assert "bob owns the deploy" not in calls[0]
    assert "we ship friday" in calls[1]
    assert "bob owns the deploy" in calls[1]
    assert updates[-1]["output"] == "summary-2"
    assert updates[-1]["turns"] == 2


def test_llm_follow_includes_system_prompt(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        seen["roles"] = [m["role"] for m in messages]
        seen["system"] = messages[0]["content"]
        return _payload("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app,
        ["llm", "summarize", "--follow", "--system", "You are a scribe", "--json"],
        input="one turn\n",
    )
    assert result.exit_code == 0
    assert seen["roles"][0] == "system"
    assert seen["system"] == "You are a scribe"


def test_llm_follow_rejects_transcript_id(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(
        app,
        ["llm", "summarize", "--follow", "--transcript-id", "t_1", "--json"],
        input="x\n",
    )
    assert result.exit_code == 2
    assert "transcript-id" in result.output


def test_llm_follow_rejects_file_arguments(monkeypatch, tmp_path):
    # --follow runs over live stdin; a file argument has no meaning there, so reject
    # it rather than silently ignore it.
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    note = tmp_path / "note.md"
    note.write_text("x")
    result = runner.invoke(app, ["llm", "summarize", str(note), "--follow", "--json"], input="x\n")
    assert result.exit_code == 2
    assert "file arguments" in result.output


def test_llm_follow_ignores_blank_lines(monkeypatch):
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        calls.append(messages[-1]["content"])
        return _payload("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app,
        ["llm", "summarize", "--follow", "--json"],
        input="first\n\n   \nsecond\n",
    )
    assert result.exit_code == 0
    # Blank/whitespace-only lines don't trigger a call.
    assert len(calls) == 2


def test_llm_output_with_follow_is_rejected(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "hi", "-f", "-o", "text"], input="x\n")
    assert result.exit_code == 2
    assert "one-shot" in result.output


def test_llm_follow_requires_a_prompt(monkeypatch):
    # --follow re-runs a prompt over each turn; with no prompt there's nothing to run.
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "--follow", "--json"], input="x\n")
    assert result.exit_code == 2
    assert "prompt" in result.output.lower()


def test_llm_follow_requires_piped_stdin(monkeypatch):
    # Interactively (no pipe) --follow would block forever; reject it with guidance.
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm._exec.stdio.stdin_is_piped", lambda: False)
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "summarize", "--follow", "--json"])
    assert result.exit_code == 2
    assert "stdin" in result.output.lower()


def test_llm_follow_empty_stdin_exits_2(monkeypatch):
    # `assembly llm -f "…" </dev/null` must not exit 0 silently: an empty pipe means the
    # prompt never ran, which is a usage error, not a success.
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        calls.append(messages)
        return _payload("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "summarize", "--follow", "--json"], input="")
    assert result.exit_code == 2
    assert "--follow needs transcript text piped on stdin" in result.output
    assert calls == []  # no API call was made


def test_llm_follow_interrupt_before_first_turn_exits_cancel(monkeypatch):
    # Ctrl-C before any turn arrives is a cancel (exit 130), not the empty-stdin usage
    # error — the user stopped watching, they didn't misuse the flag.
    _auth()

    class _InterruptIter:
        def __iter__(self):
            return self

        def __next__(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "aai_cli.commands.llm._exec.stdio.iter_piped_stdin_lines", lambda: _InterruptIter()
    )
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "summarize", "--follow", "--json"], input="")
    assert result.exit_code == 130
    assert "--follow needs transcript text piped on stdin" not in result.output


def test_llm_follow_stops_cleanly_on_interrupt(monkeypatch):
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None, extra=None):
        calls.append(messages[-1]["content"])
        if len(calls) == 2:
            raise KeyboardInterrupt  # user hits Ctrl-C mid-meeting
        return _payload("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app, ["llm", "summarize", "--follow", "--json"], input="alpha\nbeta\ngamma\n"
    )
    # Ctrl-C is a cancel (exit 130), not a clean finish.
    assert result.exit_code == 130
    updates = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(updates) == 1
    assert updates[0]["turns"] == 1
