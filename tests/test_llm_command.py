import json
import types

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.llm import KNOWN_MODELS
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_api_key("default", "sk_live")


def _login_result(*, json_mode=False):
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=7
    )


def _payload(content="four"):
    # Mimics the OpenAI SDK response object the command reads via content_of/usage_of.
    # `usage` is a CompletionUsage-like model (model_dump), not a raw dict.
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    usage = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 3})
    return types.SimpleNamespace(choices=[choice], usage=usage)


def test_llm_help_lists_command():
    result = runner.invoke(app, ["llm", "--help"])
    assert result.exit_code == 0
    assert "gateway" in result.output.lower()


def test_llm_list_models_exits_without_network(monkeypatch):
    called = {"ran": False}
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: called.__setitem__("ran", True),
    )
    result = runner.invoke(app, ["llm", "--list-models"])
    assert result.exit_code == 0
    assert "claude-sonnet-4-6" in result.output
    # Human mode prints the bare newline-joined list, not a JSON array.
    assert result.output.strip() == "\n".join(KNOWN_MODELS)
    assert called["ran"] is False


def test_llm_list_models_json_emits_machine_readable_array(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the gateway")),
    )
    result = runner.invoke(app, ["llm", "--list-models", "--json"])
    assert result.exit_code == 0
    models = json.loads(result.output)
    assert models == list(KNOWN_MODELS)  # the whole list, as a machine-readable array


def test_llm_sends_prompt_and_prints_output(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["model"] = model
        seen["messages"] = messages
        seen["transcript_id"] = transcript_id
        return _payload("4")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "What is 2+2?", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["output"] == "4"
    assert data["model"] == "claude-haiku-4-5-20251001"
    assert seen["transcript_id"] is None
    assert seen["messages"][0]["content"] == "What is 2+2?"


def test_llm_json_emits_json_even_for_interactive_human(monkeypatch):
    _auth()
    # Even at an interactive terminal, --json emits machine output (it's the single,
    # explicit opt-in; we never auto-switch on pipe/agent anymore).
    monkeypatch.setattr("aai_cli.output._stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("4"))
    result = runner.invoke(app, ["llm", "hi", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["output"] == "4"


def test_llm_transcript_id_injected(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["transcript_id"] = transcript_id
        seen["content"] = messages[0]["content"]
        return _payload("summary")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "summarize", "--transcript-id", "t_7", "--json"])
    assert result.exit_code == 0
    assert seen["transcript_id"] == "t_7"
    assert "{{ transcript }}" in seen["content"]


def test_llm_reads_content_from_stdin(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["content"] = messages[0]["content"]
        seen["transcript_id"] = transcript_id
        return _payload("done")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "summarize", "--json"], input="meeting notes here")
    assert result.exit_code == 0
    # The piped text is injected into the prompt content; no transcript id is used.
    assert "summarize" in seen["content"]
    assert "meeting notes here" in seen["content"]
    assert seen["transcript_id"] is None


def test_llm_transcript_id_takes_priority_over_stdin(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["content"] = messages[0]["content"]
        seen["transcript_id"] = transcript_id
        return _payload("s")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app, ["llm", "summarize", "--transcript-id", "t_9", "--json"], input="ignored stdin"
    )
    assert result.exit_code == 0
    assert seen["transcript_id"] == "t_9"
    assert "ignored stdin" not in seen["content"]
    assert "{{ transcript }}" in seen["content"]


def test_llm_invalid_transcript_id_exits_2_without_network(monkeypatch):
    # Same cheap local validation as `transcripts get`: a malformed id never
    # reaches the gateway.
    _auth()
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the gateway")),
    )
    result = runner.invoke(app, ["llm", "summarize", "--transcript-id", "not-a-real-id!!"])
    assert result.exit_code == 2
    assert "doesn't look like a transcript id" in result.output


def test_llm_max_tokens_must_be_at_least_one(monkeypatch):
    # min=1 on --max-tokens: 0 and negatives are rejected client-side.
    _auth()
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the gateway")),
    )
    for bad in ("0", "-5"):
        result = runner.invoke(app, ["llm", "hi", "--max-tokens", bad])
        assert result.exit_code == 2
        assert "max-tokens" in result.output.lower()


def test_llm_transcript_id_warns_about_ignored_stdin(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("s"))
    result = runner.invoke(
        app, ["llm", "summarize", "--transcript-id", "t_9"], input="ignored stdin"
    )
    assert result.exit_code == 0
    assert "Ignoring piped stdin; --transcript-id takes priority." in result.output


def test_llm_transcript_id_stdin_warning_is_machine_readable_in_json_mode(monkeypatch):
    # In --json mode the warning must ship as its own {"warning": …} line (like the
    # env-mismatch warning), keeping stderr machine-readable.
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("s"))
    result = runner.invoke(app, ["llm", "summarize", "--transcript-id", "t_9", "--json"], input="x")
    assert result.exit_code == 0
    objs = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    warning = next(o for o in objs if "warning" in o)
    assert warning == {"warning": "Ignoring piped stdin; --transcript-id takes priority."}
    payload = next(o for o in objs if "output" in o)
    assert payload["output"] == "s"


def test_llm_transcript_id_stdin_warning_suppressed_by_quiet(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("s"))
    result = runner.invoke(
        app, ["--quiet", "llm", "summarize", "--transcript-id", "t_9"], input="x"
    )
    assert result.exit_code == 0
    assert "Ignoring piped stdin" not in result.output


def test_llm_transcript_id_no_warning_when_stdin_is_a_terminal(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.stdio.stdin_is_piped", lambda: False)
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("s"))
    result = runner.invoke(app, ["llm", "summarize", "--transcript-id", "t_9"])
    assert result.exit_code == 0
    assert "Ignoring piped stdin" not in result.output


def test_llm_list_models_rejects_output_flag(monkeypatch):
    # -o selects a field of a one-shot result; in --list-models mode it would be
    # silently ignored, so reject it (mirrors how --follow rejects -o).
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the gateway")),
    )
    result = runner.invoke(app, ["llm", "--list-models", "-o", "json"])
    assert result.exit_code == 2
    assert "one-shot" in result.output


def test_llm_missing_prompt_exits_2(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm"])
    assert result.exit_code == 2


def test_llm_unauthenticated_runs_login(monkeypatch):
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        raise AssertionError(f"LLM request should not run after auto-login: {api_key}")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "hello", "--json"])
    assert result.exit_code == 4
    assert config.get_api_key("default") == "sk_from_oauth"
    assert "Run the same command again" in result.output


def test_llm_follow_summarizes_each_turn(monkeypatch):
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
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

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
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


def test_llm_follow_ignores_blank_lines(monkeypatch):
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
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


def test_llm_output_text_prints_raw_answer(monkeypatch):
    _auth()
    monkeypatch.setattr(
        "aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("just the answer")
    )
    result = runner.invoke(app, ["llm", "hi", "-o", "text"])
    assert result.exit_code == 0
    # Raw text, not JSON — composes cleanly into the next pipe stage.
    assert result.output.strip() == "just the answer"
    assert "{" not in result.output


def test_llm_json_flag_emits_json(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("hello"))
    result = runner.invoke(app, ["llm", "hi", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["output"] == "hello"


def test_llm_output_json_field_forces_json_without_flag(monkeypatch):
    # `-o json` selects machine output even without the global --json flag, at an
    # interactive terminal (where json_mode is otherwise off). Pins the
    # `output_field == "json"` half of the json_mode disjunction.
    _auth()
    monkeypatch.setattr("aai_cli.output._stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload("hi42"))
    result = runner.invoke(app, ["llm", "hi", "-o", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["output"] == "hi42"


def test_llm_output_invalid_field_exits_2(monkeypatch):
    _auth()
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "hi", "-o", "bogus"])
    assert result.exit_code == 2


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
    monkeypatch.setattr("aai_cli.commands.llm.stdio.stdin_is_piped", lambda: False)
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "summarize", "--follow", "--json"])
    assert result.exit_code == 2
    assert "stdin" in result.output.lower()


def test_llm_follow_empty_stdin_exits_2(monkeypatch):
    # `assembly llm -f "…" </dev/null` must not exit 0 silently: an empty pipe means the
    # prompt never ran, which is a usage error, not a success.
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        calls.append(messages)
        return _payload("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(app, ["llm", "summarize", "--follow", "--json"], input="")
    assert result.exit_code == 2
    assert "--follow needs transcript text piped on stdin" in result.output
    assert calls == []  # no API call was made


def test_llm_follow_interrupt_before_first_turn_still_exits_0(monkeypatch):
    # Ctrl-C before any turn arrives is the normal "stop watching" signal, not the
    # empty-stdin usage error.
    _auth()

    class _InterruptIter:
        def __iter__(self):
            return self

        def __next__(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "aai_cli.commands.llm.stdio.iter_piped_stdin_lines", lambda: _InterruptIter()
    )
    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", lambda *a, **k: _payload())
    result = runner.invoke(app, ["llm", "summarize", "--follow", "--json"], input="")
    assert result.exit_code == 0
    assert "--follow needs transcript text piped on stdin" not in result.output


def test_llm_follow_stops_cleanly_on_interrupt(monkeypatch):
    _auth()
    calls = []

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        calls.append(messages[-1]["content"])
        if len(calls) == 2:
            raise KeyboardInterrupt  # user hits Ctrl-C mid-meeting
        return _payload("ok")

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app, ["llm", "summarize", "--follow", "--json"], input="alpha\nbeta\ngamma\n"
    )
    # Ctrl-C is a normal stop, not an error.
    assert result.exit_code == 0
    updates = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(updates) == 1
    assert updates[0]["turns"] == 1


def test_llm_passes_model_and_max_tokens(monkeypatch):
    _auth()
    seen = {}

    def fake_complete(api_key, *, model, messages, max_tokens, transcript_id=None):
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return _payload()

    monkeypatch.setattr("aai_cli.commands.llm.gateway.complete", fake_complete)
    result = runner.invoke(
        app, ["llm", "hi", "--model", "gemini-2.5-flash", "--max-tokens", "42", "--json"]
    )
    assert result.exit_code == 0
    assert seen["model"] == "gemini-2.5-flash"
    assert seen["max_tokens"] == 42


def test_no_prompt_suggests_list_models():
    result = runner.invoke(app, ["llm", "--json"])
    assert result.exit_code == 2
    assert "--list-models" in result.output
