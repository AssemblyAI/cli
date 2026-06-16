"""`assembly agent-cascade --show-code` tests.

Split from test_agent_cascade_command.py (which holds the run-path wiring) so the
print-only path's many invocations live in their own file. The cascade is
sandbox-only, so the happy paths run under `--sandbox`; the generated code_gen
rendering itself is covered by test_code_gen_agent_cascade.py.
"""

from __future__ import annotations

from typer.testing import CliRunner

from aai_cli.commands.agent_cascade import _exec
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


def test_show_code_prints_sandbox_script_without_running(monkeypatch):
    # Print-only: emits the cascade script, never wires deps or opens audio, no auth.
    def _boom(**kwargs):
        raise AssertionError("must not run a cascade")

    monkeypatch.setattr(_exec.engine, "run_cascade", _boom)
    monkeypatch.setattr(
        config, "resolve_api_key", lambda **_: (_ for _ in ()).throw(AssertionError("no auth"))
    )
    result = runner.invoke(
        app,
        ["--sandbox", "agent-cascade", "--voice", "jane", "--greeting", "Hi there", "--show-code"],
    )
    assert result.exit_code == 0
    # Targets the sandbox the key was minted for — all three legs.
    assert "streaming.sandbox000" in result.stdout
    assert "streaming-tts.sandbox000" in result.stdout
    assert "llm-gateway" in result.stdout
    assert "voice=jane" in result.stdout  # the chosen voice rides the TTS URL
    assert "Hi there" in result.stdout  # the greeting is injected
    compile(result.stdout, "<show-code>", "exec")  # the script is runnable Python


def test_show_code_defaults_off_at_the_argv_seam(monkeypatch):
    # Pin the Typer default (omitted -> False, so a bare run holds a conversation) and the
    # explicit form, captured at the argv->options seam so the run body never executes.
    captured = {}

    def fake_run(opts, state, *, json_mode):
        captured["opts"] = opts

    monkeypatch.setattr(_exec, "run_agent_cascade", fake_run)
    assert runner.invoke(app, ["agent-cascade"]).exit_code == 0
    assert captured["opts"].show_code is False
    assert runner.invoke(app, ["agent-cascade", "--show-code"]).exit_code == 0
    assert captured["opts"].show_code is True


def test_show_code_injects_speech_model(monkeypatch):
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kw: None)
    result = runner.invoke(
        app, ["--sandbox", "agent-cascade", "--speech-model", "u3-rt-pro", "--show-code"]
    )
    assert result.exit_code == 0
    assert "speech_model=u3-rt-pro" in result.stdout


def test_show_code_reflects_no_format_turns(monkeypatch):
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kw: None)
    formatted = runner.invoke(app, ["--sandbox", "agent-cascade", "--show-code"])
    bare = runner.invoke(app, ["--sandbox", "agent-cascade", "--no-format-turns", "--show-code"])
    # With formatting on the cue waits for the punctuated turn; off, a bare end-of-turn fires.
    assert "turn_is_formatted" in formatted.stdout
    assert "turn_is_formatted" not in bare.stdout
    assert "format_turns=false" in bare.stdout


def test_show_code_threads_model_and_max_tokens(monkeypatch):
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kw: None)
    result = runner.invoke(
        app,
        ["--sandbox", "agent-cascade", "--model", "claude-x", "--max-tokens", "321", "--show-code"],
    )
    assert result.exit_code == 0
    assert "claude-x" in result.stdout
    assert "MAX_TOKENS = 321" in result.stdout


def test_show_code_file_source_warns_on_stderr(monkeypatch):
    # The generated script is mic-driven; a passed source must warn, not be dropped silently.
    monkeypatch.setattr(
        _exec.engine, "run_cascade", lambda **kw: (_ for _ in ()).throw(AssertionError("no run"))
    )
    result = runner.invoke(app, ["--sandbox", "agent-cascade", "clip.wav", "--show-code"])
    assert result.exit_code == 0
    assert "uses the microphone" in result.stderr
    assert "uses the microphone" not in result.stdout  # stdout stays a clean script
    compile(result.stdout, "<show-code>", "exec")


def test_show_code_mic_emits_no_warning(monkeypatch):
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kw: None)
    result = runner.invoke(app, ["--sandbox", "agent-cascade", "--show-code"])
    assert result.exit_code == 0
    assert "uses the microphone" not in result.stderr  # mic script matches the run, nothing to warn


def test_show_code_in_production_is_rejected_with_sandbox_hint():
    # --show-code still honors the sandbox-only guard, so the generated URLs are valid.
    result = runner.invoke(app, ["agent-cascade", "--show-code"])
    assert result.exit_code == 2
    assert "only available in the sandbox" in result.output
