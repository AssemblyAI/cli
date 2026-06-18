"""Tests for the `assembly code` command wiring (commands/code/* + _exec).

The functions here are intentionally unannotated: they drive the command through
lightweight fakes (SimpleNamespace state, string agent sentinels) that the strict
type-checker would otherwise reject — the test suite skips untyped bodies by design.
"""

from __future__ import annotations

import builtins
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aai_cli.code_agent.ask_tool import AskBridge
from aai_cli.commands.code import _exec
from aai_cli.core.errors import CLIError
from aai_cli.main import app

runner = CliRunner()

_DEFAULTS = _exec.CodeOptions(prompt=None)


def _opts(**over) -> _exec.CodeOptions:
    return dataclasses.replace(_DEFAULTS, **over)


def test_command_parses_flags_into_options(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        _exec, "run_code", lambda opts, state, *, json_mode: captured.update(o=opts)
    )
    result = runner.invoke(
        app, ["code", "build a thing", "--auto", "--no-web", "--session", "s1", "--fresh"]
    )
    assert result.exit_code == 0
    opts = captured["o"]
    assert opts.prompt == "build a thing"
    assert opts.auto is True and opts.web is False
    assert opts.session == "s1" and opts.persist is False  # an explicit --session is honored


def test_command_defaults_to_a_fresh_unique_session_each_run(monkeypatch):
    # No --session: each invocation gets its own id (so a run never silently resumes the
    # previous conversation), and two runs differ.
    seen = []
    monkeypatch.setattr(
        _exec, "run_code", lambda opts, state, *, json_mode: seen.append(opts.session)
    )
    assert runner.invoke(app, ["code"]).exit_code == 0
    assert runner.invoke(app, ["code"]).exit_code == 0
    assert seen[0] != "default"  # not the old shared, auto-resumed thread
    assert seen[0] and seen[1] and seen[0] != seen[1]  # a distinct id per run


def test_run_code_dispatches_to_tui_with_voice_by_default_when_tty(monkeypatch):
    # The default (voice + tui in a TTY) now routes voice *into* the TUI: spoken turns are
    # entered into the prompt there, rather than running the separate voice REPL.
    calls = {}
    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr(_exec, "build_voice_session", lambda key: f"VOICE:{key}")
    monkeypatch.setattr(
        _exec, "_run_tui", lambda agent, opts, bridge, *, voice: calls.update(tui=(agent, voice))
    )
    monkeypatch.setattr(_exec, "_run_voice", lambda *a: calls.update(voice=True))
    monkeypatch.setattr(_exec, "_run_repl", lambda *a: calls.update(repl=True))
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    _exec.run_code(_opts(), state, json_mode=False)
    assert calls == {"tui": ("AGENT", "VOICE:k")}  # voice session handed to the TUI


def test_run_code_uses_voice_repl_when_tui_off(monkeypatch):
    # --no-tui keeps the plain voice REPL (speak, hear the reply) instead of the TUI.
    calls = {}
    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr(
        _exec, "_run_voice", lambda agent, opts, bridge, key: calls.update(voice=(agent, key))
    )
    monkeypatch.setattr(_exec, "_run_tui", lambda *a, **k: calls.update(tui=True))
    monkeypatch.setattr(_exec, "_run_repl", lambda *a: calls.update(repl=True))
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    _exec.run_code(_opts(tui=False), state, json_mode=False)
    assert calls == {"voice": ("AGENT", "k")}


def test_run_code_dispatches_to_tui_when_voice_off(monkeypatch):
    calls = {}
    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr(_exec, "_run_voice", lambda *a: calls.update(voice=True))
    monkeypatch.setattr(_exec, "_run_tui", lambda agent, opts, bridge: calls.update(tui=agent))
    monkeypatch.setattr(_exec, "_run_repl", lambda *a: calls.update(repl=True))
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    _exec.run_code(_opts(voice=False), state, json_mode=False)
    assert calls == {"tui": "AGENT"}


def test_run_code_repl_when_voice_and_tui_off(monkeypatch):
    calls = {}
    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr(_exec, "_run_voice", lambda *a: calls.update(voice=True))
    monkeypatch.setattr(_exec, "_run_tui", lambda *a: calls.update(tui=True))
    monkeypatch.setattr(_exec, "_run_repl", lambda agent, opts, bridge: calls.update(repl=agent))
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    _exec.run_code(_opts(voice=False, tui=False), state, json_mode=False)
    assert calls == {"repl": "AGENT"}


def test_run_code_falls_back_to_repl_off_tty(monkeypatch):
    calls = {}
    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr(_exec, "_run_tui", lambda *a: calls.update(tui=True))
    monkeypatch.setattr(_exec, "_run_repl", lambda agent, opts, bridge: calls.update(repl=agent))
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: False)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    _exec.run_code(_opts(), state, json_mode=False)
    assert calls == {"repl": "AGENT"}


def test_run_code_maps_keyboard_interrupt_to_exit_130(monkeypatch):
    import typer

    from aai_cli.core import errors

    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(_exec, "build_voice_session", lambda key: "VOICE")
    monkeypatch.setattr(_exec, "_run_tui", boom)  # the default front-end in a TTY
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    with pytest.raises(typer.Exit) as exc:
        _exec.run_code(_opts(), state, json_mode=False)
    assert exc.value.exit_code == errors.CANCELLED_EXIT_CODE


def test_assemble_tools_includes_cli_fetch_ask_and_optional_extras(monkeypatch):
    monkeypatch.setattr(_exec, "load_docs_tools", lambda: ["docs"])
    monkeypatch.setattr(_exec, "build_web_search_tool", lambda: "search")
    tools = _exec._assemble_tools("k", _opts(docs=True, web=True), AskBridge())
    assert [getattr(t, "name", t) for t in tools[:3]] == ["assembly", "fetch_url", "ask_user"]
    assert "docs" in tools and "search" in tools

    monkeypatch.setattr(_exec, "build_web_search_tool", lambda: None)
    tools = _exec._assemble_tools("k", _opts(docs=False, web=True), AskBridge())
    assert [t.name for t in tools] == ["assembly", "fetch_url", "ask_user"]


def test_assemble_middlewares_skills_and_memory(monkeypatch):
    monkeypatch.setattr(_exec, "build_skills_middleware", lambda: "SKILLS")
    monkeypatch.setattr(_exec, "build_memory_middleware", lambda: "MEM")
    assert _exec._assemble_middlewares(_opts(skills=True, memory=True)) == ["SKILLS", "MEM"]

    monkeypatch.setattr(_exec, "build_skills_middleware", lambda: None)
    assert _exec._assemble_middlewares(_opts(skills=True, memory=False)) == []


def test_build_agent_wires_model_tools_and_checkpointer(monkeypatch):
    seen = {}
    monkeypatch.setattr(_exec, "build_model", lambda key, *, model: f"model:{model}")
    monkeypatch.setattr(_exec, "_assemble_tools", lambda key, opts, bridge: ["t"])
    monkeypatch.setattr(_exec, "_assemble_middlewares", lambda opts: ["m"])
    monkeypatch.setattr(_exec, "build_checkpointer", lambda *, persist: f"ckpt:{persist}")
    monkeypatch.setattr(_exec, "build_agent", lambda **kw: seen.update(kw) or "AGENT")

    agent = _exec._build_agent("k", _opts(model="gpt-5", persist=False), AskBridge())
    assert agent == "AGENT"
    assert seen["model"] == "model:gpt-5"
    assert seen["tools"] == ["t"] and seen["middlewares"] == ["m"]
    assert seen["checkpointer"] == "ckpt:False"


def test_web_note_only_without_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert _exec._web_note(_opts(web=True)) is not None
    assert _exec._web_note(_opts(web=False)) is None
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-x")
    assert _exec._web_note(_opts(web=True)) is None


def test_confirm_reads_yes_no(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a: "y")
    assert _exec._confirm("write_file", {"file_path": "a"}) is True
    monkeypatch.setattr(builtins, "input", lambda *a: "n")
    assert _exec._confirm("write_file", {}) is False

    def eof(*a):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert _exec._confirm("write_file", {}) is False


def test_ask_repl_and_read_line(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a: "the answer")
    assert _exec._ask_repl("q?") == "the answer"
    assert _exec._read_line() == "the answer"

    def eof(*a):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert _exec._ask_repl("q?") == ""
    assert _exec._read_line() is None


def test_run_repl_prints_banner_and_runs(monkeypatch):
    class Dummy:
        def invoke(self, *a, **k):
            return {"messages": []}

    def eof(*a):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)  # immediate EOF ends the loop
    bridge = AskBridge()
    _exec._run_repl(Dummy(), _opts(session="s2"), bridge)
    assert bridge.handler is _exec._ask_repl  # the REPL wired the ask handler


def test_run_tui_invokes_app_run(monkeypatch):
    seen = {}

    class FakeApp:
        def __init__(self, **kw):
            seen.update(kw)

        def run(self, **kw):
            seen["run_kw"] = kw

    monkeypatch.setattr("aai_cli.code_agent.tui.CodeAgentApp", FakeApp)
    _exec._run_tui("AGENT", _opts(prompt="hi", session="s", root_dir=Path()), AskBridge())
    assert seen["agent"] == "AGENT" and seen["thread_id"] == "s"
    assert seen["run_kw"] == {"mouse": False}


def test_voice_sink_renders_all_events_and_speaks_only_assistant_text():
    from aai_cli.code_agent.events import AssistantText, ToolCall

    rendered, spoken = [], []
    voice = SimpleNamespace(speak=spoken.append)

    def renderer(event):
        rendered.append(event)

    sink = _exec._voice_sink(renderer, voice)
    sink(AssistantText("here you go"))
    sink(ToolCall(name="write_file", args={}))

    assert [type(e).__name__ for e in rendered] == ["AssistantText", "ToolCall"]
    assert spoken == ["here you go"]  # only the assistant's prose is read back


def test_announce_voice_message_depends_on_readback():
    notes = []
    renderer = SimpleNamespace(notice=notes.append)

    _exec._announce_voice(renderer, SimpleNamespace(readback=True))
    assert "read back" in notes[-1]

    _exec._announce_voice(renderer, SimpleNamespace(readback=False))
    assert "sandbox" in notes[-1] and "text" in notes[-1]


def test_voice_read_line_returns_spoken_line():
    notes = []
    renderer = SimpleNamespace(notice=notes.append)
    voice = SimpleNamespace(listen=lambda: "add a flag")

    read_line = _exec._voice_read_line(voice, renderer)
    assert read_line() == "add a flag"
    assert any("Heard: add a flag" in n for n in notes)


def test_voice_read_line_passes_through_none_for_eof():
    renderer = SimpleNamespace(notice=lambda *a: None)
    voice = SimpleNamespace(listen=lambda: None)
    assert _exec._voice_read_line(voice, renderer)() is None


def test_voice_read_line_falls_back_to_typed_input_when_no_mic(monkeypatch):
    notes = []
    renderer = SimpleNamespace(notice=notes.append)
    calls = {"listen": 0}

    def flaky_mic():
        calls["listen"] += 1
        if calls["listen"] == 1:
            raise CLIError("no device", error_type="mic_missing", exit_code=2)
        return "SPOKEN AGAIN"  # would leak through only if the mic were retried

    voice = SimpleNamespace(listen=flaky_mic)
    monkeypatch.setattr(builtins, "input", lambda *a: "typed instead")

    read_line = _exec._voice_read_line(voice, renderer)
    assert read_line() == "typed instead"  # first call: mic fails -> typed input
    assert read_line() == "typed instead"  # stays typed; the mic is not retried
    assert calls["listen"] == 1  # the latch flipped, so listen() was attempted only once
    assert any("switching to typed input" in n.lower() for n in notes)


def test_voice_read_line_reraises_non_audio_errors():
    renderer = SimpleNamespace(notice=lambda *a: None)

    def boom():
        raise CLIError("gateway down", error_type="api_error", exit_code=1)

    voice = SimpleNamespace(listen=boom)
    with pytest.raises(CLIError):
        _exec._voice_read_line(voice, renderer)()


def test_run_voice_wires_ask_handler_and_drives_repl(monkeypatch):
    class Dummy:
        def invoke(self, *a, **k):
            return {"messages": []}

    voice = SimpleNamespace(readback=False, listen=lambda: None, speak=lambda *a: None)
    monkeypatch.setattr(_exec, "build_voice_session", lambda key: voice)
    bridge = AskBridge()
    _exec._run_voice(Dummy(), _opts(session="s3"), bridge, "k")
    assert bridge.handler is _exec._ask_repl
