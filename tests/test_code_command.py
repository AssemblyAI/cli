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
    assert opts.session == "s1" and opts.persist is False


def test_run_code_dispatches_to_tui_when_tty(monkeypatch):
    calls = {}
    monkeypatch.setattr(_exec, "_build_agent", lambda key, opts, bridge: "AGENT")
    monkeypatch.setattr(_exec, "_run_tui", lambda agent, opts, bridge: calls.update(tui=agent))
    monkeypatch.setattr(_exec, "_run_repl", lambda *a: calls.update(repl=True))
    monkeypatch.setattr("aai_cli.core.stdio.stdout_is_tty", lambda: True)
    monkeypatch.setattr("aai_cli.core.stdio.stdin_is_tty", lambda: True)
    state = SimpleNamespace(resolve_api_key=lambda: "k")

    _exec.run_code(_opts(), state, json_mode=False)
    assert calls == {"tui": "AGENT"}


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

    def boom(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(_exec, "_run_tui", boom)
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
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert _exec._web_note(_opts(web=True)) is not None
    assert _exec._web_note(_opts(web=False)) is None
    monkeypatch.setenv("TAVILY_API_KEY", "tvly")
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
