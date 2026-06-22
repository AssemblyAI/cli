"""Tests for run_agent_cascade's TUI selection + wiring.

Split out of test_live_tui.py to keep both files under the 500-line file-length gate.
"""

from __future__ import annotations

import types

import pytest
import typer

from aai_cli.agent_cascade import engine
from aai_cli.app.context import AppState
from aai_cli.commands.agent_cascade import _exec
from aai_cli.commands.agent_cascade._exec import run_agent_cascade
from aai_cli.core import config, stdio
from aai_cli.core.errors import CLIError
from tests.test_agent_cascade_command import _opts

# --- run_agent_cascade -> TUI selection + wiring -----------------------------


def test_should_use_tui_only_for_interactive_human_mic_sessions(monkeypatch) -> None:
    # The TUI is the default for a live mic session in human mode on a TTY. Each of the four
    # disqualifiers (file input, --json, -o text, no TTY) falls back to the line renderer.
    monkeypatch.setattr(stdio, "stdout_is_tty", lambda: True)
    monkeypatch.setattr(stdio, "stdin_is_tty", lambda: True)
    assert _exec._should_use_tui(from_file=False, json_mode=False, text_mode=False) is True
    assert _exec._should_use_tui(from_file=True, json_mode=False, text_mode=False) is False
    assert _exec._should_use_tui(from_file=False, json_mode=True, text_mode=False) is False
    assert _exec._should_use_tui(from_file=False, json_mode=False, text_mode=True) is False
    monkeypatch.setattr(stdio, "stdout_is_tty", lambda: False)
    assert _exec._should_use_tui(from_file=False, json_mode=False, text_mode=False) is False


def test_web_search_note_tracks_the_firecrawl_key(monkeypatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert "FIRECRAWL_API_KEY" in (_exec._web_search_note() or "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-x")
    assert _exec._web_search_note() is None


def _wire_tui(monkeypatch):
    """Stub auth/audio/deps so run_agent_cascade reaches the TUI launch on an interactive mic run."""
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(stdio, "stdout_is_tty", lambda: True)
    monkeypatch.setattr(stdio, "stdin_is_tty", lambda: True)
    fake_duplex = types.SimpleNamespace(mic=object(), player=object(), close=lambda: None)
    monkeypatch.setattr(_exec, "DuplexAudio", lambda **kwargs: fake_duplex)
    monkeypatch.setattr(engine.CascadeDeps, "real", lambda *a, **k: "deps")
    return fake_duplex


def test_interactive_human_run_launches_the_tui(monkeypatch) -> None:
    # A mic session in human mode on a TTY runs the Textual app, not the line renderer.
    fake_duplex = _wire_tui(monkeypatch)
    captured: dict[str, object] = {}

    class FakeApp:
        error = None  # no fatal leg failure -> the launcher re-raises nothing

        def __init__(self, *, run_conversation, on_stop, web_note):
            captured["run_conversation"] = run_conversation
            captured["on_stop"] = on_stop

        def run(self, **kwargs):
            captured["ran"] = kwargs

    monkeypatch.setattr("aai_cli.agent_cascade.tui.LiveAgentApp", FakeApp)
    # AgentRenderer must NOT be built on the TUI path — fail loudly if the line path is taken.
    monkeypatch.setattr(
        _exec, "AgentRenderer", lambda **kw: pytest.fail("line renderer used in TUI mode")
    )
    run_agent_cascade(_opts(), AppState(), json_mode=False)
    assert callable(captured["run_conversation"])  # the TUI was launched with a cascade closure
    assert captured["on_stop"] is fake_duplex.close  # quit closes the audio
    assert captured["ran"] == {"mouse": False}  # mouse off so transcript text stays selectable


def test_tui_setup_keyboard_interrupt_exits_clean(monkeypatch) -> None:
    # Ctrl-C during TUI setup (mic open / graph build / --mcp-config load) lands before
    # Textual captures the keyboard; it must exit 130, not surface a raw traceback.
    _wire_tui(monkeypatch)

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(_exec, "_run_live_tui", boom)
    with pytest.raises(typer.Exit) as exc:
        run_agent_cascade(_opts(), AppState(), json_mode=False)
    assert exc.value.exit_code == 130


def test_tui_run_conversation_drives_the_cascade(monkeypatch) -> None:
    # The closure handed to the app runs the cascade with the duplex player and the wired
    # deps, and the cascade's on_session wires the session's reply-interrupt onto the app.
    fake_duplex = _wire_tui(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run_cascade(**kw):
        captured.update(kw)
        # run_cascade hands the freshly built session to on_session before the conversation.
        kw["on_session"](types.SimpleNamespace(interrupt_reply="session-interrupt"))

    monkeypatch.setattr(engine, "run_cascade", fake_run_cascade)

    class FakeApp:
        error = None  # the conversation completes cleanly here

        def __init__(self, *, run_conversation, on_stop, web_note):
            self._rc = run_conversation

        def run(self, **kwargs):
            self._rc("renderer-sentinel")  # the app would call this on its worker thread

        def set_interrupt(self, interrupt):
            captured["interrupt"] = interrupt

    monkeypatch.setattr("aai_cli.agent_cascade.tui.LiveAgentApp", FakeApp)
    run_agent_cascade(_opts(), AppState(), json_mode=False)
    assert captured["player"] is fake_duplex.player
    assert captured["deps"] == "deps"
    assert captured["renderer"] == "renderer-sentinel"
    # The session's interrupt_reply was wired onto the app (so Escape/Ctrl-C can use it).
    assert captured["interrupt"] == "session-interrupt"


def test_tui_reraises_a_fatal_leg_error_for_the_exit_code(monkeypatch) -> None:
    # A fatal leg failure is caught on the TUI worker thread and parked on app.error; the
    # launcher must re-raise it after the app tears down so the command exits with the
    # error's code (api_error -> exit 1) instead of a silent success.
    _wire_tui(monkeypatch)
    boom = CLIError("streaming STT closed", error_type="api_error", exit_code=1)

    class FakeApp:
        error = boom  # the worker thread recorded a fatal cascade error

        def __init__(self, *, run_conversation, on_stop, web_note):
            pass

        def run(self, **kwargs):
            pass

    monkeypatch.setattr("aai_cli.agent_cascade.tui.LiveAgentApp", FakeApp)
    with pytest.raises(CLIError) as exc:
        run_agent_cascade(_opts(), AppState(), json_mode=False)
    assert exc.value is boom
