"""Tests for the voice-only `assembly live` Textual TUI (``LiveAgentApp``).

Drives the real Textual app headless. Most tests call the transcript/phase methods directly
(they always run on the UI thread), mirroring the code-TUI suite; two drive the worker leg with
a scripted ``run_conversation`` through the real ``_TuiRenderer`` to cover the off-thread hop,
the error path, and teardown — all without a mic, speaker, or socket.
"""

from __future__ import annotations

import asyncio
import threading
import types

import pytest
import typer
from textual.widgets import Static

from aai_cli.agent_cascade import engine
from aai_cli.agent_cascade.tui import LiveAgentApp, _TuiRenderer
from aai_cli.app.context import AppState
from aai_cli.code_agent.messages import AssistantMessage, ErrorMessage, Note, UserMessage
from aai_cli.commands.agent_cascade import _exec
from aai_cli.commands.agent_cascade._exec import run_agent_cascade
from aai_cli.core import config, stdio
from aai_cli.core.errors import CLIError
from tests.test_agent_cascade_command import _opts


def _run(coro) -> None:
    asyncio.run(coro)


def _wait_until(pilot, predicate):
    """Pump the event loop until ``predicate`` holds (lets a worker thread land)."""

    async def loop() -> bool:
        for _ in range(200):
            await pilot.pause(0.01)
            if predicate():
                return True
        return False

    return loop()


def _app(run_conversation=None, on_stop=None, on_toggle_listen=None, web_note=None):
    """A LiveAgentApp whose worker stays alive for the test, releasing on teardown.

    The real ``run_conversation`` blocks on the live mic; the default here blocks on an event
    so the app doesn't auto-exit (an instant return makes the worker close the app). Teardown
    always sets that event — and still runs any test-supplied ``on_stop`` — so no worker leaks.
    """
    release = threading.Event()

    def stop() -> None:
        release.set()
        if on_stop is not None:
            on_stop()

    def block(renderer) -> None:
        release.wait(30)  # block like a live mic; teardown releases it well before this

    return LiveAgentApp(
        run_conversation=run_conversation or block,
        on_stop=stop,
        on_toggle_listen=on_toggle_listen or (lambda: True),
        web_note=web_note,
    )


def _voicebar(app) -> str:
    return str(app.query_one("#voicebar", Static).render())


def test_splash_and_status_render() -> None:
    # The session opens on the ASSEMBLY wordmark + ready line, and the footer shows the
    # interrupt/quit controls — there is no text prompt mounted (input is voice-only).
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            splash = str(app.query_one("#log").children[0].render())
            assert "█" in splash and "Listening… start talking" in splash  # the wordmark splash
            assert "Listening" in _voicebar(app)  # opens in the listening phase
            status = str(app.query_one("#status", Static).render())
            assert "interrupt" in status and "Ctrl-Q to quit" in status
            assert len(app.query("#prompt")) == 0  # no text input — voice only
            assert app.ENABLE_COMMAND_PALETTE is False  # the voice UI hides the command palette

    _run(go())


def test_user_partial_grows_then_finalizes_into_thinking() -> None:
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_user_partial("what is")
            app.show_user_partial("what is the weather")
            # One growing user line, not two — the partial updates in place.
            assert len(app.query(UserMessage)) == 1
            assert "Listening" in _voicebar(app)
            app.show_user_final("what is the weather")
            assert "» what is the weather" in str(app.query_one(UserMessage).render())
            assert "Thinking" in _voicebar(app)  # a finalized turn -> the LLM is thinking

    _run(go())


def test_user_final_without_a_prior_partial_still_shows_the_turn() -> None:
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_user_final("hello")  # no partial first (formatted turn arrives whole)
            assert "» hello" in str(app.query_one(UserMessage).render())
            assert "Thinking" in _voicebar(app)

    _run(go())


def test_reply_streams_sentences_and_finalizes_back_to_listening() -> None:
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.begin_reply()
            assert "Speaking" in _voicebar(app)
            app.show_agent_sentence("Hello.")
            app.show_agent_sentence("How can I help?")
            reply = app.query_one(AssistantMessage)
            assert reply.text == "Hello. How can I help? "
            app.end_reply(interrupted=False)
            assert "Listening" in _voicebar(app)  # reply done -> back to listening
            assert len(app.query(Note)) == 0  # not interrupted -> no interrupted aside

    _run(go())


def test_show_tool_call_mounts_an_inline_affordance() -> None:
    # A tool call mid-turn drops a dim "Searching the web…" note, so the thinking pause reads
    # as progress rather than a hang (the live tool affordance).
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_tool_call("Searching the web")
            notes = [str(n.render()) for n in app.query(Note)]
            assert any("Searching the web" in n for n in notes)

    _run(go())


def test_agent_sentence_without_begin_reply_mounts_a_reply() -> None:
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_agent_sentence("Standalone.")  # defensive: no begin_reply first
            assert app.query_one(AssistantMessage).text == "Standalone. "

    _run(go())


def test_interrupted_reply_notes_the_barge_in() -> None:
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.begin_reply()
            app.show_agent_sentence("As I was saying")
            app.end_reply(interrupted=True)  # the user barged in
            assert any("interrupted" in str(n.render()) for n in app.query(Note))
            assert "Listening" in _voicebar(app)

    _run(go())


def test_end_reply_without_an_active_reply_is_a_safe_noop() -> None:
    # A reply_done with no open reply widget (e.g. a turn that produced no spoken sentence) must
    # not touch the absent widget — it just returns to listening.
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.end_reply(interrupted=False)  # no begin_reply first
            assert len(app.query(AssistantMessage)) == 0  # nothing mounted
            assert "Listening" in _voicebar(app)

    _run(go())


def test_voice_bar_tick_advances_then_survives_teardown() -> None:
    # Each tick advances the meter; once #voicebar is gone (a teardown tick) it must no-op, not raise.
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            before = _voicebar(app)
            app._tick_voice()
            assert _voicebar(app) != before  # the meter advanced a frame
            await app.query_one("#voicebar", Static).remove()
            app._tick_voice()  # bar gone -> no-op, no NoMatches
            assert len(app.query("#voicebar")) == 0

    _run(go())


def test_web_note_is_surfaced_as_a_notification() -> None:
    async def go() -> None:
        app = _app(web_note="Web search is off — set FIRECRAWL_API_KEY")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert any("FIRECRAWL_API_KEY" in n.message for n in app._notifications)

    _run(go())


def test_escape_interrupts_a_playing_reply_via_the_session_hook() -> None:
    # Escape fires the session's reply-interrupt (set once the cascade has a session) and
    # never quits — the worker unwinds and the renderer returns the bar to listening.
    async def go() -> None:
        fired: list[bool] = []

        def hook() -> bool:
            fired.append(True)
            return True

        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.set_interrupt(hook)
            app.action_interrupt()
            assert fired == [True]

    _run(go())


def test_space_toggles_listening_and_paints_paused() -> None:
    # Space starts/stops listening: it drives the duplex mic mute (the returned state) and
    # repaints the voice bar to "Paused" while muted, then back to "Listening" on resume.
    async def go() -> None:
        state = {"on": True}

        def toggle() -> bool:
            state["on"] = not state["on"]
            return state["on"]

        app = _app(on_toggle_listen=toggle)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert "Listening" in _voicebar(app)  # opens listening
            await pilot.press("space")  # the Space binding -> action_toggle_listen -> stop
            assert state["on"] is False and app._listening is False  # mic muted
            assert "Paused" in _voicebar(app)  # muted shows paused, not listening
            # Muting only gates the user's input: a reply still in flight keeps "Speaking".
            app._set_phase("speaking")
            assert "Speaking" in _voicebar(app) and "Paused" not in _voicebar(app)
            app._set_phase("listening")
            await pilot.press("space")  # resume listening
            assert state["on"] is True and app._listening is True
            assert "Listening" in _voicebar(app)

    _run(go())


def test_ctrl_c_interrupts_a_playing_reply_without_quitting(monkeypatch) -> None:
    # While a reply is playing (the hook returns True), Ctrl-C interrupts it and stays — it
    # must NOT quit, so a long answer can be cut off without ending the session.
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            exited: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.set_interrupt(lambda: True)  # a reply is playing
            app.action_interrupt_or_quit()
            assert exited == []  # interrupted, not quit

    _run(go())


def test_ctrl_c_quits_when_nothing_is_playing(monkeypatch) -> None:
    # With no reply playing (the hook returns False, or none is wired yet), Ctrl-C quits.
    async def go() -> None:
        stops: list[bool] = []
        app = _app(on_stop=lambda: stops.append(True))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            exited: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.set_interrupt(lambda: False)  # nothing playing
            app.action_interrupt_or_quit()
            assert stops == [True] and exited == [True]

    _run(go())


def test_interrupt_before_a_session_is_wired_is_a_safe_noop(monkeypatch) -> None:
    # A keypress before the cascade has built its session (no interrupt hook yet): Escape is a
    # no-op and Ctrl-C falls through to quit, so an early press can never wedge the UI.
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            exited: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.action_interrupt()  # no hook wired -> nothing happens, no crash
            assert exited == []
            app.action_interrupt_or_quit()  # nothing to interrupt -> quits
            assert exited == [True]

    _run(go())


def test_action_stop_tears_down_audio_and_exits(monkeypatch) -> None:
    async def go() -> None:
        stops: list[bool] = []
        app = _app(on_stop=lambda: stops.append(True))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            exited: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.action_stop()
            assert stops == [True]  # the audio was closed (unblocks the cascade worker)
            assert exited == [True]
            app.action_stop()  # idempotent: a second stop never re-closes the audio
            assert stops == [True]

    _run(go())


def test_worker_drives_the_renderer_and_unmount_closes_audio() -> None:
    # The blocking run_conversation runs on a worker thread and reaches the UI through the real
    # _TuiRenderer; tearing the app down fires on_stop, which (in production) ends the mic and
    # lets the worker return.
    async def go() -> None:
        done = threading.Event()

        def run_conversation(renderer) -> None:
            # A full spoken turn, exercising every _TuiRenderer leg (each hops to the UI thread).
            renderer.connected()
            renderer.user_partial("turn it")
            renderer.user_final("turn it up")
            renderer.tool_call("Searching the web")
            renderer.reply_started()
            renderer.agent_transcript("Done.", interrupted=False)
            renderer.reply_done(interrupted=False)
            done.wait(30)  # block until teardown's on_stop fires (timeout is just a leak guard)

        app = _app(run_conversation=run_conversation, on_stop=done.set)
        async with app.run_test(size=(100, 30)) as pilot:
            # Wait for the reply *text* to land, not just the widget to mount: agent_transcript
            # sets the text via a separate call_from_thread hop, so a widget-only wait races it
            # (empty text on a slow runner — a Windows CI flake).
            assert await _wait_until(
                pilot,
                lambda: (
                    bool(app.query(AssistantMessage))
                    and app.query_one(AssistantMessage).text == "Done. "
                ),
            )
            assert "» turn it up" in str(app.query_one(UserMessage).render())
            assert app.query_one(AssistantMessage).text == "Done. "
            # The tool_call leg hopped to the UI thread and surfaced the affordance note.
            assert any("Searching the web" in str(n.render()) for n in app.query(Note))
        assert done.is_set()  # leaving the run_test context unmounted -> on_stop released it

    _run(go())


def test_worker_surfaces_a_leg_error_in_the_transcript() -> None:
    async def go() -> None:
        boom_error = CLIError("gateway down", error_type="api_error", exit_code=1)

        def boom(renderer) -> None:
            raise boom_error

        app = _app(run_conversation=boom)
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _wait_until(pilot, lambda: bool(app.query(ErrorMessage)))
            assert "gateway down" in str(app.query_one(ErrorMessage).render())
            # The error is also kept on the app so the launcher can re-raise it for the
            # right exit code, not just shown inline (where a torn-down TUI would lose it).
            assert app.error is boom_error

    _run(go())


def test_tui_renderer_drops_calls_after_the_app_stops() -> None:
    # A renderer call that lands after teardown must be swallowed (the turn is moot), not raised
    # as an unhandled worker-thread error. This app was never started, so is_running is False.
    app = _app()
    assert app.is_running is False
    renderer = _TuiRenderer(app)
    renderer.user_final("ignored")  # returns without raising
    renderer.reply_done(interrupted=False)


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
    fake_duplex = types.SimpleNamespace(
        mic=object(), player=object(), close=lambda: None, toggle_listening=lambda: True
    )
    monkeypatch.setattr(_exec, "DuplexAudio", lambda **kwargs: fake_duplex)
    monkeypatch.setattr(engine.CascadeDeps, "real", lambda *a, **k: "deps")
    return fake_duplex


def test_interactive_human_run_launches_the_tui(monkeypatch) -> None:
    # A mic session in human mode on a TTY runs the Textual app, not the line renderer.
    fake_duplex = _wire_tui(monkeypatch)
    captured: dict[str, object] = {}

    class FakeApp:
        error = None  # no fatal leg failure -> the launcher re-raises nothing

        def __init__(self, *, run_conversation, on_stop, on_toggle_listen, web_note):
            captured["run_conversation"] = run_conversation
            captured["on_stop"] = on_stop
            captured["on_toggle_listen"] = on_toggle_listen

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
    # Space toggles listening through the duplex's in-place mic mute (no reconnect).
    assert captured["on_toggle_listen"] is fake_duplex.toggle_listening
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

        def __init__(self, *, run_conversation, on_stop, on_toggle_listen, web_note):
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

        def __init__(self, *, run_conversation, on_stop, on_toggle_listen, web_note):
            pass

        def run(self, **kwargs):
            pass

    monkeypatch.setattr("aai_cli.agent_cascade.tui.LiveAgentApp", FakeApp)
    with pytest.raises(CLIError) as exc:
        run_agent_cascade(_opts(), AppState(), json_mode=False)
    assert exc.value is boom
