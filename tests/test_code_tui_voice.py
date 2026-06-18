"""Tests for the `assembly code` TUI's voice integration.

Drives the real Textual app (headless) with a fake agent and a scripted voice double, so
the listen→enter-into-the-prompt→submit cycle and the spoken-summary readback are exercised
without a microphone, speaker, or socket. Split from test_code_tui.py to keep each file under
the 500-line gate.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from textual.widgets import Input, Static

from aai_cli.code_agent.tui import CodeAgentApp
from aai_cli.core.errors import CLIError


class FakeAgent:
    """Replays scripted invoke() results so a turn can complete without a model."""

    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results
        self.calls = 0

    def invoke(self, *args, **kwargs):
        result = self._results[self.calls]
        self.calls += 1
        return result


class FakeVoice:
    """A scripted voice I/O double: listen() replays transcripts, speak() records text."""

    def __init__(self, transcripts: list[str] | None = None, *, error: CLIError | None = None):
        self._transcripts = list(transcripts or [])
        self._error = error
        self.spoken: list[str] = []
        self.listens = 0

    def listen(self) -> str | None:
        self.listens += 1
        if self._error is not None:
            raise self._error
        return self._transcripts.pop(0) if self._transcripts else None

    def speak(self, text: str) -> None:
        self.spoken.append(text)


def _run(coro) -> None:
    asyncio.run(coro)


def _wait_until(pilot, predicate):
    """Pump the event loop until ``predicate`` holds (lets a voice worker thread land)."""

    async def loop() -> bool:
        for _ in range(200):
            await pilot.pause(0.01)
            if predicate():
                return True
        return False

    return loop()


def test_voice_active_requires_a_session_and_an_available_mic() -> None:
    async def go() -> None:
        no_voice = CodeAgentApp(agent=FakeAgent([]))
        async with no_voice.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert no_voice._voice_active() is False  # no voice session at all

        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app._voice_active() is True
            app._voice_typed = True
            assert app._voice_active() is False  # mic ruled out -> inactive

    _run(go())


def test_enter_and_submit_fills_prompt_then_clears_and_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            submitted: list[str] = []
            monkeypatch.setattr(app, "_submit", submitted.append)
            app._enter_and_submit("add a verbose flag")
            assert submitted == ["add a verbose flag"]  # the spoken turn was submitted
            assert app.query_one("#prompt", Input).value == ""  # prompt cleared afterwards

    _run(go())


def test_voice_on_mount_listens_and_submits_the_spoken_turn() -> None:
    async def go() -> None:
        agent = FakeAgent([{"messages": [HumanMessage("do x"), AIMessage("done")]}])
        voice = FakeVoice(transcripts=["do x"])
        app = CodeAgentApp(agent=agent, voice=voice)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # on_mount (no initial prompt) starts listening; the captured turn drives the agent.
            assert await _wait_until(pilot, lambda: agent.calls >= 1)
            assert voice.listens >= 1

    _run(go())


def test_capture_voice_turn_is_a_noop_once_typed() -> None:
    async def go() -> None:
        voice = FakeVoice(transcripts=["ignored"])
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        app._voice_typed = True  # set before mount so on_mount never auto-listens
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._capture_voice_turn()  # typed -> returns before listen (safe on the UI thread)
            assert voice.listens == 0

    _run(go())


def test_voice_degrades_to_typed_on_capture_error() -> None:
    async def go() -> None:
        voice = FakeVoice(error=CLIError("no mic", error_type="mic_missing", exit_code=2))
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert await _wait_until(pilot, lambda: app._voice_typed)
            assert app._voice_typed is True  # a capture failure drops voice for the session

    _run(go())


def test_voice_followup_reads_a_summary_of_the_last_reply() -> None:
    async def go() -> None:
        voice = FakeVoice()
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._voice_typed = True  # isolate the readback: the post-speak listen is a no-op
            app._last_reply = "Here is the plan.\n```py\ncode\n```"
            app._voice_followup()
            assert await _wait_until(pilot, lambda: bool(voice.spoken))
            assert voice.spoken == ["Here is the plan."]  # summary only — the code is stripped

    _run(go())


def test_voice_followup_is_a_noop_without_voice() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))  # no voice session
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._voice_followup()  # returns immediately without speaking or listening
            assert app._voice is None

    _run(go())


def test_toggle_voice_pauses_and_resumes_capture() -> None:
    # Ctrl-V flips voice off (no capture, no readback) and back on; the state badge tracks it.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # Assert via the methods, not the `_voice_paused` attribute: mypy narrows the
            # attribute and can't see action_toggle_voice() flip it back, flagging the second
            # check unreachable. The method calls reflect the same state without that trap.
            assert app._voice_active()
            assert app._voice_state() == "on"
            app.action_toggle_voice()  # pause
            assert not app._voice_active()
            assert app._voice_state() == "off"
            app.action_toggle_voice()  # resume
            assert app._voice_active()
            assert app._voice_state() == "on"

    _run(go())


def test_paused_voice_skips_followup_readback() -> None:
    # While paused, the post-turn followup neither speaks a summary nor listens.
    async def go() -> None:
        voice = FakeVoice(transcripts=["ignored"])
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        app._voice_paused = True  # set before mount so on_mount never auto-listens
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._last_reply = "a reply"
            app._voice_followup()
            await pilot.pause()
            assert voice.spoken == []  # paused: no readback
            assert voice.listens == 0  # paused: no capture

    _run(go())


def test_voice_mode_swaps_text_input_for_listening_affordance() -> None:
    # While voice capture is on, the text prompt is hidden and a "listening" bar shows;
    # toggling voice off (Ctrl-V) brings the text box back. (Re-query each check so mypy
    # doesn't narrow a stored display bool across the toggles.)
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        app._voice_paused = True  # start paused so on_mount doesn't race a capture thread
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.query_one("#promptbar").display is True  # paused -> text box visible
            assert app.query_one("#voicebar").display is False
            app.action_toggle_voice()  # voice on
            await pilot.pause()
            assert app.query_one("#promptbar").display is False  # text box hidden
            assert app.query_one("#voicebar").display is True  # listening affordance shown
            app.action_toggle_voice()  # voice off
            await pilot.pause()
            assert app.query_one("#promptbar").display is True  # text box back
            assert app.query_one("#voicebar").display is False

    _run(go())


def test_voice_capture_failure_restores_the_text_input() -> None:
    # When the mic is ruled out mid-session, the listening bar is replaced by the text box.
    async def go() -> None:
        voice = FakeVoice(error=CLIError("no mic", error_type="mic_missing", exit_code=2))
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert await _wait_until(pilot, lambda: app._voice_typed)
            await pilot.pause()
            assert app.query_one("#promptbar").display is True  # text box restored on failure
            assert app.query_one("#voicebar").display is False

    _run(go())


def test_voice_bar_distinguishes_phases() -> None:
    # The bar shows a distinct label per phase; only the listening phase carries the type hint.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        app._voice_paused = True  # quiet the auto-listen; drive phases directly
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._set_voice_phase("listening")
            bar = str(app.query_one("#voicebar", Static).render())
            assert "Listening" in bar and "Ctrl-V to type" in bar
            app._set_voice_phase("thinking")
            bar = str(app.query_one("#voicebar", Static).render())
            assert "Thinking" in bar and "Ctrl-V to type" not in bar  # hint is listening-only
            app._set_voice_phase("speaking")
            assert "Speaking" in str(app.query_one("#voicebar", Static).render())

    _run(go())


def test_spinner_suppressed_in_voice_mode() -> None:
    # In voice mode the bar carries the "thinking" state, so the separate spinner stays hidden;
    # pausing voice brings the spinner back.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._start_spinner()
            assert app.query_one("#spinner", Static).display is False  # voice active -> no spinner
            app._voice_paused = True
            app._start_spinner()
            assert app.query_one("#spinner", Static).display is True  # paused -> spinner shows

    _run(go())


def test_voice_bar_animation_timer_runs_and_advances() -> None:
    # The meter animation timer runs only while the bar is shown, and a tick changes the frame.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        app._voice_paused = True
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # Read into fresh locals each time: asserting `is None`/`is not None` on the same
            # attribute across the opaque toggle would make mypy flag the later check unreachable.
            paused_timer = app._voice_timer
            assert paused_timer is None  # paused -> no animation
            app.action_toggle_voice()  # voice on -> bar shown, timer running
            await pilot.pause()
            running_timer = app._voice_timer
            assert running_timer is not None
            before = str(app.query_one("#voicebar", Static).render())
            app._tick_voice()
            assert str(app.query_one("#voicebar", Static).render()) != before  # meter advanced
            app.action_toggle_voice()  # voice off -> timer stopped
            await pilot.pause()
            stopped_timer = app._voice_timer
            assert stopped_timer is None

    _run(go())


def test_submit_sets_thinking_phase() -> None:
    async def go() -> None:
        agent = FakeAgent([{"messages": [HumanMessage("go"), AIMessage("done")]}])
        app = CodeAgentApp(agent=agent, voice=FakeVoice())
        app._voice_paused = True  # keep the post-turn followup from flipping the phase
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._submit("go")
            assert app._voice_phase == "thinking"  # set synchronously when the turn starts
            await app.workers.wait_for_complete()

    _run(go())


def test_run_leg_swallows_callback_error_after_the_app_stops() -> None:
    # A voice leg still in flight when the app tears down calls back onto a dead UI thread;
    # the resulting RuntimeError must be dropped (the spoken turn is moot), not surface as an
    # unhandled thread exception. This app was never started, so is_running is False.
    app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
    assert app.is_running is False
    ran: list[bool] = []

    def boom() -> None:
        ran.append(True)
        raise RuntimeError("App is not running")

    app._run_leg(boom)  # returns without raising — the teardown-race error is swallowed
    assert ran == [True]  # the leg body did run; only its post-teardown error was dropped


def test_run_leg_reraises_a_genuine_failure_while_the_app_is_live() -> None:
    # While the app is running, a real exception in a leg is a bug and must propagate (so it's
    # reported), not be silently swallowed like the teardown race above.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), voice=FakeVoice())
        app._voice_paused = True  # no auto-listen thread racing this assertion
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.is_running is True

            def boom() -> None:
                raise ValueError("genuine bug")

            with pytest.raises(ValueError, match="genuine bug"):
                app._run_leg(boom)

    _run(go())


def test_toggle_voice_without_session_notifies_and_stays_off() -> None:
    # With no voice front-end the toggle is a no-op (notice only) and never marks a pause.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))  # no voice
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_toggle_voice()
            assert app._voice_paused is False  # nothing to pause
            assert app._voice_state() is None  # no badge without a session

    _run(go())
