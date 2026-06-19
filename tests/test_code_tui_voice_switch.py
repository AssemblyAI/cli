"""Tests for switching between voice and text mode in the `assembly code` TUI.

Switching input mode (Ctrl-V) and interrupting (Escape / Ctrl-C) both have to stop an
in-flight microphone capture so it neither keeps the mic open behind the text prompt nor
submits a turn the user no longer wants. These cancel-safety cases are split out of
test_code_tui_voice.py to keep each file under the 500-line gate, reusing that module's
app/voice doubles.
"""

from __future__ import annotations

import threading

import pytest

from aai_cli.code_agent.tui import CodeAgentApp
from tests.test_code_tui_voice import FakeAgent, FakeVoice, _run, _wait_until


def test_toggle_voice_off_cancels_in_flight_capture() -> None:
    # Switching to text (Ctrl-V) must release the mic now — cancel the blocking listen()
    # rather than leaving a capture running unseen behind the text prompt.
    async def go() -> None:
        voice = FakeVoice()
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        app._voice_paused = True  # start paused so on_mount doesn't race a capture thread
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_toggle_voice()  # voice on
            assert voice.cancels == 0  # turning on never cancels
            app.action_toggle_voice()  # voice off -> must cancel the in-flight capture
            assert voice.cancels == 1

    _run(go())


def test_capture_after_switching_to_text_is_not_submitted(monkeypatch: pytest.MonkeyPatch) -> None:
    # A turn that finalizes in the window between the user pressing Ctrl-V and the capture
    # unwinding must NOT be submitted — otherwise a spoken phrase lands as a turn after the
    # user already switched to typing.
    async def go() -> None:
        voice = FakeVoice()
        app = CodeAgentApp(agent=FakeAgent([]), voice=voice)
        app._voice_paused = True  # block the on_mount auto-listen
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            submitted: list[str] = []
            monkeypatch.setattr(app, "_submit", submitted.append)  # spy: _enter_and_submit calls it

            def listen() -> str:
                voice.listens += 1
                app._voice_paused = True  # user switched to text DURING the capture
                return "late turn"

            monkeypatch.setattr(voice, "listen", listen)
            app._voice_paused = False  # active when the capture starts
            thread = threading.Thread(target=app._capture_voice_turn)
            thread.start()
            assert await _wait_until(pilot, lambda: not thread.is_alive())
            assert submitted == []  # the late turn was dropped, not submitted

    _run(go())
