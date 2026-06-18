"""The voice front-end legs for the coding-agent TUI, split out to keep `tui.py` small.

These are the speak-to-it / read-back mechanics that run *off* the UI thread (mic capture and
TTS readback block), marshaling back via ``call_from_thread``. They live in a mixin that
:class:`~aai_cli.code_agent.tui.CodeAgentApp` inherits, so the app stays one ``App`` with the
voice methods folded in. The render/toggle side (the voice bar, Ctrl-V) stays in `tui.py`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol

from textual.app import App
from textual.widgets import Input

from aai_cli.code_agent.voice import spoken_summary
from aai_cli.core import errors

if TYPE_CHECKING:
    from collections.abc import Callable


class _VoiceIO(Protocol):
    """The speak-to-it / read-back slice the TUI drives; :class:`VoiceSession` satisfies it."""

    def listen(self) -> str | None:
        """Capture one spoken turn and return its transcript (``None`` on no speech)."""

    def speak(self, text: str) -> None:
        """Read ``text`` back aloud (a no-op when readback is unavailable)."""

    def cancel(self) -> None:
        """Stop an in-flight listen/readback so the current voice activity ends promptly."""


class _VoiceLegs(App[None]):
    """Mixin holding the off-thread voice capture/readback legs for ``CodeAgentApp``.

    Extends ``App`` so the inherited ``query_one``/``call_from_thread`` are typed; the voice
    state and the few app methods it leans on (``_set_voice_phase``/``_sync_input_mode``/
    ``_submit``) are provided by the concrete app and declared here for the type checker.
    """

    if TYPE_CHECKING:  # provided by CodeAgentApp (state set in __init__, methods defined there)
        _voice: _VoiceIO | None
        _voice_typed: bool
        _voice_paused: bool
        _last_reply: str

        def _set_voice_phase(self, phase: str) -> None: ...
        def _sync_input_mode(self) -> None: ...
        def _submit(self, text: str) -> None: ...
        def _note(self, text: str) -> None: ...

    def _voice_active(self) -> bool:
        """Voice capture is on: a session exists, the mic isn't ruled out, and it isn't paused."""
        return self._voice is not None and not self._voice_typed and not self._voice_paused

    def _spawn(self, target: Callable[[], None]) -> None:
        """Run ``target`` on a daemon thread — voice legs block, so they stay off the UI thread."""
        thread = threading.Thread(
            target=lambda: self._run_leg(target),
            daemon=True,  # pragma: no mutate — daemon flag only affects process exit, unassertable
        )
        thread.start()

    def _run_leg(self, target: Callable[[], None]) -> None:
        """Run one voice leg, dropping the callback error a torn-down app raises mid-flight.

        A leg calls back onto the UI thread (``call_from_thread``); if the app stops — a quit,
        or a test's ``run_test`` block exiting — while the leg is mid-call, that callback raises
        ``RuntimeError`` in this daemon thread, which would otherwise surface as an unhandled
        thread exception (a flaky Windows CI failure). The spoken turn is moot once the app is
        gone, so swallow it then; a genuine failure while the app is still live still propagates.
        """
        try:
            target()
        except Exception:
            if self.is_running:
                raise

    def _begin_listening(self) -> None:
        """Capture the next spoken turn on a background thread (no-op when voice is off)."""
        if not self._voice_active():
            return
        self._spawn(self._capture_voice_turn)

    def _voice_followup(self) -> None:
        """After a turn finishes: read back a spoken summary, then listen for the next turn."""
        voice = self._voice
        if voice is None or self._voice_paused:  # paused via Ctrl-V: no readback, no listen
            return
        self._spawn(lambda: self._speak_then_listen(voice))

    def _speak_then_listen(self, voice: _VoiceIO) -> None:
        """Read a summary of the last reply aloud (no code), then capture the next spoken turn."""
        self.call_from_thread(self._set_voice_phase, "speaking")
        voice.speak(spoken_summary(self._last_reply))
        self._capture_voice_turn()

    def _capture_voice_turn(self) -> None:
        """Listen for one spoken turn; enter it into the prompt, or degrade to typing."""
        voice = self._voice
        if voice is None or self._voice_typed or self._voice_paused:
            return
        self.call_from_thread(self._set_voice_phase, "listening")
        try:
            transcript = voice.listen()
        except errors.CLIError as exc:
            # A capture failure (no mic, STT error) drops voice for the rest of the session
            # rather than wedging it — the user just types instead.
            self._voice_typed = True
            self.call_from_thread(self._notice_voice_off, exc.message)
            return
        if transcript:
            self.call_from_thread(self._enter_and_submit, transcript)

    def _notice_voice_off(self, detail: str) -> None:
        """Tell the user voice input stopped and that input is now typed (UI thread)."""
        self._note(f"voice input off: {detail}; type your request instead")
        self._sync_input_mode()  # mic ruled out -> bring the text box back

    def _enter_and_submit(self, text: str) -> None:
        """Show the spoken text in the prompt, then submit it as a turn (UI thread)."""
        prompt = self.query_one("#prompt", Input)
        prompt.value = text
        self._submit(text)
        prompt.value = ""
