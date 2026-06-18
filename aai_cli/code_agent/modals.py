"""Bottom-docked modal screens for the coding-agent TUI: tool approval and agent questions.

Split out of `tui.py` to keep each module under the file-length gate. Both are transparent
``ModalScreen``s docked at the bottom, so the transcript stays visible above them (see the
``ModalScreen { background: transparent }`` rule in :class:`~aai_cli.code_agent.tui.CodeAgentApp`).

In voice mode each modal is also **spoken and voice-answerable**: when constructed with a
``voice`` IO it speaks the prompt and listens for a spoken reply (approve / auto / reject, or a
free-text answer), off the UI thread. The keyboard path always stays available as a fallback.
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label

from aai_cli.code_agent import banner, risk
from aai_cli.code_agent.summarize import describe_args, full_args
from aai_cli.core import errors

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from aai_cli.code_agent.voice_ui import _VoiceIO


def _spawn(target: Callable[[], None]) -> None:
    """Run ``target`` on a daemon thread — the voice legs block, so they stay off the UI thread."""
    threading.Thread(target=target, daemon=True).start()  # pragma: no mutate


# Spoken-answer vocabulary. "auto" wins first (it implies approval); an unclear answer falls
# back to "reject" — the same safe default as the keyboard, so a tool never runs on a guess.
_REJECT_WORDS = frozenset({"no", "reject", "deny", "stop", "cancel", "nope", "nah"})
_APPROVE_WORDS = frozenset({"yes", "approve", "yeah", "yep", "yup", "sure", "ok", "okay"})


def approval_from_speech(text: str) -> str:
    """Map a spoken reply to ``"approve"`` / ``"auto"`` / ``"reject"`` (unclear → reject)."""
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    if "auto" in lowered or "always" in lowered:
        return "auto"
    if words & _REJECT_WORDS or "don't" in lowered or "do not" in lowered:
        return "reject"
    if words & _APPROVE_WORDS or "go ahead" in lowered or "do it" in lowered:
        return "approve"
    return "reject"


class ApprovalScreen(ModalScreen[str]):
    """A compact, bottom-docked prompt to approve/auto-approve/reject one tool call.

    Keyboard ``y / a / n`` (and ``e`` to expand the args); in voice mode it also speaks the
    prompt and accepts a spoken approve/auto/reject. The transparent background leaves the
    transcript visible, and a risky call (``rm -rf``, an internal fetch) carries a warning.
    """

    DEFAULT_CSS = """
    ApprovalScreen { align: center bottom; background: transparent; }
    ApprovalScreen #approvalbox {
        dock: bottom; width: 1fr; height: auto;
        border: round #f59e0b; background: #000000; padding: 0 1; margin: 0 1 1 1;
    }
    ApprovalScreen #approvalbox Label { height: auto; }
    """
    BINDINGS: ClassVar = [
        ("y", "approve", "Approve"),
        ("a", "auto", "Auto-approve"),
        ("n", "reject", "Reject"),
        ("e", "expand", "Expand"),
    ]

    def __init__(
        self, name: str, args: Mapping[str, object], *, voice: _VoiceIO | None = None
    ) -> None:
        super().__init__()
        self._tool_name = name  # not _name: that shadows Textual Widget's str|None attr
        self._args = args
        self._expanded = False  # toggled by `e`; collapsed (one-line) by default
        self._voice = voice  # when set, the prompt is spoken and a spoken answer is accepted
        self._answered = False  # guards against a voice answer and a keypress both dismissing

    def compose(self) -> ComposeResult:
        with Vertical(id="approvalbox"):
            warning = risk.risk_warning(self._tool_name, self._args)
            if warning:
                yield Label(f"[b #f04438]⚠ {escape(warning)}[/]", id="approvalwarn")
            yield Label(self._detail_markup(), id="approvaldetail")
            yield Label(
                f"[b #22c55e]y[/] approve   [b {banner.BRAND_HEX}]a[/] auto-approve   "
                "[b #f04438]n[/] reject   [b]e[/] expand"
            )

    def on_mount(self) -> None:
        if (voice := self._voice) is not None:  # drive the decision by voice, off the UI thread
            _spawn(lambda: self._drive_by_voice(voice))

    def _drive_by_voice(self, voice: _VoiceIO) -> None:
        """Speak the prompt and accept a spoken approve/auto/reject (keyboard still works)."""
        try:
            voice.speak(self._spoken_prompt())
            transcript = voice.listen()
        except errors.CLIError:
            return  # mic/STT failed: leave the keyboard hint as the way to answer
        if transcript:  # silence (None) must not auto-reject a tool — wait for speech or a key
            self.app.call_from_thread(self._decide, approval_from_speech(transcript))

    def _spoken_prompt(self) -> str:
        """The read-aloud version of the prompt: the tool, its arg, any warning, the options."""
        parts = [f"Run {self._tool_name}."]
        detail = describe_args(self._args)
        if detail:
            parts.append(f"{detail}.")
        warning = risk.risk_warning(self._tool_name, self._args)
        if warning:
            parts.append(f"Warning: {warning}")
        parts.append("Say approve, auto-approve, or reject.")
        return " ".join(parts)

    def _decide(self, decision: str) -> None:
        """Dismiss once, whether the answer came by spoken reply or keypress."""
        if self._answered:
            return
        self._answered = True
        self.dismiss(decision)

    def _detail_markup(self) -> str:
        """The 'Run tool X?' line — the compact arg, or the full args when expanded."""
        args = full_args(self._args) if self._expanded else describe_args(self._args)
        return f"Run tool [b]{escape(self._tool_name)}[/b]?  [dim]{escape(args)}[/dim]"

    def action_expand(self) -> None:
        """Toggle between the compact identifying arg and the full args (``e``)."""
        self._expanded = not self._expanded
        self.query_one("#approvaldetail", Label).update(self._detail_markup())

    def action_approve(self) -> None:
        self._decide("approve")

    def action_auto(self) -> None:
        self._decide("auto")

    def action_reject(self) -> None:
        self._decide("reject")


class AskScreen(ModalScreen[str]):
    """A bottom-docked prompt that relays a question from the agent and returns the answer.

    In voice mode it speaks the question and takes a spoken answer; otherwise the user types.
    """

    DEFAULT_CSS = """
    AskScreen { align: center bottom; background: transparent; }
    AskScreen #askbox {
        dock: bottom; width: 1fr; height: auto;
        border: round #3a3f55; background: #000000; padding: 0 1; margin: 0 1 1 1;
    }
    """

    def __init__(self, question: str, *, voice: _VoiceIO | None = None) -> None:
        super().__init__()
        self._question = question
        self._voice = voice
        self._answered = False

    def compose(self) -> ComposeResult:
        with Vertical(id="askbox"):
            yield Label(f"[b]The agent asks:[/b] {escape(self._question)}")
            yield Input(id="answer", placeholder="Type your answer and press Enter…")

    def on_mount(self) -> None:
        if (voice := self._voice) is not None:
            _spawn(lambda: self._drive_by_voice(voice))

    def _drive_by_voice(self, voice: _VoiceIO) -> None:
        """Speak the question and submit a spoken answer (typing still works)."""
        try:
            voice.speak(f"The agent asks: {self._question}")
            transcript = voice.listen()
        except errors.CLIError:
            return
        if transcript:
            self.app.call_from_thread(self._answer, transcript)

    def _answer(self, text: str) -> None:
        """Dismiss once with the answer, whether spoken or typed."""
        if self._answered:
            return
        self._answered = True
        self.dismiss(text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._answer(event.value)
