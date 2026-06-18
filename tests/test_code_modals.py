"""Tests for the spoken/voice-answerable approval and ask modals.

The pure ``approval_from_speech`` mapping is unit-tested directly; the screen wiring (speak the
prompt, listen, dismiss with the mapped decision) is driven through the real app headless with
a scripted voice double — no mic, speaker, or socket.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input

from aai_cli.code_agent.modals import ApprovalScreen, AskScreen, approval_from_speech
from aai_cli.code_agent.tui import CodeAgentApp
from aai_cli.core.errors import CLIError


class FakeAgent:
    def invoke(self, *a, **k):
        return {}


class FakeVoice:
    """Scripted voice IO: speak() records, listen() replays one transcript (or raises)."""

    def __init__(self, transcript: str | None = None, *, error: CLIError | None = None) -> None:
        self._transcript = transcript
        self._error = error
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def listen(self) -> str | None:
        if self._error is not None:
            raise self._error
        return self._transcript

    def cancel(self) -> None:
        """No-op: the modal voice path never interrupts an in-flight leg."""


def _run(coro) -> None:
    asyncio.run(coro)


@pytest.mark.parametrize(
    ("said", "decision"),
    [
        ("yes please", "approve"),
        ("approve that", "approve"),
        ("go ahead", "approve"),
        ("auto approve", "auto"),
        ("always do this", "auto"),
        ("no", "reject"),
        ("reject it", "reject"),
        ("don't", "reject"),
        ("yes but no", "reject"),  # reject wins over approve when both are heard (safer)
        ("uhh what", "reject"),  # unclear -> safe default
    ],
)
def test_approval_from_speech(said: str, decision: str) -> None:
    assert approval_from_speech(said) == decision


async def _push_and_wait(app, pilot, screen) -> object:
    box: dict[str, object] = {}
    app.push_screen(screen, lambda result: box.update(value=result))
    for _ in range(300):
        await pilot.pause(0.01)
        if "value" in box:
            break
    return box.get("value", "__pending__")


def test_spoken_approval_speaks_prompt_and_maps_answer() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            voice = FakeVoice(transcript="yes go for it")
            result = await _push_and_wait(
                app, pilot, ApprovalScreen("execute", {"command": "rm -rf build"}, voice=voice)
            )
            assert result == "approve"  # spoken "yes" mapped to approve
            prompt = voice.spoken[0]
            assert "Run execute" in prompt and "rm -rf build" in prompt
            assert "Warning:" in prompt  # the risky command is read aloud
            assert "approve, auto-approve, or reject" in prompt

    _run(go())


def test_spoken_approval_rejects_on_no() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            result = await _push_and_wait(
                app, pilot, ApprovalScreen("write_file", {"file_path": "x"}, voice=FakeVoice("no"))
            )
            assert result == "reject"

    _run(go())


def test_spoken_ask_speaks_question_and_returns_transcript() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            voice = FakeVoice(transcript="use port 8080")
            result = await _push_and_wait(app, pilot, AskScreen("Which port?", voice=voice))
            assert result == "use port 8080"  # spoken answer returned verbatim
            assert "The agent asks: Which port?" in voice.spoken[0]

    _run(go())


def test_silence_does_not_auto_reject() -> None:
    # No speech (listen -> None) must not auto-decide — the modal waits for speech or a keypress
    # rather than rejecting a tool on a pause.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            box: dict[str, object] = {}
            app.push_screen(
                ApprovalScreen("execute", {"command": "ls"}, voice=FakeVoice(None)),
                lambda result: box.update(value=result),
            )
            for _ in range(50):
                await pilot.pause(0.01)
            assert "value" not in box  # silence -> not dismissed

    _run(go())


def test_voice_failure_falls_back_to_keyboard() -> None:
    # If the mic/STT fails, the modal isn't auto-dismissed — the user can still press a key.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            voice = FakeVoice(error=CLIError("no mic", error_type="mic_missing", exit_code=2))
            box: dict[str, object] = {}
            app.push_screen(
                ApprovalScreen("execute", {"command": "ls"}, voice=voice),
                lambda result: box.update(value=result),
            )
            for _ in range(50):
                await pilot.pause(0.01)
            assert "value" not in box  # voice failed -> not auto-dismissed
            await pilot.press("n")  # keyboard still works
            await pilot.pause()
            assert box.get("value") == "reject"

    _run(go())


def test_ask_voice_failure_falls_back_to_typing() -> None:
    # An ask modal whose voice fails isn't dismissed; the user types the answer instead.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            voice = FakeVoice(error=CLIError("no mic", error_type="mic_missing", exit_code=2))
            box: dict[str, object] = {}
            app.push_screen(AskScreen("Which port?", voice=voice), lambda r: box.update(value=r))
            for _ in range(50):
                await pilot.pause(0.01)
            assert "value" not in box  # voice failed -> not auto-dismissed
            app.screen.query_one("#answer", Input).value = "8080"
            await pilot.press("enter")
            await pilot.pause()
            assert box.get("value") == "8080"

    _run(go())


def test_spoken_prompt_omits_detail_when_no_args() -> None:
    # A tool with no identifying arg reads as just "Run <tool>. Say approve…" (no detail clause).
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            voice = FakeVoice(transcript="yes")
            result = await _push_and_wait(app, pilot, ApprovalScreen("noop", {}, voice=voice))
            assert result == "approve"
            assert "Run noop. Say approve" in voice.spoken[0]  # straight to the options

    _run(go())


def test_ask_silence_does_not_dismiss() -> None:
    # No spoken answer (listen -> None) leaves the ask modal up for typing.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            box: dict[str, object] = {}
            app.push_screen(AskScreen("Q?", voice=FakeVoice(None)), lambda r: box.update(value=r))
            for _ in range(50):
                await pilot.pause(0.01)
            assert "value" not in box  # silence -> not dismissed

    _run(go())


def test_decide_and_answer_are_idempotent() -> None:
    # A spoken reply and a keypress can race; the second one is ignored so the modal dismisses
    # exactly once with the first decision.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            approval: dict[str, object] = {}
            screen = ApprovalScreen("execute", {"command": "ls"})
            app.push_screen(screen, lambda r: approval.update(value=r))
            await pilot.pause()
            screen._decide("approve")  # first decision dismisses
            await pilot.pause()
            screen._decide("reject")  # second is ignored (already answered)
            await pilot.pause()
            assert approval["value"] == "approve"

            answer: dict[str, object] = {}
            ask = AskScreen("Q?")
            app.push_screen(ask, lambda r: answer.update(value=r))
            await pilot.pause()
            ask._answer("first")
            await pilot.pause()
            ask._answer("second")  # ignored
            await pilot.pause()
            assert answer["value"] == "first"

    _run(go())
