"""Tests for the `assembly code` Textual TUI.

Pilot tests drive the real Textual app (headless) with a fake agent, so compose,
splash, the worker turn, event rendering, and the approval/ask modals are all
exercised without a network or a real terminal.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from textual.widgets import Input, RichLog, Static

from aai_cli.code_agent import tui
from aai_cli.code_agent.events import AssistantText, ErrorText, ToolCall, ToolResult
from aai_cli.code_agent.tui import ApprovalScreen, AskScreen, CodeAgentApp


class FakeAgent:
    """Replays scripted invoke() results (turn + interrupt-resume)."""

    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results
        self.calls = 0

    def invoke(self, *args, **kwargs):
        result = self._results[self.calls]
        self.calls += 1
        return result


class _Interrupt:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value


# --- pure helpers -------------------------------------------------------------


def test_format_args_and_abbrev_home() -> None:
    assert tui._format_args({"a": 1, "b": "x"}) == "a=1, b='x'"
    assert tui._abbrev_home(Path.home() / "proj") == "~/proj"
    # A path outside home renders as-is; compare to the platform-native string so this
    # holds on Windows (where str(Path(...)) uses backslashes) as well as POSIX.
    outside = Path("/etc/hosts")
    assert tui._abbrev_home(outside) == str(outside)


def test_approval_decision_defaults_to_reject() -> None:
    assert tui._approval_decision("approve") == "approve"
    assert tui._approval_decision("auto") == "auto"
    # A button with no id (Textual allows None) is treated as a rejection, not approval.
    assert tui._approval_decision(None) == "reject"
    assert tui._approval_decision("") == "reject"


def test_git_branch_and_status(tmp_path: Path) -> None:
    assert tui._git_branch(tmp_path) is None  # no .git
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/feature-x\n")
    assert tui._git_branch(tmp_path) == "feature-x"
    (tmp_path / ".git" / "HEAD").write_text("a1b2c3d4e5f6\n")  # detached
    assert tui._git_branch(tmp_path) == "a1b2c3d4"

    status = tui._status_text(tmp_path, auto_approve=True)
    assert "auto" in status and "a1b2c3d4" in status
    assert "manual" in tui._status_text(tmp_path, auto_approve=False)


# --- pilot tests --------------------------------------------------------------


def _run(coro) -> None:
    asyncio.run(coro)


def test_mount_renders_splash_and_focuses_input() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), web_note="no key", thread_id="t1")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            log = app.query_one("#log", RichLog)
            assert len(log.lines) > 6  # wordmark + tagline
            assert app.focused is app.query_one("#prompt", Input)

    _run(go())


def test_initial_prompt_runs_a_turn_on_mount() -> None:
    async def go() -> None:
        agent = FakeAgent([{"messages": [HumanMessage("seed"), AIMessage("seeded reply")]}])
        app = CodeAgentApp(agent=agent, initial="kick off")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert agent.calls == 1  # the initial prompt drove one turn

    _run(go())


def test_submit_runs_turn_and_renders_reply() -> None:
    async def go() -> None:
        agent = FakeAgent([{"messages": [HumanMessage("go"), AIMessage("all done")]}])
        app = CodeAgentApp(agent=agent)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # "[build" contains unbalanced Rich markup: without escaping, _submit's
            # log.write would raise MarkupError, so this also guards the escape().
            app.query_one("#prompt", Input).value = "[build"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one("#prompt", Input).disabled is False  # re-enabled

    _run(go())


def test_write_event_each_type_and_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    copied: list[str] = []
    monkeypatch.setattr("pyperclip.copy", copied.append)

    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # Each value carries unbalanced "[" markup: without escaping, RichLog.write
            # would raise MarkupError here, so these calls also guard the escape() paths.
            app._write_event(AssistantText("[reply"))
            app._write_event(ToolCall(name="write_file", args={"file_path": "[a"}))
            app._write_event(ToolResult(name="write_file", content="[unclosed"))
            app._write_event(ErrorText("[boom"))
            assert app._last_reply == "[reply"
            app.action_copy_last()
            assert copied == ["[reply"]

    _run(go())


def _drive_modal(app, call, keys: list[str]):
    """Run ``call`` (which blocks on a modal) on a thread; dismiss with ``keys``."""

    async def go():
        box: dict[str, object] = {}
        thread = threading.Thread(target=lambda: box.update(result=call()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            thread.start()
            for _ in range(200):
                await pilot.pause(0.01)
                if len(app.screen_stack) > 1:
                    break
            await pilot.press(*keys)
            thread.join(timeout=3)
            await pilot.pause()
        return box.get("result")

    return asyncio.run(go())


def test_approval_modal_approve_and_reject() -> None:
    app = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app, lambda: app._approve("write_file", {"file_path": "a"}), ["y"]) is True

    app2 = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app2, lambda: app2._approve("execute", {"cmd": "ls"}), ["n"]) is False


def test_ask_modal_returns_typed_answer() -> None:
    app = CodeAgentApp(agent=FakeAgent([]))
    answer = _drive_modal(app, lambda: app._ask("which port?"), ["8", "0", "8", "0", "enter"])
    assert answer == "8080"


def test_full_turn_with_approval_interrupt() -> None:
    async def go() -> None:
        agent = FakeAgent(
            [
                {
                    "__interrupt__": [
                        _Interrupt({"action_requests": [{"name": "write_file", "args": {}}]})
                    ]
                },
                {"messages": [HumanMessage("go"), AIMessage("written")]},
            ]
        )
        app = CodeAgentApp(agent=agent)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one("#prompt", Input).value = "write it"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause(0.01)
                if len(app.screen_stack) > 1:
                    break
            await pilot.press("y")  # approve
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert agent.calls == 2  # initial + resume

    _run(go())


def test_approval_button_press_dismisses() -> None:
    # Covers ApprovalScreen.on_button_pressed (the click path; key paths are covered
    # by the approve/reject modal tests above). The bracketed name/args also guard the
    # compose() escape() — without it, Label markup parsing would raise on mount.
    results: list[str | None] = []

    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ApprovalScreen("exec[", {"cmd": "[ls"}), results.append)
            await pilot.pause()
            await pilot.click("#reject")
            await pilot.pause()

    _run(go())
    assert results == ["reject"]


def test_approval_box_is_compact_and_bottom_docked() -> None:
    # Regression guard: the approval prompt must not take over the whole screen — it
    # docks a short box at the bottom so the transcript stays visible above it.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ApprovalScreen("write_file", {"file_path": "x.py"}))
            await pilot.pause()
            box = app.screen.query_one("#approvalbox")
            assert box.region.height <= 8  # a handful of rows, not the full 30
            assert box.region.bottom <= 30  # anchored within the bottom of the screen
            assert box.region.y >= 15  # sits in the lower half, transcript visible above

    _run(go())


def test_approval_auto_approve_flips_mode_and_skips_later_prompts() -> None:
    # Picking "Auto-approve (a)" approves this call, flips the badge manual→auto, and
    # makes every later _approve return True without ever pushing a modal.
    app = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app, lambda: app._approve("execute", {"cmd": "ls"}), ["a"]) is True
    assert app._auto_approve is True
    assert app._session.auto_approve is True
    # A second decision short-circuits: it returns True even though no modal can be driven.
    assert app._approve("write_file", {"file_path": "x"}) is True


def test_refresh_status_rerenders_badge() -> None:
    # _enable_auto_approve (worker thread) marshals a _refresh_status onto the UI thread;
    # this drives that re-render directly, asserting the badge tracks the mode flip.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert "manual" in str(app.query_one("#status", Static).render())
            app._auto_approve = True
            app._refresh_status()
            await pilot.pause()
            assert "auto" in str(app.query_one("#status", Static).render())

    _run(go())


def test_escape_interrupts_a_running_turn() -> None:
    # While a turn is in flight (prompt disabled), Escape signals the session to stop its
    # agent loop; it never quits the app. Drives the real "escape" binding end to end.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one("#prompt", Input).disabled = True  # simulate a turn in progress
            await pilot.press("escape")
            await pilot.pause()
            assert app._session._cancel.is_set()  # the loop was asked to stop

    _run(go())


def test_escape_is_a_noop_when_idle() -> None:
    # Idle (prompt enabled): Escape does nothing — no cancel signal, no quit.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_interrupt()  # idle: nothing to interrupt
            assert app._session._cancel.is_set() is False

    _run(go())


def test_ctrl_c_interrupts_running_turn_and_does_not_arm_quit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            exited: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.query_one("#prompt", Input).disabled = True  # a turn is running
            app.action_quit_or_interrupt()
            assert app._session._cancel.is_set()  # interrupted the turn
            assert exited == []  # did NOT quit, because a turn was in flight
            assert app._quit_pending is False  # interrupting never arms the quit hint

    _run(go())


def test_ctrl_c_needs_a_double_press_to_quit_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            exited: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
            app.action_quit_or_interrupt()  # first idle press: arms, does not quit
            assert exited == []
            assert app._quit_pending is True
            app.action_quit_or_interrupt()  # second press confirms the quit
            assert exited == [True]
            assert app._session._cancel.is_set() is False  # nothing was cancelled

    _run(go())


def test_clear_quit_pending_resets_the_flag() -> None:
    # The timer-fired reset (covered directly since the timer won't fire within the test).
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._quit_pending = True
            app._clear_quit_pending()
            assert app._quit_pending is False

    _run(go())


def test_spinner_text_formats_frame_and_elapsed() -> None:
    assert tui._spinner_text(46, "✶") == "✶ Working… (46s)"
    assert tui._spinner_text(0, "✷") == "✷ Working… (0s)"


def test_spinner_starts_ticks_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # Re-query for each display check: a stored `spinner.display` would let mypy
            # narrow the bool across the start/stop calls and flag the next assert dead.
            assert app.query_one("#spinner", Static).display is False  # hidden at rest
            app._start_spinner()
            await pilot.pause()
            assert app.query_one("#spinner", Static).display is True
            # _tick wires the elapsed seconds off the start time; pin "now" to assert it.
            monkeypatch.setattr(time, "monotonic", lambda: app._turn_started + 7.0)
            app._tick()
            assert "Working… (7s)" in str(app.query_one("#spinner", Static).render())
            app._stop_spinner()
            assert app.query_one("#spinner", Static).display is False
            assert app._spin_timer is None

    _run(go())


def test_stop_spinner_is_a_noop_when_not_started() -> None:
    # The timer-None branch of _stop_spinner: stopping before any turn just hides.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._stop_spinner()
            assert app.query_one("#spinner", Static).display is False

    _run(go())


def test_ask_screen_compose_escapes_markup() -> None:
    # Mounting AskScreen with a bracketed question exercises its compose() escape();
    # without it, the Label markup parse would raise MarkupError on mount.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(AskScreen("which port [x?"), lambda answer: None)
            await pilot.pause()
            app.screen.query_one("#answer", Input).value = "8080"
            await pilot.press("enter")
            await pilot.pause()

    _run(go())
