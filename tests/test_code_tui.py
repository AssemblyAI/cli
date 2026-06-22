"""Tests for the `assembly code` Textual TUI.

Pilot tests drive the real Textual app (headless) with a fake agent, so compose,
splash, the worker turn, event rendering, and the approval/ask modals are all
exercised without a network or a real terminal.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Input, Label, Static

from aai_cli.code_agent.events import AssistantText, ErrorText, ToolCall, ToolResult
from aai_cli.code_agent.modals import ApprovalScreen, AskScreen
from aai_cli.code_agent.tui import CodeAgentApp


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


# --- pilot tests --------------------------------------------------------------


def _run(coro) -> None:
    asyncio.run(coro)


def test_mount_renders_splash_and_focuses_input() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]), web_note="no key", thread_id="t1")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            log = app.query_one("#log", VerticalScroll)
            assert len(log.children) >= 1  # the splash is mounted into the transcript
            assert "Ready to code" in str(log.children[0].render())  # splash intro shown
            assert app.focused is app.query_one("#prompt", Input)
            # The bordered prompt bar must fit inside the screen so its right border isn't
            # clipped off-edge — `width: 100%` honors the side margins where the docked
            # default (`1fr`) would overflow to x=1..101 on a 100-wide screen.
            assert app.query_one("#promptbar", Horizontal).region.right <= 100

    _run(go())


def test_prompt_bar_does_not_overlap_status_footer() -> None:
    # The prompt bar and the two-row status footer both dock to the bottom, so docked
    # siblings overlay rather than stack: the bar's bottom margin must reserve the full
    # status height or the footer's top row paints over the box's bottom border (which
    # left the rounded box looking open at the bottom). region.bottom is exclusive, so
    # "no overlap" is bar.bottom <= status.y.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            bar = app.query_one("#promptbar", Horizontal).region
            status = app.query_one("#status", Static).region
            assert bar.bottom <= status.y

    _run(go())


def test_voicebar_render_after_the_bar_is_gone_is_a_safe_noop() -> None:
    # The 0.3s animation timer drives _render_voicebar and can fire one last tick during teardown,
    # after #voicebar is removed but before the interval is cancelled; it must no-op, not raise the
    # NoMatches that surfaced as a py3.13 CI flake.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await app.query_one("#voicebar", Static).remove()
            assert len(app.query("#voicebar")) == 0
            app._render_voicebar()  # must not raise now that the bar is gone

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


def test_approval_modal_dismisses_on_escape_or_ctrl_c() -> None:
    # Escape / Ctrl-C decline the tool (the safe cancel), like pressing "n".
    app = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app, lambda: app._approve("execute", {"cmd": "ls"}), ["escape"]) is False
    app2 = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app2, lambda: app2._approve("execute", {"cmd": "ls"}), ["ctrl+c"]) is False


def test_ask_modal_dismisses_on_escape_or_ctrl_c_with_no_answer() -> None:
    # Escape / Ctrl-C cancel the question; the agent gets an empty answer.
    app = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app, lambda: app._ask("which port?"), ["escape"]) == ""
    app2 = CodeAgentApp(agent=FakeAgent([]))
    assert _drive_modal(app2, lambda: app2._ask("which port?"), ["ctrl+c"]) == ""


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


def test_approval_prompt_renders_keyboard_hint() -> None:
    # The prompt is a plain y/a/n keyboard hint, not clickable buttons — assert each
    # option's copy renders so dropping one is caught. The bracketed name/args also guard
    # the compose() escape(): without it, Label markup parsing would raise on mount.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ApprovalScreen("exec[", {"cmd": "[ls"}))
            await pilot.pause()
            rendered = " ".join(str(label.render()) for label in app.screen.query(Label))
            assert "approve" in rendered
            assert "auto-approve" in rendered
            assert "reject" in rendered

    _run(go())


def test_approval_expands_args_on_e() -> None:
    # Collapsed, the prompt shows only the identifying arg (the filename); pressing `e`
    # expands it to the full args, revealing the file content that was elided.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(
                ApprovalScreen("write_file", {"file_path": "x.py", "content": "SECRET"})
            )
            await pilot.pause()
            detail = app.screen.query_one("#approvaldetail", Label)
            assert "SECRET" not in str(detail.render())  # collapsed: content elided
            await pilot.press("e")
            await pilot.pause()
            assert "SECRET" in str(detail.render())  # expanded: full args shown
            await pilot.press("e")  # toggles back
            await pilot.pause()
            assert "SECRET" not in str(detail.render())

    _run(go())


def test_approval_shows_risk_warning_for_dangerous_command() -> None:
    # A destructive shell command carries a one-line warning above the prompt; a benign one
    # mounts no warning label at all.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ApprovalScreen("execute", {"command": "rm -rf build/"}))
            await pilot.pause()
            warn = app.screen.query("#approvalwarn")
            assert warn  # warning present
            assert "deletes files" in str(warn.first().render())
            app.pop_screen()
            await pilot.pause()
            app.push_screen(ApprovalScreen("execute", {"command": "ls -la"}))
            await pilot.pause()
            assert not app.screen.query("#approvalwarn")  # benign: no warning mounted

    _run(go())


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
            # The box must fit inside the screen so its rounded border isn't clipped off the
            # right edge: a docked `width: 1fr` container ignores horizontal margin and
            # overflows to x=1..101 on a 100-wide screen (the bug `width: 100%` fixes).
            assert box.region.right <= 100

    _run(go())


def test_modals_are_transparent_so_transcript_stays_visible() -> None:
    # Regression guard: the app's `Screen { background: #000000 }` canvas rule matches every
    # Screen subclass, and app CSS beats a widget's DEFAULT_CSS — so without the explicit
    # `ModalScreen { background: transparent }` app rule, the modal paints opaque black and
    # blanks the transcript behind it. Assert each modal resolves to a see-through background
    # (alpha 0); an opaque modal (alpha 1.0) — the bug — fails here.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ApprovalScreen("write_file", {"file_path": "x.py"}))
            await pilot.pause()
            assert app.screen.styles.background.a == 0  # approval modal is see-through
            app.pop_screen()
            await pilot.pause()
            app.push_screen(AskScreen("which port?"))
            await pilot.pause()
            assert app.screen.styles.background.a == 0  # ask modal is see-through

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
            # Stop the live interval first so only this deterministic tick writes the
            # readout — otherwise a real-time auto-tick can race the assert on a loaded
            # runner, which flaked CI with "(6s)" vs "(7s)". update()->render() is
            # synchronous, so no pilot.pause() is needed (and pausing here deadlocks).
            assert app._spin_timer is not None
            app._spin_timer.stop()
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
