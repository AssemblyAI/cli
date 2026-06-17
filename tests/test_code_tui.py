"""Tests for the `assembly code` Textual TUI.

Pilot tests drive the real Textual app (headless) with a fake agent, so compose,
splash, the worker turn, event rendering, and the approval/ask modals are all
exercised without a network or a real terminal.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from textual.widgets import Input, RichLog

from aai_cli.code_agent import tui
from aai_cli.code_agent.events import AssistantText, ErrorText, ToolCall, ToolResult
from aai_cli.code_agent.tui import ApprovalScreen, CodeAgentApp


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
    assert tui._abbrev_home(Path("/etc/hosts")) == "/etc/hosts"


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
            app.query_one("#prompt", Input).value = "build"
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
            app._write_event(AssistantText("reply text"))
            app._write_event(ToolCall(name="write_file", args={"file_path": "a"}))
            app._write_event(ToolResult(name="write_file", content="Updated a"))
            app._write_event(ErrorText("kaboom"))
            assert app._last_reply == "reply text"
            app.action_copy_last()
            assert copied == ["reply text"]

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
    # by the approve/reject modal tests above).
    results: list[bool | None] = []

    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ApprovalScreen("execute", {"cmd": "ls"}), results.append)
            await pilot.pause()
            await pilot.click("#reject")
            await pilot.pause()

    _run(go())
    assert results == [False]
