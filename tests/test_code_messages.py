"""Tests for the mounted-widget transcript of the `assembly code` TUI.

Drives the real Textual app (headless) and asserts on the mounted message widgets: the reply
streams into one AssistantMessage in place and renders as Markdown, and a long tool result is
a collapsible ToolOutput (Ctrl-O / click). Split from test_code_tui.py to stay under the
file-length gate.
"""

from __future__ import annotations

import asyncio

from aai_cli.code_agent.events import AssistantDelta, AssistantText, ToolResult
from aai_cli.code_agent.messages import AssistantMessage, ToolOutput, UserMessage
from aai_cli.code_agent.tui import CodeAgentApp


class FakeAgent:
    """Replays scripted invoke() results so a turn can complete without a model."""

    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = results
        self.calls = 0

    def invoke(self, *args, **kwargs):
        result = self._results[self.calls]
        self.calls += 1
        return result


def _run(coro) -> None:
    asyncio.run(coro)


def test_assistant_reply_renders_as_markdown_widget() -> None:
    # The reply mounts an AssistantMessage rendered as Markdown — the fence markers are
    # consumed and the code shows; the raw text is kept for clipboard copy.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            reply = "Here you go:\n\n```python\nprint('hi')\n```"
            app._write_event(AssistantText(reply))
            await pilot.pause()
            msg = app.query_one(AssistantMessage)
            text = "\n".join(msg.render_line(y).text for y in range(msg.size.height))
            assert "```" not in text  # markdown consumed the fence markers
            assert "print('hi')" in text  # the code itself renders
            assert app._last_reply == reply  # raw markdown kept for clipboard copy

    _run(go())


def test_assistant_deltas_stream_in_place_then_finalize() -> None:
    # Tokens stream into a single AssistantMessage in place (no separate region); the final
    # AssistantText finalizes that same widget rather than mounting a second one.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._write_event(AssistantDelta("Hello, "))
            app._write_event(AssistantDelta("world!"))
            await pilot.pause()
            assert len(app.query(AssistantMessage)) == 1  # one widget, updated in place
            assert app.query_one(AssistantMessage).text == "Hello, world!"
            streaming = app._streaming_msg  # local: asserting on the attr would poison the
            assert streaming is not None  # later `is None` check (mypy can't see the reset)
            app._write_event(AssistantText("Hello, world!"))
            await pilot.pause()
            assert app._streaming_msg is None  # finalized
            assert app._last_reply == "Hello, world!"
            assert len(app.query(AssistantMessage)) == 1  # finalized in place, not a 2nd widget

    _run(go())


def test_finish_turn_finalizes_a_dangling_streamed_reply() -> None:
    # A turn cancelled mid-generation leaves a streamed-but-unfinalized reply; finishing the
    # turn commits what streamed in (so it isn't lost) and clears the streaming reference.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._write_event(AssistantDelta("partial repl"))
            await pilot.pause()
            streaming = app._streaming_msg  # local so the later `is None` check stays reachable
            assert streaming is not None
            app._finish_turn()
            assert app._streaming_msg is None  # finalized, not left dangling
            assert app.query_one(AssistantMessage).text == "partial repl"  # kept what streamed

    _run(go())


def test_user_message_prefixes_and_set_text_replaces_in_place() -> None:
    # The prompt echo carries the "» " prefix; set_text() swaps the body in place (used to grow
    # an interim voice transcript), keeping the same widget rather than mounting a new line.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            msg = UserMessage("hi")
            await app.query_one("#log").mount(msg)
            assert "» hi" in str(msg.render())
            msg.set_text("hi there friend")
            assert "» hi there friend" in str(msg.render())  # body replaced, not appended

    _run(go())


def test_short_tool_output_is_not_expandable() -> None:
    # Output that already fits has no expand affordance and Ctrl-O is a no-op on it.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._write_event(ToolResult(name="execute", content="ok"))
            await pilot.pause()
            out = app.query_one(ToolOutput)
            before = str(out.render())
            assert "Ctrl+O" not in before  # nothing to expand -> no hint
            out.toggle()
            assert str(out.render()) == before  # toggle is a no-op when it all fits

    _run(go())


def test_tool_output_toggles_on_click_and_ctrl_o_is_safe_with_no_output() -> None:
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_toggle_output()  # no tool output yet -> safe no-op
            app._write_event(
                ToolResult(name="execute", content="\n".join(f"x{i}" for i in range(20)))
            )
            await pilot.pause()
            out = app.query_one(ToolOutput)
            assert "x19" not in str(out.render())
            out.on_click()  # clicking expands
            assert "x19" in str(out.render())

    _run(go())


def test_tool_output_expands_and_collapses_on_ctrl_o() -> None:
    # A long tool result mounts a collapsed ToolOutput (preview + "more lines"); Ctrl-O
    # expands it to the full content and toggles back.
    async def go() -> None:
        app = CodeAgentApp(agent=FakeAgent([]))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._write_event(
                ToolResult(name="execute", content="\n".join(f"ln{i}" for i in range(20)))
            )
            await pilot.pause()
            out = app.query_one(ToolOutput)
            collapsed = str(out.render())
            assert "ln0" in collapsed and "more lines" in collapsed and "ln19" not in collapsed
            app.action_toggle_output()  # Ctrl-O expands the most recent output
            assert "ln19" in str(out.render())  # full content now shown
            app.action_toggle_output()  # toggles back to the preview
            assert "ln19" not in str(out.render())

    _run(go())
