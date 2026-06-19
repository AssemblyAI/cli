"""Visual-regression snapshots for the `assembly code` and `assembly live` Textual TUIs.

Each test renders an app (or a pushed modal) to an SVG via ``pytest-textual-snapshot``'s
``snap_compare`` fixture and diffs it against a committed golden under
``tests/__snapshots__/test_tui_snapshots/``. This pins the *painted frame* — the splash, the
prompt bar, the docked status line, the voice bar, the message widgets, and the compact
approval/ask modals — so a CSS, layout, or docking regression that the per-widget pilot tests
(``test_code_tui.py`` / ``test_live_tui.py``) can't see fails loudly here instead.

Regenerate after an intentional UI change with ``uv run pytest tests/test_tui_snapshots.py
--snapshot-update`` and **eyeball every changed SVG** before committing — a snapshot only
guards against regressions if the baseline it captured was actually correct. The helpers in
``tests/_tui_snapshot.py`` freeze the four sources of non-determinism (version string, voice-bar
animation, the cascade worker, and the cwd/branch status line); see that module's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.widgets import Static

from aai_cli.agent_cascade.tui import LiveAgentApp
from aai_cli.code_agent.events import AssistantDelta, AssistantText, ErrorText, ToolCall, ToolResult
from aai_cli.code_agent.messages import UserMessage
from aai_cli.code_agent.modals import ApprovalScreen, AskScreen
from aai_cli.code_agent.tui import _SPIN_FRAMES, CodeAgentApp
from aai_cli.code_agent.tui_status import _spinner_text
from tests import _tui_snapshot as h

if TYPE_CHECKING:
    from textual.pilot import Pilot


# More than the 4-line preview budget, so summarize_result clips it and the ToolOutput
# row becomes expandable — the collapsed/expanded snapshots below pin both states.
_LONG_OUTPUT = "\n".join(f"tests/test_module_{i}.py .... [ {i * 10}%]" for i in range(8))


@pytest.fixture(autouse=True)
def _pin_version(monkeypatch: pytest.MonkeyPatch) -> None:
    h.pin_banner_version(monkeypatch)


def test_fake_agent_returns_empty_state() -> None:
    """The snapshot double satisfies CompiledAgent.invoke with an inert empty turn."""
    assert h.FakeAgent().invoke("prompt") == {}


# --- assembly code -----------------------------------------------------------


def test_code_splash(snap_compare, tmp_path, monkeypatch) -> None:
    """The idle startup frame: ASSEMBLY wordmark splash, prompt bar, and `manual` status line."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_status_auto_approve(snap_compare, tmp_path, monkeypatch) -> None:
    """Auto-approve flips the bottom badge from `manual` to `auto` — a one-glyph status diff."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)

    assert snap_compare(
        h.build_code_app(cwd=cwd, auto_approve=True),
        terminal_size=h.TERMINAL_SIZE,
        run_before=run_before,
    )


def test_code_transcript(snap_compare, tmp_path, monkeypatch) -> None:
    """A populated transcript: the user echo, a Markdown reply, a tool-call line, tool output."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, CodeAgentApp)
        h.freeze_animation(app)
        app._mount(UserMessage("add a /health endpoint"))
        app._write_event(AssistantText("Adding a **health check**:\n\n1. New route\n2. A test"))
        app._write_event(ToolCall(name="write_file", args={"file_path": "app.py"}))
        app._write_event(ToolResult(name="write_file", content="wrote 8 lines to app.py"))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_approval_modal(snap_compare, tmp_path, monkeypatch) -> None:
    """The compact, bottom-docked approval prompt for a risky command (warning + y/a/n hint)."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)
        pilot.app.push_screen(ApprovalScreen("execute", {"command": "rm -rf build/"}))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_ask_modal(snap_compare, tmp_path, monkeypatch) -> None:
    """The bottom-docked ask prompt: the agent's question above a text input."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)
        pilot.app.push_screen(AskScreen("Which port should the dev server use?"))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_approval_modal_expanded(snap_compare, tmp_path, monkeypatch) -> None:
    """`e` expands the approval prompt from the identifying arg to the full args.

    Collapsed, a write_file call shows only the filename; expanded, it reveals the file
    content that was elided — a taller box, pinned so the reveal can't regress.
    """
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)
        pilot.app.push_screen(
            ApprovalScreen(
                "write_file", {"file_path": "app.py", "content": "PORT = 8080\nDEBUG = 1"}
            )
        )

    assert snap_compare(
        h.build_code_app(cwd=cwd), press=["e"], terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_tool_output_collapsed(snap_compare, tmp_path, monkeypatch) -> None:
    """Long tool output clips to a preview with a `(Ctrl+O to expand)` hint."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, CodeAgentApp)
        h.freeze_animation(app)
        app._mount(UserMessage("run the tests"))
        app._write_event(ToolCall(name="execute", args={"command": "pytest -q"}))
        app._write_event(ToolResult(name="execute", content=_LONG_OUTPUT))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_tool_output_expanded(snap_compare, tmp_path, monkeypatch) -> None:
    """Ctrl+O expands the clipped tool output to the full content with a collapse hint."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, CodeAgentApp)
        h.freeze_animation(app)
        app._mount(UserMessage("run the tests"))
        app._write_event(ToolCall(name="execute", args={"command": "pytest -q"}))
        app._write_event(ToolResult(name="execute", content=_LONG_OUTPUT))
        await pilot.pause()  # let the ToolOutput mount before toggling it
        app.action_toggle_output()  # Ctrl+O

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_working_spinner(snap_compare, tmp_path, monkeypatch) -> None:
    """The working indicator: a spinner glyph + elapsed seconds, docked just above the prompt."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, CodeAgentApp)
        h.freeze_animation(app)
        app._mount(UserMessage("build a web scraper"))
        spinner = app.query_one("#spinner", Static)
        spinner.display = True
        # Render a fixed elapsed/frame through the real formatter — driving the live _tick
        # would tie the readout to wall-clock timing and flake.
        spinner.update(_spinner_text(7, _SPIN_FRAMES[0]))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_streaming_reply(snap_compare, tmp_path, monkeypatch) -> None:
    """A reply mid-stream is plain text (literal markdown) before finalize swaps it to Markdown."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, CodeAgentApp)
        h.freeze_animation(app)
        app._mount(UserMessage("explain the plan"))
        app._write_event(AssistantDelta("Here's the plan. First **scaffold** the project, "))
        app._write_event(AssistantDelta("then wire up the tests."))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_approval_modal_benign(snap_compare, tmp_path, monkeypatch) -> None:
    """A benign command mounts no warning label — the no-warning variant of the approval prompt."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)
        pilot.app.push_screen(ApprovalScreen("execute", {"command": "ls -la"}))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


def test_code_error(snap_compare, tmp_path, monkeypatch) -> None:
    """A failed turn renders as a red ✗ error line instead of crashing the UI."""
    cwd = h.stable_workdir(tmp_path, monkeypatch)

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, CodeAgentApp)
        h.freeze_animation(app)
        app._mount(UserMessage("deploy to prod"))
        app._write_event(ErrorText("gateway unreachable: connection refused"))

    assert snap_compare(
        h.build_code_app(cwd=cwd), terminal_size=h.TERMINAL_SIZE, run_before=run_before
    )


# --- assembly live -----------------------------------------------------------


def test_live_splash_listening(snap_compare) -> None:
    """The hands-free startup frame: the wordmark splash above the blue `Listening…` voice bar."""

    async def run_before(pilot: Pilot[None]) -> None:
        h.freeze_animation(pilot.app)

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_conversation(snap_compare) -> None:
    """A spoken turn mid-reply: the user transcript, the streamed reply, the green `Speaking…` bar."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_final("what's the weather like in Boston?")
        app.begin_reply()
        app.show_agent_sentence("It's sunny and about sixty degrees right now.")
        h.freeze_animation(app)  # begin_reply switched the phase, which repainted the bar

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_thinking(snap_compare) -> None:
    """After a finalized turn, the bar shows the amber `Thinking…` phase before the reply."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_final("what's the weather like in Boston?")
        h.freeze_animation(app)  # show_user_final switched the phase to thinking

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_user_partial(snap_compare) -> None:
    """An interim (still-being-spoken) user transcript grows in place while listening."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_partial("what's the weather like in")

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_tool_call_note(snap_compare) -> None:
    """A tool the agent uses mid-turn drops a dim progress note so the wait doesn't read as a hang."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_final("what's the weather like in Boston?")
        app.show_tool_call("Searching the web")

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_interrupted(snap_compare) -> None:
    """An interrupted reply is finalized and tagged `(interrupted)`, then returns to listening."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_final("tell me a long story")
        app.begin_reply()
        app.show_agent_sentence("Once upon a time, in a faraway land,")
        app.end_reply(interrupted=True)

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_error(snap_compare) -> None:
    """A cascade failure surfaces as a red ✗ error line in the transcript."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app._show_error("Streaming STT connection lost")

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)
