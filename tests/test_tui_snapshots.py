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

from aai_cli.agent_cascade.tui import LiveAgentApp
from aai_cli.code_agent.events import AssistantText, ToolCall, ToolResult
from aai_cli.code_agent.messages import UserMessage
from aai_cli.code_agent.modals import ApprovalScreen, AskScreen
from aai_cli.code_agent.tui import CodeAgentApp
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
