"""Visual-regression snapshots for the `assembly live` Textual TUI.

Each test renders the app to an SVG via ``pytest-textual-snapshot``'s ``snap_compare``
fixture and diffs it against a committed golden under
``tests/__snapshots__/test_tui_snapshots/``. This pins the *painted frame* — the splash,
the voice bar, and the message widgets — so a CSS, layout, or docking regression that the
per-widget pilot tests (``test_live_tui.py``) can't see fails loudly here instead.

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
from tests import _tui_snapshot as h

if TYPE_CHECKING:
    from textual.pilot import Pilot


@pytest.fixture(autouse=True)
def _pin_version(monkeypatch: pytest.MonkeyPatch) -> None:
    h.pin_banner_version(monkeypatch)


def test_fake_agent_returns_empty_state() -> None:
    """The snapshot double satisfies CompiledAgent.invoke with an inert empty turn."""
    assert h.FakeAgent().invoke("prompt") == {}


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


def test_live_paused(snap_compare) -> None:
    """A muted mic (Space stops listening) shows a flat, non-animating meter and a grey
    `Paused` label, so a paused session reads as idle rather than actively listening."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app._listening = False  # Space muted the mic while idle -> the paused phase
        app._render_voicebar()

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)


def test_live_tool_call_note(snap_compare) -> None:
    """Tool calls mid-turn show the friendly label plus its identifying detail; the block is
    lifted off the prompt by a blank line, and a consecutive call stays tight beneath it."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_final("what's the weather like in Boston?")
        app.show_tool_call("Searching the web · Boston weather")
        app.show_tool_call("Using read_file · forecast.md")

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
