"""Tests for the `assembly live` write_todos plan panel (the voice TUI's checklist).

Drives the real Textual app headless: most tests call ``show_todos`` directly (it runs on the
UI thread), and one drives the worker leg through the real ``_TuiRenderer`` to cover the
off-thread ``todos_updated`` hop. Split from ``test_live_tui.py`` to keep that file under the
500-line gate. The plan-markup helper (``messages._todos_markup``) is unit-tested as a pure
function.
"""

from __future__ import annotations

import asyncio
import threading

from aai_cli.agent_cascade import banner, messages
from aai_cli.agent_cascade.messages import TodoList
from aai_cli.agent_cascade.plan import TodoItem
from aai_cli.agent_cascade.tui import LiveAgentApp


def _run(coro) -> None:
    asyncio.run(coro)


def _app(run_conversation=None, on_stop=None):
    """A LiveAgentApp whose worker blocks until teardown (mirrors test_live_tui._app)."""
    release = threading.Event()

    def stop() -> None:
        release.set()
        if on_stop is not None:
            on_stop()

    def block(renderer) -> None:
        release.wait(30)  # block like a live mic; teardown releases it well before this

    return LiveAgentApp(
        run_conversation=run_conversation or block,
        on_stop=stop,
        on_toggle_listen=lambda: True,
    )


def test_show_todos_mounts_a_plan_and_revises_it_in_place() -> None:
    # The first plan of a turn mounts one TodoList; a later revision within the same turn repaints
    # that same widget rather than stacking a second panel.
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_user_final("book a flight then check the weather")
            app.show_todos(
                (
                    TodoItem(content="Book a flight", status="in_progress"),
                    TodoItem(content="Check the weather", status="pending"),
                )
            )
            panels = list(app.query(TodoList))
            assert len(panels) == 1
            assert "Book a flight" in str(panels[0].render())
            # A revision marking the first task done updates the same panel in place.
            app.show_todos(
                (
                    TodoItem(content="Book a flight", status="completed"),
                    TodoItem(content="Check the weather", status="in_progress"),
                )
            )
            panels = list(app.query(TodoList))
            assert len(panels) == 1  # still one panel, revised in place
            assert panels[0] is app._todo_widget

    _run(go())


def test_show_todos_starts_a_fresh_panel_each_turn() -> None:
    # A new turn (show_user_final) resets the plan reference, so the next turn's plan mounts its
    # own panel instead of editing the prior turn's (which has scrolled up).
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_user_final("first task")
            app.show_todos((TodoItem(content="One", status="in_progress"),))
            app.show_user_final("second, unrelated task")
            app.show_todos((TodoItem(content="Two", status="in_progress"),))
            assert len(list(app.query(TodoList))) == 2

    _run(go())


def test_todos_updated_hops_to_the_ui_thread_from_the_worker() -> None:
    # The cascade runs on a worker thread; todos_updated must hop to the UI thread (via the real
    # _TuiRenderer) and mount the plan panel — the off-thread leg the direct calls above skip.
    done = threading.Event()

    def run_conversation(renderer) -> None:
        renderer.connected()
        renderer.user_final("plan it")
        renderer.todos_updated((TodoItem(content="Find it", status="in_progress"),))
        done.wait(30)

    async def go() -> None:
        app = _app(run_conversation=run_conversation, on_stop=done.set)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(200):
                await pilot.pause(0.01)
                if any("Find it" in str(p.render()) for p in app.query(TodoList)):
                    break
            assert any("Find it" in str(p.render()) for p in app.query(TodoList))
        assert done.is_set()

    _run(go())


def test_todos_markup_glyphs_and_completed_strikethrough() -> None:
    # The plan heading sits above one glyph-prefixed line per task; completed tasks are struck
    # through and the in-progress task is brand-accented, so the panel reads as a live checklist.
    text = messages._todos_markup(
        (
            TodoItem(content="Booked", status="completed"),
            TodoItem(content="Booking", status="in_progress"),
            TodoItem(content="Queued", status="pending"),
            TodoItem(content="Mystery", status="weird"),
        )
    )
    plain = text.plain
    assert plain.startswith("Plan")
    assert "✓ Booked" in plain  # completed glyph
    assert "▸ Booking" in plain  # in_progress glyph
    assert "○ Queued" in plain  # pending glyph
    assert "○ Mystery" in plain  # unknown status falls back to the pending glyph
    # The completed task's content carries a strikethrough; the in-progress one is brand-accented.
    styles = {span.style for span in text.spans}
    assert any("strike" in str(s) for s in styles)
    assert banner.BRAND_HEX in styles
