"""Helpers for the Textual TUI visual-snapshot suite (``test_tui_snapshots.py``).

``pytest-textual-snapshot``'s ``snap_compare`` fixture renders a Textual ``App`` to an
SVG and diffs it against a committed golden, catching the CSS / layout / docking
regressions the behavioral pilot tests (``test_live_tui.py``) can't see — those assert
on one widget at a time, never the whole painted frame.

Two things make :class:`~aai_cli.agent_cascade.tui.LiveAgentApp` non-deterministic under
a raw render, so the goldens would churn or flake without neutralising them here:

* **The splash prints ``banner.version()``**, which hatch-vcs derives from the git tag
  (``v0.1.devN+g<sha>``) — a different string on every commit. ``pin_banner_version``
  freezes it.
* **The voice bar animates its meter on a 0.3s ``set_interval``.** How many times it has
  ticked by screenshot time depends on wall-clock scheduling, so the frame would differ
  run-to-run. :func:`freeze_animation` pins the meter to one frame and stops the timer.
* **``LiveAgentApp`` kicks the blocking cascade on a worker thread on mount**; if that
  worker returns it exits the app before the screenshot. :func:`build_live_app` returns a
  subclass whose ``_start`` is a no-op, so a snapshot drives the transcript directly with
  no thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App

from aai_cli.agent_cascade.tui import LiveAgentApp

if TYPE_CHECKING:
    import pytest

# A render width/height wide enough for the ASSEMBLY wordmark splash (~75 cells) plus a
# margin, shared by both apps so every golden is captured at the same terminal size.
TERMINAL_SIZE = (100, 30)
# A stable stand-in for banner.version() in the splash (the real string changes per commit).
_PINNED_VERSION = "v9.9.9"


class FakeAgent:
    """A no-op agent satisfying the ``CompiledAgent`` shape; a snapshot never runs a turn.

    ``invoke`` returns an empty state — it exists only so the type checks and the app can be
    constructed, and is covered by ``test_fake_agent_returns_empty_state`` rather than by any
    render (which deliberately never sends a turn).
    """

    def invoke(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {}


class _SnapshotLiveApp(LiveAgentApp):
    """``LiveAgentApp`` whose cascade worker never starts, so the app stays up for a render.

    The real ``_start`` runs the blocking conversation on a thread; in a snapshot we drive the
    transcript methods directly (see :func:`tests.test_tui_snapshots`), so starting the worker
    would only race the screenshot and exit the app the moment the no-op conversation returns.
    """

    def _start(self) -> None:
        pass


def build_live_app() -> _SnapshotLiveApp:
    """A ``LiveAgentApp`` whose cascade worker is stubbed out so a snapshot can drive it."""
    return _SnapshotLiveApp(
        run_conversation=lambda renderer: None,
        on_stop=lambda: None,
        on_toggle_listen=lambda: True,
    )


def freeze_animation(app: App[None]) -> None:
    """Stop every TUI animation timer so the captured frame is byte-stable.

    The voice bar's meter advances on a 0.3s ``set_interval``; left running, the number of
    ticks by screenshot time depends on wall-clock scheduling, so the frame would flake. Stop
    that timer — ``run_before`` is the first thing the screenshot harness runs, before any
    pause, so no tick fires before the stop, and the bar then holds the frame from its last
    explicit render (a fixed count per test).
    """
    assert isinstance(app, LiveAgentApp)
    if app._voice_timer is not None:
        app._voice_timer.stop()


def pin_banner_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the splash version string (otherwise it changes on every commit)."""
    monkeypatch.setattr("aai_cli.agent_cascade.banner.version", lambda: _PINNED_VERSION)
