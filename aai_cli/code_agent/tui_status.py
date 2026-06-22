"""Pure text helpers for the coding-agent TUI's status line and working indicator.

Split out of `tui.py` (to keep it under the file-length gate) and free of any Textual
imports, so they unit-test as plain functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyperclip
from rich.markup import escape

from aai_cli.ui import theme

if TYPE_CHECKING:
    from collections.abc import Callable

# Animated meter for the voice bar — a 3-cell block-char pulse (BMP, single-width, no emoji).
# Public: both the `code` and `live` TUIs cycle it for their bar animation.
VOICE_FRAMES = ("▁▃▅", "▃▅▇", "▅▇▆", "▆▇▅", "▇▅▃", "▅▃▁")  # pragma: no mutate
# The at-rest meter shown while paused: a flat, non-animating frame (same width/alphabet as
# VOICE_FRAMES) so a muted mic reads as idle rather than as an active, pulsing meter.
VOICE_FLAT = "▁▁▁"
# The voice phases the bar distinguishes, each (label, accent color). Shared by the `code`
# and `live` TUIs so both read the same: blue while listening, amber thinking, green speaking.
_VOICE_PHASES: dict[str, tuple[str, str]] = {
    "listening": ("Listening — speak your request", theme.BRAND),
    "thinking": ("Thinking…", "#f59e0b"),
    "speaking": ("Speaking…", "#22c55e"),
    # `live`'s mic is muted (start/stop listening) — dimmed so a paused session reads as idle.
    "paused": ("Paused — press space to resume listening", "#6b7280"),
}


def voicebar_markup(phase: str, frame: str, *, hint: str = "") -> str:
    """The voice bar's content for one phase: an accented meter, the phase label, and a hint.

    ``hint`` is appended verbatim (already-marked-up trailing copy, e.g. a Ctrl-V tip); the
    label is escaped so a phase string can't inject styling.
    """
    label, color = _VOICE_PHASES[phase]
    if phase == "paused":
        frame = VOICE_FLAT  # a muted mic shows a flat meter, not the animated pulse it was handed
    return f"[{color}]{frame}[/] {escape(label)}{hint}"


def _spinner_text(elapsed_s: int, frame: str) -> str:
    """The working-indicator line: a spinner glyph and the elapsed seconds."""
    return f"{frame} Working… ({elapsed_s}s)"


def keyhints_text(*, voice: bool) -> str:
    """The dim key-legend footer for the `code` TUI — the shortcuts worth surfacing.

    The keyboard chords are otherwise undiscoverable (the app has no Footer widget). The
    Ctrl-V voice toggle is only listed when the session has a voice front-end. Caret notation
    (``^Y``) keeps the legend short enough to fit a narrow terminal; the chords are bold so
    they read against the dim labels.
    """
    hints = ["[b]^Y[/b] copy"]
    if voice:
        hints.append("[b]^V[/b] voice")
    hints += ["[b]^O[/b] expand", "[b]esc[/b] interrupt", "[b]^C[/b] quit"]
    return f"[dim]{' · '.join(hints)}[/dim]"


def copy_note(reply: str, copier: Callable[[str], None]) -> str:
    """Copy ``reply`` to the clipboard via ``copier``, returning the transcript note to show.

    Keeps the Ctrl-Y action a one-liner and handles its two non-happy paths so they can't
    surprise the user: nothing has been said yet, and a headless/clipboard-less box where
    ``pyperclip`` raises (an unhandled raise there would tear down the whole TUI). ``copier``
    is ``pyperclip.copy`` in production, injected so this unit-tests with no real clipboard.
    """
    if not reply:
        return "(nothing to copy yet)"
    try:
        copier(reply)
    except pyperclip.PyperclipException:
        return "(couldn't copy: no clipboard available)"
    return "(copied last reply to clipboard)"


def _abbrev_home(path: Path) -> str:
    """Render ``path`` with the home directory collapsed to ``~``."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _git_branch(start: Path) -> str | None:
    """The current git branch for ``start`` (walking up to the repo root), or None."""
    for directory in (start, *start.parents):
        head = directory / ".git" / "HEAD"
        if head.is_file():
            ref = head.read_text(encoding="utf-8").strip()
            return ref.removeprefix("ref: refs/heads/") if ref.startswith("ref: ") else ref[:8]
    return None


def _status_text(cwd: Path, *, auto_approve: bool, voice_state: str | None = None) -> str:
    """The two-row bottom footer: a status line, and a dim key-legend beneath it.

    Row one is a mode badge, the working directory, the git branch, and voice state; row two
    is :func:`keyhints_text`. ``voice_state`` is ``"on"``/``"off"`` when the session has a
    voice front-end (so the Ctrl-V toggle shows its effect, and the legend lists it), or
    ``None`` when voice isn't wired up at all.
    """
    mode = "auto" if auto_approve else "manual"
    badge = f"[black on #f59e0b] {mode} [/]"
    parts = [badge, f"[dim]{_abbrev_home(cwd)}[/dim]"]
    branch = _git_branch(cwd)
    if branch:
        parts.append(f"[dim]↗ {branch}[/dim]")
    if voice_state is not None:
        # A filled/hollow dot (BMP glyphs, like the rest of the UI — no double-width emoji).
        glyph, color = ("●", "#22c55e") if voice_state == "on" else ("○", "#6b7280")
        parts.append(f"[{color}]{glyph} voice {voice_state}[/]")
    return " ".join(parts) + "\n" + keyhints_text(voice=voice_state is not None)
