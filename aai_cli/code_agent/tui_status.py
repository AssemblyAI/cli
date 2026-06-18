"""Pure text helpers for the coding-agent TUI's status line and working indicator.

Split out of `tui.py` (to keep it under the file-length gate) and free of any Textual
imports, so they unit-test as plain functions.
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape

from aai_cli.ui import theme

# Animated meter for the voice bar — a 3-cell block-char pulse (BMP, single-width, no emoji).
# Public: both the `code` and `live` TUIs cycle it for their bar animation.
VOICE_FRAMES = ("▁▃▅", "▃▅▇", "▅▇▆", "▆▇▅", "▇▅▃", "▅▃▁")  # pragma: no mutate
# The voice phases the bar distinguishes, each (label, accent color). Shared by the `code`
# and `live` TUIs so both read the same: blue while listening, amber thinking, green speaking.
_VOICE_PHASES: dict[str, tuple[str, str]] = {
    "listening": ("Listening — speak your request", theme.BRAND),
    "thinking": ("Thinking…", "#f59e0b"),
    "speaking": ("Speaking…", "#22c55e"),
}


def voicebar_markup(phase: str, frame: str, *, hint: str = "") -> str:
    """The voice bar's content for one phase: an accented meter, the phase label, and a hint.

    ``hint`` is appended verbatim (already-marked-up trailing copy, e.g. a Ctrl-V tip); the
    label is escaped so a phase string can't inject styling.
    """
    label, color = _VOICE_PHASES[phase]
    return f"[{color}]{frame}[/] {escape(label)}{hint}"


def _spinner_text(elapsed_s: int, frame: str) -> str:
    """The working-indicator line: a spinner glyph and the elapsed seconds."""
    return f"{frame} Working… ({elapsed_s}s)"


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
    """The bottom status line: a mode badge, the working directory, git branch, and voice state.

    ``voice_state`` is ``"on"``/``"off"`` when the session has a voice front-end (so the
    Ctrl-V toggle shows its effect), or ``None`` when voice isn't wired up at all.
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
    return " ".join(parts)
