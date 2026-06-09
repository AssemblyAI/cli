from __future__ import annotations

from typing import IO, Any

from rich.console import Console
from rich.theme import Theme

# AssemblyAI brand accent. Defined once so the whole CLI can be re-tinted here.
BRAND = "#2545D3"
# Secondary accent — a darker blue used where a second hue is needed (agent label,
# links, the second speaker), kept in the blue family rather than the old cyan.
DARK_BLUE = "#1E3A8A"

# A fixed affordance vocabulary used across all human-facing output, mirroring the
# Vercel/Supabase convention of a small, consistent status alphabet: one glyph per
# meaning so a user learns it once. Unicode (the CLI already prints …/—); Rich
# re-encodes for terminals that can't render it.
SYMBOL_SUCCESS = "✓"
SYMBOL_ERROR = "✗"
SYMBOL_WARN = "!"
SYMBOL_HINT = "›"  # noqa: RUF001 — deliberate angle-quote glyph, not a '>' typo

# Per-speaker label colors, rotated deterministically by speaker_style().
SPEAKER_STYLES: tuple[str, ...] = (
    "aai.speaker.0",
    "aai.speaker.1",
    "aai.speaker.2",
    "aai.speaker.3",
    "aai.speaker.4",
)

THEME = Theme(
    {
        "aai.brand": f"bold {BRAND}",
        "aai.heading": f"bold {BRAND}",
        "aai.label": BRAND,
        # Conversation labels: the human keeps the brand accent, the agent gets a
        # distinct hue so "you:" and "agent:" are easy to tell apart at a glance.
        "aai.you": BRAND,
        "aai.agent": DARK_BLUE,
        # Links/URLs in the dark-blue secondary accent so a clickable target stands out
        # from prose without shouting (Vercel/Supabase use a cool accent for the same).
        "aai.url": DARK_BLUE,
        # Semantic status colors. Success is bold so the ✓ reads as a confident "done"
        # (Supabase-style); error/warn follow the universal red/yellow; muted secondary
        # text stays dim so it recedes (the Vercel "quiet by default" look).
        "aai.success": "bold green",
        "aai.error": "bold red",
        "aai.warn": "yellow",
        "aai.muted": "dim",
        "aai.speaker.0": BRAND,
        "aai.speaker.1": DARK_BLUE,
        "aai.speaker.2": "magenta",
        "aai.speaker.3": "green",
        "aai.speaker.4": "yellow",
    }
)

# Status strings grouped by the semantic style they render in.
_SUCCESS = {"completed", "installed", "removed", "ok", "present", "authenticated", "created"}
_ERROR = {"error", "failed"}
_WARN = {"queued", "processing", "in_progress", "running"}


def make_console(file: IO[str] | None = None, **kwargs: Any) -> Console:
    """Build a Console with the AssemblyAI theme attached so `aai.*` names resolve."""
    return Console(file=file, theme=THEME, **kwargs)


def speaker_style(speaker: object) -> str:
    """Deterministically map a speaker id to one of SPEAKER_STYLES."""
    key = str(speaker)
    idx = sum(ord(c) for c in key) % len(SPEAKER_STYLES)
    return SPEAKER_STYLES[idx]


def status_style(status: str) -> str:
    """Map a transcript/setup status to a semantic style name (muted if unknown)."""
    normalized = status.strip().lower()
    if normalized in _SUCCESS:
        return "aai.success"
    if normalized in _ERROR:
        return "aai.error"
    if normalized in _WARN:
        return "aai.warn"
    return "aai.muted"
