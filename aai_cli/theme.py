from __future__ import annotations

from typing import IO, Any

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

# Cobolt — the AssemblyAI brand purple, taken from the marketing design system.
# The website's primary CTA is Cobolt 500; its dark-mode primary is Cobolt 300.
# A terminal's background is unknown (dark or light), so the CLI uses the mid step,
# Cobolt 400, as its accent: bright enough to read on black, saturated enough to read
# on white, and unambiguously the brand purple either way.
COBOLT_500 = "#3923c7"  # web primary CTA / brand purple
COBOLT_400 = "#614fd2"  # CLI primary accent (terminal-legible mid step)
COBOLT_300 = "#887bdd"  # web dark-mode primary / hover; CLI secondary accent

# Brand accent. Defined once so the whole CLI can be re-tinted here.
BRAND = COBOLT_400
# Secondary accent — the lighter Cobolt 300 used where a second hue is needed (agent
# label, links), so it stays in the brand family yet reads distinct from BRAND.
ACCENT = COBOLT_300

# Semantic error red, matching the design system's --color-error. Rich downsamples it
# to plain red on terminals without truecolor.
ERROR = "#F04438"

# Muted warm gray, matching the design system's --color-text-muted (Black 300). A true
# mid-gray that reads on both dark and light terminals; used for quiet labels (e.g. the
# help "Usage:" header) that should recede rather than carry the brand accent.
MUTED = "#777673"

# A fixed affordance vocabulary used across all human-facing output, mirroring the
# Vercel/Supabase convention of a small, consistent status alphabet: one glyph per
# meaning so a user learns it once. Unicode (the CLI already prints …/—); Rich
# re-encodes for terminals that can't render it.
SYMBOL_SUCCESS = "✓"
SYMBOL_ERROR = "✗"
SYMBOL_WARN = "!"
SYMBOL_HINT = "›"  # noqa: RUF001 — deliberate angle-quote glyph, not a '>' typo

# Per-speaker label colors, rotated deterministically by speaker_style(). Deliberately
# excludes the brand purple: that hue is reserved for "you" (aai.you) so a diarized system
# speaker can never be tinted the same color as your own mic.
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
        # Conversation labels: the human keeps the brand accent (reserved — never reused
        # for a diarized speaker, see SPEAKER_STYLES), the agent gets a distinct hue so
        # "you:" and "agent:" are easy to tell apart at a glance.
        "aai.you": BRAND,
        "aai.agent": ACCENT,
        # Links/URLs in the lighter Cobolt secondary accent so a clickable target stands
        # out from prose without shouting (Vercel/Supabase use a cool accent for the same).
        "aai.url": ACCENT,
        # Semantic status colors. Success is bold so the ✓ reads as a confident "done"
        # (Supabase-style); error uses the brand's semantic red, warn the universal yellow;
        # muted secondary text stays dim so it recedes (the Vercel "quiet by default" look).
        "aai.success": "bold green",
        "aai.error": f"bold {ERROR}",
        "aai.warn": "yellow",
        "aai.muted": "dim",
        "aai.speaker.0": "dark_orange",
        "aai.speaker.1": "cyan",
        "aai.speaker.2": "magenta",
        "aai.speaker.3": "green",
        "aai.speaker.4": "yellow",
    }
)

# Status strings grouped by the semantic style they render in.
_SUCCESS = {"completed", "installed", "removed", "ok", "present", "authenticated", "created"}
_ERROR = {"error", "failed"}
_WARN = {"queued", "processing", "in_progress", "running"}


class PipeSafeConsole(Console):
    """A Console honoring the CLI's closed-pipe contract.

    Rich's default ``on_broken_pipe`` redirects stdout to devnull and raises
    ``SystemExit(1)``, so ``assembly … | head`` would report failure. Re-raise the
    ``BrokenPipeError`` instead so the entry point (``main.run``) can treat a closed
    downstream pipe as success.
    """

    def on_broken_pipe(self) -> None:
        """Re-raise BrokenPipeError instead of swallowing it, so a closed downstream
        pipe propagates to ``run()``'s exit-0 handler rather than being hidden by Rich."""
        raise BrokenPipeError()


def make_console(file: IO[str] | None = None, **kwargs: Any) -> Console:
    """Build a Console with the AssemblyAI theme attached so `aai.*` names resolve."""
    return PipeSafeConsole(file=file, theme=THEME, **kwargs)


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


def status_text(status: str) -> Text:
    """A status string rendered in its semantic status color (see ``status_style``)."""
    return Text(status, style=status_style(status))
