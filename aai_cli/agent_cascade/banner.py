"""The `assembly code` startup splash — the ASSEMBLY wordmark + a short intro.

Rendered once at session start (in the TUI transcript and the headless REPL). The
wordmark is the ANSI-Shadow block font; built from a per-letter map so the rows stay
aligned without hand-editing one giant string. The accent is the AssemblyAI brand blue.
"""

from __future__ import annotations

from aai_cli.ui import theme

# The wordmark accent — the AssemblyAI brand blue (Cobolt 400), as a hex literal so it
# renders identically in Rich and Textual without our theme being loaded.
BRAND_HEX = theme.BRAND

# Intro copy, shared by both front-ends so the wording stays in one place.
READY_LINE = "Ready to code! What would you like to build?"
TIP_LINE = "Tip: approve tools as they run, or pass --auto to skip the prompts."

# Each glyph is six rows tall (ANSI-Shadow). Only the letters in "ASSEMBLY" are needed.
_LETTERS: dict[str, list[str]] = {
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "M": ["███╗   ███╗", "████╗ ████║", "██╔████╔██║", "██║╚██╔╝██║", "██║ ╚═╝ ██║", "╚═╝     ╚═╝"],
    "B": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██████╔╝", "╚═════╝ "],
    "L": ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    "Y": ["██╗   ██╗", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚██╔╝  ", "   ██║   ", "   ╚═╝   "],
}
_ROWS = 6


def wordmark() -> list[str]:
    """The six plain rows of the ASSEMBLY block wordmark."""
    return [" ".join(_LETTERS[ch][row] for ch in "ASSEMBLY") for row in range(_ROWS)]


def version() -> str:
    """The CLI version string (e.g. ``v0.1.19``)."""
    from aai_cli import __version__

    return f"v{__version__}"
