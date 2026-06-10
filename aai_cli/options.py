"""Shared Typer option factories for flags every command repeats.

Centralizing them keeps the flag name, default, and help text uniform across
the ~26 command signatures instead of copy-pasting (and drifting) per command.
"""

from __future__ import annotations

import typer


def json_option(help_text: str = "Output raw JSON.") -> bool:
    """The standard ``--json`` flag; pass ``help_text`` where the output shape differs."""
    flag: bool = typer.Option(False, "--json", help=help_text)
    return flag
