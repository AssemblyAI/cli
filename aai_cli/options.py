"""Shared Typer option factories for flags every command repeats.

Centralizing them keeps the flag name, default, and help text uniform across
the ~26 command signatures instead of copy-pasting (and drifting) per command.
"""

from __future__ import annotations

import typer

from aai_cli import help_panels

DEFAULT_BATCH_CONCURRENCY = 4


def json_option(help_text: str = "Output raw JSON") -> bool:
    """The standard ``--json``/``-j`` flag; pass ``help_text`` where the shape differs."""
    flag: bool = typer.Option(False, "--json", "-j", help=help_text)
    return flag


def chars_per_caption_option() -> int | None:
    """The ``--chars-per-caption`` knob for the ``-o srt``/``-o vtt`` subtitle exports."""
    value: int | None = typer.Option(
        None,
        "--chars-per-caption",
        min=1,
        help="Max characters per caption line (only with -o srt or -o vtt)",
    )
    return value


# Batch-mode flags for `transcribe` (see transcribe_batch.py). Defined here because
# this module owns the FBT003 carve-out for Typer's boolean positional defaults.


def batch_from_stdin_option() -> bool:
    """The ``--from-stdin`` flag: batch mode fed one path/URL per stdin line."""
    flag: bool = typer.Option(
        False,
        "--from-stdin",
        help="Batch mode: read audio paths/URLs from stdin, one per line "
        "(composes with find/ls/yt-dlp output)",
        rich_help_panel=help_panels.OPT_BATCH,
    )
    return flag


def batch_concurrency_option() -> int:
    """The ``--concurrency`` option: how many sources transcribe at once in batch mode."""
    value: int = typer.Option(
        DEFAULT_BATCH_CONCURRENCY,
        "--concurrency",
        min=1,
        help="How many sources to transcribe at once in batch mode",
        rich_help_panel=help_panels.OPT_BATCH,
    )
    return value


def batch_force_option() -> bool:
    """The ``--force`` flag: re-transcribe even when a completed sidecar exists."""
    flag: bool = typer.Option(
        False,
        "--force",
        help="Batch mode: re-transcribe sources whose sidecar already records a completed run",
        rich_help_panel=help_panels.OPT_BATCH,
    )
    return flag
