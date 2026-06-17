from __future__ import annotations

import enum
from collections.abc import Iterable


def complete_prefix(options: Iterable[str], incomplete: str) -> list[str]:
    """Shell-completion callback body: the ``options`` that start with the typed prefix."""
    return [option for option in options if option.startswith(incomplete)]


def render_grouped(groups: Iterable[tuple[str, Iterable[str]]]) -> str:
    """Render labelled groups of names as an indented, blank-line-separated block list.

    Each non-empty group renders as a ``label:`` header followed by its names
    indented two spaces, one per line; groups are separated by a blank line and
    empty groups are skipped. Backs the voice commands' grouped ``--list-voices``.
    """
    blocks: list[str] = []
    for label, names in groups:
        listing = "\n".join(f"  {name}" for name in names)
        if listing:
            blocks.append(f"{label}:\n{listing}")
    return "\n\n".join(blocks)


# CLI-owned closed value sets for ``-o/--output``. ``StrEnum`` members *are* their
# string values: Typer renders them as choices in ``--help`` (e.g.
# [text|id|status|utterances|srt|json]), validates input with a clean listing error,
# and completes them on Tab — while existing ``field == "text"`` comparisons and
# ``select_transcript_field(t, field)`` calls keep working unchanged.


class TranscriptOutput(enum.StrEnum):
    """Single-field output modes for a finished transcript (`transcribe`, `transcripts get`)."""

    text = "text"
    id = "id"
    status = "status"
    utterances = "utterances"
    srt = "srt"
    vtt = "vtt"
    json = "json"


class TextOrJson(enum.StrEnum):
    """Output mode for the streaming/LLM commands: plain finalized text or raw JSON."""

    text = "text"
    json = "json"


class Scope(enum.StrEnum):
    """Coding-agent config scope for `assembly setup` (passed through to `claude mcp add`)."""

    user = "user"
    project = "project"
    local = "local"


class ConfigKey(enum.StrEnum):
    """The settings `assembly config get/set` exposes (the persisted, non-secret ones)."""

    active_profile = "active_profile"
    env = "env"
    telemetry_enabled = "telemetry_enabled"


class ColorMode(enum.StrEnum):
    """The conventional tri-state for ANSI color (`--color`), matching git/gh/cargo."""

    auto = "auto"
    always = "always"
    never = "never"
