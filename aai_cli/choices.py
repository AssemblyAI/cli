from __future__ import annotations

import enum

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
