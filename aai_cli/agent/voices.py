from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    """A known Voice Agent voice id and the language group it belongs to."""

    name: str
    language: str


ENGLISH = "English"
MULTILINGUAL = "Multilingual"

# Known Voice Agent voice IDs (from the Voice Agent quickstart). The server is
# the source of truth; this list backs --list-voices and catches obvious typos.
VOICES: list[Voice] = [
    Voice("ivy", ENGLISH),
    Voice("james", ENGLISH),
    Voice("tyler", ENGLISH),
    Voice("winter", ENGLISH),
    Voice("sam", ENGLISH),
    Voice("mia", ENGLISH),
    Voice("bella", ENGLISH),
    Voice("david", ENGLISH),
    Voice("jack", ENGLISH),
    Voice("kyle", ENGLISH),
    Voice("helen", ENGLISH),
    Voice("martha", ENGLISH),
    Voice("river", ENGLISH),
    Voice("emma", ENGLISH),
    Voice("victor", ENGLISH),
    Voice("eleanor", ENGLISH),
    Voice("sophie", ENGLISH),
    Voice("oliver", ENGLISH),
    Voice("arjun", MULTILINGUAL),
    Voice("ethan", MULTILINGUAL),
    Voice("dmitri", MULTILINGUAL),
    Voice("lukas", MULTILINGUAL),
    Voice("lena", MULTILINGUAL),
    Voice("pierre", MULTILINGUAL),
    Voice("mina", MULTILINGUAL),
    Voice("ren", MULTILINGUAL),
    Voice("mei", MULTILINGUAL),
    Voice("joon", MULTILINGUAL),
    Voice("giulia", MULTILINGUAL),
    Voice("luca", MULTILINGUAL),
    Voice("lucia", MULTILINGUAL),
    Voice("hana", MULTILINGUAL),
    Voice("mateo", MULTILINGUAL),
    Voice("diego", MULTILINGUAL),
]

# The plain ids, for membership checks and completion.
VOICE_NAMES: list[str] = [voice.name for voice in VOICES]

DEFAULT_VOICE = "ivy"


def format_voice_list() -> str:
    """Human-readable voice IDs for --list-voices, grouped by language."""
    groups = dict.fromkeys(voice.language for voice in VOICES)
    blocks: list[str] = []
    for language in groups:
        names = "\n".join(f"  {voice.name}" for voice in VOICES if voice.language == language)
        blocks.append(f"{language}:\n{names}")
    return "\n\n".join(blocks)


def complete_voice(incomplete: str) -> list[str]:
    """Shell-completion callback for ``--voice``: known voice ids matching the prefix."""
    return [name for name in VOICE_NAMES if name.startswith(incomplete)]
