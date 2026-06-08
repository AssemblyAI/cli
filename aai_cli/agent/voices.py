from __future__ import annotations

# Known Voice Agent voice IDs (from the Voice Agent quickstart). The server is
# the source of truth; this list backs --list-voices and catches obvious typos.
VOICES: list[str] = [
    # English
    "ivy",
    "james",
    "tyler",
    "winter",
    "sam",
    "mia",
    "bella",
    "david",
    "jack",
    "kyle",
    "helen",
    "martha",
    "river",
    "emma",
    "victor",
    "eleanor",
    "sophie",
    "oliver",
    # Multilingual
    "arjun",
    "ethan",
    "dmitri",
    "lukas",
    "lena",
    "pierre",
    "mina",
    "ren",
    "mei",
    "joon",
    "giulia",
    "luca",
    "lucia",
    "hana",
    "mateo",
    "diego",
]

DEFAULT_VOICE = "ivy"


def format_voice_list() -> str:
    """Human-readable, newline-separated voice IDs for --list-voices."""
    return "\n".join(VOICES)


def complete_voice(incomplete: str) -> list[str]:
    """Shell-completion callback for ``--voice``: known voice ids matching the prefix."""
    return [v for v in VOICES if v.startswith(incomplete)]
